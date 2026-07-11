from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from src.common.event_exposure import EventExposureTracker
from src.common.reservation import TickerReservation
from src.trading.kalshi.learning import PerformanceTracker
from src.trading.kalshi.strategy import MispricingStrategy, Opportunity, Lock, event_key
from src.trading.kalshi.trading_client import KalshiTradingClient, Order, Position

logger = logging.getLogger(__name__)


@dataclass
class _PendingFill:
    """A resting (not-yet-confirmed-filled) order — a real bet is only
    recorded once we've confirmed real money actually moved, not just that
    an order was accepted. Maker orders posted at the mid routinely rest
    instead of filling immediately, and can later be cancelled unfilled —
    counting those as bets would record P&L for money that was never
    actually at risk (confirmed root cause of a real/paper P&L mismatch
    during the first live session: cancel_order was silently failing,
    leaving orders resting and eventually filling at stale prices, while
    record_bet had already logged them as settled bets regardless).

    Persisted to disk (see TradingBot._save_pending_fills) so a bet doesn't
    silently fall out of the learning data if the process restarts between
    an order resting and its fill being confirmed — the position itself is
    always safe either way (re-derived fresh from the exchange every scan),
    this only protects the calibration record of that one bet's outcome.
    """
    ticker: str
    series: str
    side: str
    entry_price: int
    predicted_win_rate: float
    count: int
    close_time: Optional[datetime]

    def to_json_dict(self) -> dict:
        d = asdict(self)
        d["close_time"] = self.close_time.isoformat() if self.close_time else None
        return d

    @classmethod
    def from_json_dict(cls, d: dict) -> "_PendingFill":
        close_time = d.get("close_time")
        return cls(
            ticker=d["ticker"],
            series=d["series"],
            side=d["side"],
            entry_price=d["entry_price"],
            predicted_win_rate=d["predicted_win_rate"],
            count=d["count"],
            close_time=datetime.fromisoformat(close_time) if close_time else None,
        )


def run_indexer_loop(
    stop_event: threading.Event,
    interval: int = 1800,
    ticker_prefixes: Optional[list] = None,
):
    """Fetch new Kalshi trades incrementally every `interval` seconds.

    Intended to run in a single background daemon thread shared across all bots.
    Runs immediately on startup, then waits `interval` seconds between runs.
    Always uses an incremental time window — never full backfill (which would
    scan 9M+ markets and run for days). Pass ticker_prefixes to restrict the
    scan to relevant series (e.g. crypto + macro) and avoid esports markets.
    """
    from src.indexers.kalshi.trades import KalshiTradesIndexer
    first_run = True
    while True:
        try:
            logger.info("Indexer: fetching new trades...")
            if first_run:
                # On startup use a 24-hour window to catch recent settlements.
                min_ts = int((datetime.now(timezone.utc) - timedelta(hours=24)).timestamp())
                first_run = False
            else:
                # Incremental: only fetch trades from the last 2 hours.
                min_ts = int((datetime.now(timezone.utc) - timedelta(hours=2)).timestamp())
            KalshiTradesIndexer(min_ts=min_ts, ticker_prefixes=ticker_prefixes).run()
            logger.info("Indexer: done.")
        except Exception as exc:
            logger.warning("Indexer error: %s", exc)
        if stop_event.wait(timeout=interval):
            break


class TradingBot:
    """Polls Kalshi on a fixed interval, places bets where edge exists,
    and cancels stale limit orders that no longer have edge.

    Args:
        client: Authenticated Kalshi trading client.
        strategy: Signal generator (mispricing strategy).
        poll_interval: Seconds between scans.
        max_open_positions: Maximum concurrent positions this bot manages.
        balance_fraction: Fraction of the total account balance this bot may
            spend (0.0–1.0). Use <1.0 when running multiple bots to prevent
            them from racing each other for the same dollars.
        series_list: If set, only fetch markets from these series prefixes.
            Used to give each bot a disjoint market universe so they can't
            bid on the same ticker simultaneously.
        win_rate_reload_interval: Seconds between win-rate reloads from disk.
    """

    def __init__(
        self,
        client: KalshiTradingClient,
        strategy: MispricingStrategy,
        poll_interval: int = 60,
        max_open_positions: int = 10,
        balance_fraction: float = 1.0,
        series_list: Optional[list[str]] = None,
        win_rate_reload_interval: int = 200,
        tracker: Optional[PerformanceTracker] = None,
        reservation: Optional[TickerReservation] = None,
        event_tracker: Optional[EventExposureTracker] = None,
        pending_fills_path: Optional[Path] = None,
    ):
        self.client = client
        self.strategy = strategy
        self.poll_interval = poll_interval
        self.max_open_positions = max_open_positions
        self.balance_fraction = balance_fraction
        self.series_list = series_list
        self.win_rate_reload_interval = win_rate_reload_interval
        self.tracker = tracker
        self.reservation = reservation
        self.event_tracker = event_tracker
        self._last_reload = 0.0
        self._dead_tickers: set[str] = set()
        self._last_daily_summary = 0.0
        # order_id -> _PendingFill, for orders placed but not yet confirmed
        # filled (see _PendingFill docstring).
        self._pending_fills: dict[str, _PendingFill] = {}
        self._pending_fills_path = pending_fills_path
        if self._pending_fills_path:
            self._load_pending_fills()

    def _load_pending_fills(self):
        path = self._pending_fills_path
        if not path or not path.exists():
            return
        try:
            raw = json.loads(path.read_text())
            self._pending_fills = {
                order_id: _PendingFill.from_json_dict(d) for order_id, d in raw.items()
            }
            if self._pending_fills:
                logger.info(
                    "Loaded %d pending fill(s) from a prior run — will be "
                    "reconciled against live positions on the first scan",
                    len(self._pending_fills),
                )
        except Exception as exc:
            logger.warning("Failed to load pending fills from %s: %s", path, exc)

    def _save_pending_fills(self):
        path = self._pending_fills_path
        if not path:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(
                {oid: pf.to_json_dict() for oid, pf in self._pending_fills.items()},
                indent=2,
            ))
            os.replace(tmp, path)
        except Exception as exc:
            logger.warning("Failed to save pending fills to %s: %s", path, exc)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _held_tickers(self) -> set[str]:
        held: set[str] = set()
        for pos in self.client.get_positions():
            held.add(pos.ticker)
        for order in self.client.get_open_orders():
            held.add(order.ticker)
        return held

    def _cancel_stale(self, open_orders: list[Order], opps: list[Opportunity]):
        live = {(o.market.ticker, o.side) for o in opps}
        for order in open_orders:
            if (order.ticker, order.side) not in live:
                logger.info("Cancelling stale order %s on %s", order.order_id, order.ticker)
                self.client.cancel_order(order.order_id)

    def _reconcile_pending_fills(self, positions: list[Position], open_orders: list[Order]) -> None:
        """Resolve orders placed on a prior cycle that were resting (not yet
        confirmed filled) at that time. Only call record_bet once we know
        real money actually moved — checked against the current live
        position, not just the fact that an order was once accepted.
        """
        if not self._pending_fills:
            return
        still_open_ids = {o.order_id for o in open_orders}
        net_by_ticker = {p.ticker: p.position for p in positions}
        resolved_any = False

        for order_id, pending in list(self._pending_fills.items()):
            if order_id in still_open_ids:
                continue  # still resting — check again next cycle
            del self._pending_fills[order_id]
            resolved_any = True
            net = net_by_ticker.get(pending.ticker, 0)
            filled = (pending.side == "yes" and net > 0) or (pending.side == "no" and net < 0)
            if not filled:
                logger.info(
                    "Order %s on %s never filled (cancelled/expired unfilled) — no bet recorded",
                    order_id, pending.ticker,
                )
                continue
            actual_count = min(pending.count, abs(net))
            logger.info(
                "Confirmed fill for %s on %s: %dx (of %d requested)",
                order_id, pending.ticker, actual_count, pending.count,
            )
            if self.tracker:
                try:
                    self.tracker.record_bet(
                        ticker=pending.ticker,
                        series=pending.series,
                        side=pending.side,
                        entry_price=pending.entry_price,
                        predicted_win_rate=pending.predicted_win_rate,
                        count=actual_count,
                        close_time=pending.close_time,
                    )
                except Exception as exc:
                    logger.error(
                        "CRITICAL: confirmed fill for %s on %s but record_bet failed — "
                        "learning data may be incomplete: %s",
                        order_id, pending.ticker, exc,
                    )

        if resolved_any:
            self._save_pending_fills()

    def _limit_price(self, opp: Opportunity) -> tuple[int, int]:
        """Return (limit_price_in_side_cents, yes_price_for_api) for a maker order.

        Posts at the midpoint between bid and ask instead of hitting the ask.
        Maker orders capture half the spread rather than paying it — research
        shows takers lose ~32% on average vs ~10% for makers.

        Returns (limit_cents, yes_price) where limit_cents is the price we pay
        in the bet's native side (used for cost/logging) and yes_price is the
        YES-equivalent price the Kalshi API expects.
        """
        if opp.side == "yes":
            mid = (opp.market.yes_bid + opp.market.yes_ask) // 2
            mid = max(1, min(99, mid))
            return mid, mid
        else:
            # no_bid (in NO-price terms) ≈ 100 - yes_ask
            no_bid_est = 100 - opp.market.yes_ask
            no_mid = (no_bid_est + opp.market.no_ask) // 2
            no_mid = max(1, min(99, no_mid))
            return no_mid, 100 - no_mid   # API needs YES-equivalent price

    def _place_orders(
        self, opps: list[Opportunity], held: set[str], slots: int, budget_cents: int
    ) -> int:
        placed = 0
        remaining = budget_cents
        for opp in opps:
            if placed >= slots:
                break
            if opp.market.ticker in held:
                continue
            if self.reservation and not self.reservation.try_reserve(opp.market.ticker):
                continue

            limit_cents, yes_price = self._limit_price(opp)
            cost = limit_cents * opp.count
            ekey = event_key(opp.market.ticker)
            if cost > remaining:
                logger.debug(
                    "Skipping %s: cost %d¢ > remaining budget %d¢",
                    opp.market.ticker, cost, remaining,
                )
                if self.reservation:
                    self.reservation.release(opp.market.ticker)
                continue

            if self.event_tracker and not self.event_tracker.try_reserve(ekey, cost):
                logger.debug(
                    "Skipping %s: event %s already at its daily exposure cap",
                    opp.market.ticker, ekey,
                )
                if self.reservation:
                    self.reservation.release(opp.market.ticker)
                continue

            try:
                order = self.client.place_order(
                    ticker=opp.market.ticker,
                    side=opp.side,
                    count=opp.count,
                    yes_price=yes_price,
                )
            except Exception as exc:
                # Order genuinely never went through — safe to give back both
                # the ticker claim and the event budget.
                if self.reservation:
                    self.reservation.release(opp.market.ticker)
                if self.event_tracker:
                    self.event_tracker.release(ekey, cost)
                body = getattr(getattr(exc, "response", None), "text", "")
                if "410" in str(exc):
                    self._dead_tickers.add(opp.market.ticker)
                    logger.debug("Market gone (410), blacklisting: %s", opp.market.ticker)
                elif "insufficient_balance" in body:
                    logger.debug("Insufficient balance for %s, stopping placement", opp.market.ticker)
                    break
                else:
                    logger.warning("Order failed on %s: %s  body=%s", opp.market.ticker, exc, body[:200])
                continue

            # Order placed successfully — from here on, a failure (e.g. a
            # transient file-write error in record_bet) must NOT release the
            # reservation/event budget, since the trade already happened and
            # doing so would silently reopen the ticker for a duplicate order
            # and under-count real exposure against the daily event cap.
            logger.info(
                "Placed %s on %s: %dx @ %d¢ (mid, ask was %d¢)  edge=%.1fpp  id=%s  status=%s",
                opp.side.upper(),
                opp.market.ticker,
                opp.count,
                limit_cents,
                opp.entry_price,
                opp.edge,
                order.order_id,
                order.status,
            )
            if order.status == "filled":
                # Matched immediately — real money moved right now, safe to
                # record as a confirmed bet.
                try:
                    if self.tracker:
                        self.tracker.record_bet(
                            ticker=opp.market.ticker,
                            series=self.strategy._ticker_series(opp.market.ticker),
                            side=opp.side,
                            entry_price=limit_cents,
                            predicted_win_rate=opp.win_rate,
                            count=opp.count,
                            close_time=opp.market.close_time,
                        )
                except Exception as exc:
                    logger.error(
                        "CRITICAL: order %s placed on %s but record_bet failed — "
                        "learning data may be incomplete: %s",
                        order.order_id, opp.market.ticker, exc,
                    )
            else:
                # Resting — this is the normal case for these maker orders
                # (posted at the mid, not crossing the spread). Don't record
                # a bet yet; _reconcile_pending_fills confirms on a later
                # cycle whether it actually filled before it counts.
                self._pending_fills[order.order_id] = _PendingFill(
                    ticker=opp.market.ticker,
                    series=self.strategy._ticker_series(opp.market.ticker),
                    side=opp.side,
                    entry_price=limit_cents,
                    predicted_win_rate=opp.win_rate,
                    count=opp.count,
                    close_time=opp.market.close_time,
                )
                self._save_pending_fills()
            held.add(opp.market.ticker)
            remaining -= cost
            placed += 1
        return remaining

    def _place_locks(self, locks: list[Lock], held: set[str], budget_cents: int):
        for lock in locks:
            if lock.market.ticker in held:
                continue
            if self.reservation and not self.reservation.try_reserve(lock.market.ticker):
                continue
            cost = (lock.yes_ask + lock.no_ask) * lock.count
            if cost > budget_cents:
                if self.reservation:
                    self.reservation.release(lock.market.ticker)
                continue
            yes_order = None
            try:
                yes_order = self.client.place_order(
                    ticker=lock.market.ticker,
                    side="yes",
                    count=lock.count,
                    yes_price=lock.yes_ask,
                )
                no_order = self.client.place_order(
                    ticker=lock.market.ticker,
                    side="no",
                    count=lock.count,
                    yes_price=100 - lock.no_ask,
                )
                logger.info(
                    "LOCK on %s: %dx YES@%d¢ + %dx NO@%d¢  profit=+%d¢ guaranteed",
                    lock.market.ticker,
                    lock.count, lock.yes_ask,
                    lock.count, lock.no_ask,
                    lock.profit_cents * lock.count,
                )
                held.add(lock.market.ticker)
                budget_cents -= cost
            except Exception as exc:
                if self.reservation:
                    self.reservation.release(lock.market.ticker)
                body = getattr(getattr(exc, "response", None), "text", "")
                if yes_order is not None:
                    # YES leg placed but NO leg failed — cancel YES to avoid naked position
                    logger.warning("Lock NO leg failed on %s, cancelling YES leg: %s", lock.market.ticker, exc)
                    try:
                        self.client.cancel_order(yes_order.order_id)
                    except Exception as cancel_exc:
                        logger.error(
                            "CRITICAL: YES lock-leg cancel FAILED on %s order %s — naked position: %s",
                            lock.market.ticker, yes_order.order_id, cancel_exc,
                        )
                else:
                    logger.warning("Lock YES leg failed on %s: %s  body=%s", lock.market.ticker, exc, body[:200])

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _scan_once(self):
        now = time.monotonic()

        if now - self._last_reload >= self.win_rate_reload_interval:
            logger.info("Reloading win rates from disk...")
            self.strategy.reload_win_rates()
            self._last_reload = now

        if not self.strategy.win_rates.get("yes") and not self.strategy.win_rates.get("no"):
            logger.warning("No historical data yet — run the indexer first.")
            return

        # Daily learning summary — log once every 24 hours
        if self.tracker and now - self._last_daily_summary >= 86400:
            logger.info(self.tracker.daily_summary())
            self._last_daily_summary = now

        balance_cents = self.client.get_balance()
        budget_cents = int(balance_cents * self.balance_fraction)

        # Fetch positions and orders once — reused for held set, slot calc, and cancel logic.
        positions    = self.client.get_positions()
        open_orders  = self.client.get_open_orders()
        held = {p.ticker for p in positions} | {o.ticker for o in open_orders}

        # Resolve any orders from a prior cycle that were resting (not yet
        # confirmed filled) at that time — must run before this cycle places
        # any new orders so pending state never grows unbounded.
        self._reconcile_pending_fills(positions, open_orders)

        # Count only positions AND resting orders in this bot's market universe
        # so two bots don't block each other's slot budget, and so resting limit
        # orders are counted toward the cap (not just filled positions).
        if self.series_list:
            own_positions = [
                p for p in positions
                if any(p.ticker.startswith(s) for s in self.series_list)
            ]
            own_open_orders = [
                o for o in open_orders
                if any(o.ticker.startswith(s) for s in self.series_list)
            ]
        else:
            own_positions = positions
            own_open_orders = open_orders
        slots = max(0, self.max_open_positions - len(own_positions) - len(own_open_orders))

        markets = self.client.list_liquid_open_markets(series_list=self.series_list)
        open_tickers = {m.ticker for m in markets}

        # Check if any pending bets have settled since last scan
        if self.tracker:
            n_settled = self.tracker.check_settlements(self.client, open_tickers)
            if n_settled:
                logger.info("Learning: settled %d bet(s) this cycle", n_settled)

        # held (real positions + resting orders) blocks whole events; dead
        # tickers (410-gone markets) only block themselves — a vanished
        # bracket must not lock out its siblings.
        opps = self.strategy.find_opportunities(
            markets, held | self._dead_tickers, event_blocked_tickers=held
        )

        logger.info(
            "[%s] balance=$%.2f (budget=$%.2f)  markets=%d  opps=%d  positions=%d  slots=%d",
            datetime.now().strftime("%H:%M:%S"),
            balance_cents / 100,
            budget_cents / 100,
            len(markets),
            len(opps),
            len(positions),
            slots,
        )

        # Only cancel stale orders when we have real market data — a network blip
        # returning 0 markets would otherwise wipe out all open orders.
        if markets:
            self._cancel_stale(open_orders, opps)

        remaining_budget = budget_cents
        if slots > 0 and opps and budget_cents > 0:
            remaining_budget = self._place_orders(opps, held, slots, budget_cents)

        # Arbitrage locks — scan every cycle regardless of slots (risk-free)
        locks = self.strategy.find_locks(markets, held | self._dead_tickers)
        if locks:
            logger.info("Found %d lock opportunities (guaranteed profit)", len(locks))
            self._place_locks(locks, held, remaining_budget)

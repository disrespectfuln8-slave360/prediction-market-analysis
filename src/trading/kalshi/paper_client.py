"""Paper-trading (dry-run) wrapper for KalshiTradingClient.

Every market-data call passes straight through to a real, authenticated
KalshiTradingClient — safe, since those are all read-only. Every
money-moving call (get_balance, get_positions, get_open_orders,
place_order, cancel_order) is intercepted and simulated against a local
JSON ledger instead of touching the real Kalshi account.

Fill model: honest about maker fills. An order that crosses the live book
(buy YES at >= yes_ask, buy NO at >= no_ask) fills immediately as a taker
at the touch price. Anything else RESTS, exactly like a real maker order,
and only fills when a later observation of the live market shows price
moving through it (yes_ask dropping to a resting YES bid, yes_bid rising
to a resting NO offer). This captures the adverse-selection cost the old
fill-instantly-at-mid model hid — resting orders disproportionately fill
when the market moves against you — which made paper P&L optimistic in a
way the first live session measurably confirmed. Still not modeled:
partial fills, book depth, and queue priority (we assume our order is
first at its price level), so paper remains mildly optimistic, but the
dominant error term is gone. Resting orders persist in the ledger across
restarts, show up in get_open_orders(), and are cancellable — so paper
mode drives the exact same resting/pending-fill code path in the bots as
live trading does.

Settlement is detected by polling the real market's resolution status for
every open simulated position, at most once every RECONCILE_INTERVAL
seconds; the same pass cancels (and refunds) resting orders whose market
has closed, mirroring exchange-side cancel-on-close.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

from src.indexers.kalshi.models import Market
from src.trading.kalshi.fees import maker_fee_cents, taker_fee_cents
from src.trading.kalshi.trading_client import KalshiTradingClient, Order, Position

logger = logging.getLogger(__name__)

DEFAULT_LEDGER_PATH = Path("data/kalshi/paper_ledger.json")
DEFAULT_STARTING_BALANCE_CENTS = 100_000  # $1,000 paper money
RECONCILE_INTERVAL = 20.0  # seconds between settlement checks of open paper positions


class PaperKalshiTradingClient:
    """Drop-in replacement for KalshiTradingClient that never places real orders."""

    def __init__(
        self,
        real_client: KalshiTradingClient | None = None,
        ledger_path: Path = DEFAULT_LEDGER_PATH,
        starting_balance_cents: int = DEFAULT_STARTING_BALANCE_CENTS,
    ):
        self._real = real_client or KalshiTradingClient()
        self._owns_real = real_client is None
        self.ledger_path = ledger_path
        self._lock = threading.Lock()
        self._last_reconcile = 0.0
        self._cash_cents = starting_balance_cents
        # ticker -> {"yes": count, "no": count}
        self._positions: dict[str, dict[str, int]] = {}
        # order_id -> {ticker, side, count, yes_price, placed_ts} — maker
        # orders resting at their posted price, cash already reserved.
        self._resting: dict[str, dict] = {}
        self._trade_log: list[dict] = []
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self):
        if self.ledger_path.exists():
            try:
                raw = json.loads(self.ledger_path.read_text())
                self._cash_cents = raw.get("cash_cents", self._cash_cents)
                self._positions = raw.get("positions", {})
                self._resting = raw.get("resting_orders", {})
                self._trade_log = raw.get("trade_log", [])
                logger.info(
                    "[paper] Loaded ledger: cash=$%.2f  open_tickers=%d  resting_orders=%d",
                    self._cash_cents / 100,
                    len(self._positions),
                    len(self._resting),
                )
            except Exception as exc:
                logger.warning("[paper] Failed to load ledger, starting fresh: %s", exc)

    def _save_locked(self):
        """Caller must hold self._lock."""
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "cash_cents": self._cash_cents,
                    "positions": self._positions,
                    "resting_orders": self._resting,
                    "trade_log": self._trade_log[-500:],  # bound file growth
                },
                indent=2,
            )
        )
        os.replace(tmp, self.ledger_path)

    # ── Context manager / lifecycle ──────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self._owns_real:
            self._real.close()

    # ── Market data (pass-through — read-only, safe) ─────────────────────────

    def get_market(self, ticker: str) -> Market:
        return self._real.get_market(ticker)

    def list_markets(self, limit: int = 200, status: str = "open"):
        return self._real.list_markets(limit=limit, status=status)

    def list_liquid_open_markets(self, min_volume_24h: int = 0, series_list: list[str] | None = None):
        markets = self._real.list_liquid_open_markets(min_volume_24h=min_volume_24h, series_list=series_list)
        # Piggyback on the bot's own market scan to check resting orders for
        # fills — fresh quotes at zero extra API cost, once per bot scan.
        self._sweep_resting(markets)
        return markets

    # ── Portfolio (simulated) ────────────────────────────────────────────────

    def get_balance(self) -> int:
        self._reconcile()
        with self._lock:
            return self._cash_cents

    def get_positions(self) -> list[Position]:
        self._reconcile()
        out = []
        with self._lock:
            for ticker, legs in self._positions.items():
                net = legs.get("yes", 0) - legs.get("no", 0)
                if net != 0:
                    out.append(Position(ticker=ticker, position=net))
        return out

    def get_open_orders(self) -> list[Order]:
        self._reconcile()
        with self._lock:
            return [
                Order(
                    order_id=oid,
                    ticker=o["ticker"],
                    side=o["side"],
                    action="buy",
                    count=o["count"],
                    yes_price=o["yes_price"],
                    status="resting",
                )
                for oid, o in self._resting.items()
            ]

    def _fill_locked(
        self,
        ticker: str,
        side: str,
        count: int,
        side_price: int,
        order_id: str,
        is_taker: bool,
    ):
        """Book a fill: charge the fee, update position, append to the log.
        Caller holds self._lock and has already accounted for the notional.

        Fees are charged here rather than ignored (as this client previously
        did) because the manual's §3 lists "the edge survives fees" as an
        assumption to be tested, and a paper ledger that trades for free
        cannot test it — it would clear every trade the live account would
        lose money on. Paper P&L is still mildly optimistic (no partial
        fills, no queue position), but not free.
        """
        fee = taker_fee_cents(side_price, count) if is_taker else maker_fee_cents(side_price, count)
        self._cash_cents -= fee
        leg = self._positions.setdefault(ticker, {"yes": 0, "no": 0})
        leg[side] = leg.get(side, 0) + count
        self._trade_log.append(
            {
                "order_id": order_id,
                "ticker": ticker,
                "side": side,
                "count": count,
                "price_cents": side_price,
                "fee_cents": fee,
                "role": "taker" if is_taker else "maker",
                "ts": time.time(),
            }
        )

    def place_order(self, ticker: str, side: str, count: int, yes_price: int) -> Order:
        # Reserve cash at the limit price, like the exchange holds margin for
        # a resting order. Refund the price improvement if we cross as taker.
        limit_side_price = yes_price if side == "yes" else (100 - yes_price)
        limit_side_price = max(1, min(99, limit_side_price))
        reserved = limit_side_price * count

        # Fresh top-of-book to decide taker-cross vs rest. If the quote fetch
        # fails, rest the order — a real order would sit on the book anyway.
        yes_bid = yes_ask = None
        try:
            market = self._real.get_market(ticker)
            yes_bid, yes_ask = market.yes_bid, market.yes_ask
        except Exception as exc:
            logger.debug("[paper] quote fetch failed for %s, order will rest: %s", ticker, exc)

        order_id = f"paper-{uuid.uuid4().hex[:12]}"
        with self._lock:
            if reserved > self._cash_cents:
                raise RuntimeError(f"insufficient_balance: need {reserved}c, have {self._cash_cents}c")
            self._cash_cents -= reserved

            if side == "yes":
                crosses = yes_ask is not None and 0 < yes_ask <= yes_price
                fill_side_price = yes_ask if crosses else None
            else:
                # Buying NO at no-price (100 - yes_price) crosses when the NO
                # ask (100 - yes_bid) is at or below it, i.e. yes_bid >= yes_price.
                crosses = yes_bid is not None and yes_bid >= yes_price
                fill_side_price = (100 - yes_bid) if crosses else None

            if crosses:
                # Taker fill at the touch — refund improvement vs our limit.
                self._cash_cents += (limit_side_price - fill_side_price) * count
                self._fill_locked(ticker, side, count, fill_side_price, order_id, is_taker=True)
                status = "filled"
                cash_now = self._cash_cents
            else:
                self._resting[order_id] = {
                    "ticker": ticker,
                    "side": side,
                    "count": count,
                    "yes_price": yes_price,
                    "placed_ts": time.time(),
                }
                status = "resting"
                cash_now = self._cash_cents
            self._save_locked()

        if status == "filled":
            logger.info(
                "[paper] FILLED (taker) %s %dx %s @ %d¢ (cash=$%.2f)",
                side.upper(),
                count,
                ticker,
                fill_side_price,
                cash_now / 100,
            )
        else:
            logger.info(
                "[paper] RESTING %s %dx %s @ %d¢ (reserved=%d¢, cash=$%.2f)",
                side.upper(),
                count,
                ticker,
                limit_side_price,
                reserved,
                cash_now / 100,
            )
        return Order(
            order_id=order_id,
            ticker=ticker,
            side=side,
            action="buy",
            count=count,
            yes_price=yes_price,
            status=status,
        )

    def cancel_order(self, order_id: str) -> bool:
        with self._lock:
            order = self._resting.pop(order_id, None)
            if order is None:
                # Already filled or never existed — same answer the real
                # exchange gives for a cancel that lost the race.
                return False
            side_price = order["yes_price"] if order["side"] == "yes" else (100 - order["yes_price"])
            self._cash_cents += max(1, min(99, side_price)) * order["count"]
            self._save_locked()
        logger.info("[paper] CANCELLED resting order %s on %s", order_id, order["ticker"])
        return True

    # ── Resting-order fills ───────────────────────────────────────────────────

    def _sweep_resting(self, markets) -> None:
        """Fill any resting order whose market has traded through it, using
        already-fetched market data (no extra API calls).

        A resting YES bid at P fills when the ask drops to/below P; a resting
        NO offer (yes_price P) fills when the bid rises to/above P. Fills are
        at the posted price — we were the standing order that got hit. This
        assumes we're first in queue at our level; real fills are strictly
        worse, never better.
        """
        if not self._resting:
            return
        by_ticker = {m.ticker: m for m in markets}
        filled_msgs = []
        with self._lock:
            for oid, o in list(self._resting.items()):
                m = by_ticker.get(o["ticker"])
                if m is None:
                    continue
                if o["side"] == "yes":
                    hit = m.yes_ask is not None and 0 < m.yes_ask <= o["yes_price"]
                    side_price = o["yes_price"]
                else:
                    hit = m.yes_bid is not None and m.yes_bid >= o["yes_price"]
                    side_price = 100 - o["yes_price"]
                if not hit:
                    continue
                del self._resting[oid]
                # Cash was already reserved at this exact price on placement.
                # We were the standing order that got hit, so this is a maker fill.
                self._fill_locked(o["ticker"], o["side"], o["count"], side_price, oid, is_taker=False)
                filled_msgs.append(
                    f"[paper] FILLED (maker) {o['side'].upper()} {o['count']}x "
                    f"{o['ticker']} @ {side_price}¢ — market traded through "
                    f"resting order {oid}"
                )
            if filled_msgs:
                self._save_locked()
        for msg in filled_msgs:
            logger.info(msg)

    # ── Settlement reconciliation ─────────────────────────────────────────────

    def _reconcile(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_reconcile) < RECONCILE_INTERVAL:
            return
        self._last_reconcile = now

        with self._lock:
            tickers = set(self._positions.keys())
            tickers |= {o["ticker"] for o in self._resting.values()}
        if not tickers:
            return

        for ticker in tickers:
            try:
                market = self._real.get_market(ticker)
            except Exception as exc:
                logger.debug("[paper] settlement check failed for %s: %s", ticker, exc)
                continue

            # Exchange cancels resting orders when a market leaves the live
            # book — refund their reserved cash.
            if market.status not in ("open", "active"):
                with self._lock:
                    dead = [(oid, o) for oid, o in self._resting.items() if o["ticker"] == ticker]
                    for oid, o in dead:
                        del self._resting[oid]
                        side_price = o["yes_price"] if o["side"] == "yes" else (100 - o["yes_price"])
                        self._cash_cents += max(1, min(99, side_price)) * o["count"]
                    if dead:
                        self._save_locked()
                for oid, _o in dead:
                    logger.info(
                        "[paper] EXPIRED resting order %s on %s (market %s) — never filled, cash refunded",
                        oid,
                        ticker,
                        market.status,
                    )

            if market.status != "finalized" or market.result not in ("yes", "no"):
                continue

            with self._lock:
                legs = self._positions.pop(ticker, None)
                if legs is None:
                    continue
                payout = legs.get(market.result, 0) * 100
                self._cash_cents += payout
                cash_now = self._cash_cents
                self._save_locked()

            logger.info(
                "[paper] SETTLED %s -> %s  payout=+%d¢  (held yes=%d no=%d)  cash=$%.2f",
                ticker,
                market.result.upper(),
                payout,
                legs.get("yes", 0),
                legs.get("no", 0),
                cash_now / 100,
            )

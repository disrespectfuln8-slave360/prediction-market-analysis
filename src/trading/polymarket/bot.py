from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Optional

from src.trading.polymarket.learning import PerformanceTracker
from src.trading.polymarket.strategy import MispricingStrategy, TradeOpp
from src.trading.polymarket.trading_client import Order, PolymarketTradingClient

logger = logging.getLogger(__name__)


def run_indexer_loop(stop_event: threading.Event, interval: int = 1800):
    """Keep local Polymarket trade data fresh via the incremental block-cursor indexer.

    Only re-runs the trades indexer, which resumes from its saved block cursor
    (cheap, bounded). PolymarketMarketsIndexer has no incremental mode — it does
    a full paginated re-fetch of every market — so it is NOT run here; re-running
    it every `interval` seconds would stall this loop for hours. Refresh market
    resolution data separately (e.g. `python -m src.indexers.polymarket.markets`)
    on whatever cadence fits.
    """
    from src.indexers.polymarket.trades import PolymarketTradesIndexer

    while True:
        try:
            logger.info("Indexer: fetching new Polymarket trades...")
            PolymarketTradesIndexer().run()
            logger.info("Indexer: done.")
        except Exception as exc:
            logger.warning("Indexer error: %s", exc)
        if stop_event.wait(timeout=interval):
            break


class TradingBot:
    """Polls Polymarket on a fixed interval, places bets where edge exists,
    and cancels stale limit orders that no longer have edge.

    Args:
        client: Authenticated Polymarket trading client.
        strategy: Signal generator (mispricing strategy).
        poll_interval: Seconds between scans.
        max_open_positions: Maximum concurrent positions this bot manages.
        balance_fraction: Fraction of the total account balance this bot may
            spend (0.0-1.0). Use <1.0 when running multiple bots so they don't
            race each other for the same dollars.
        tag_slugs: Gamma-api tag slugs used to scope market discovery (e.g.
            ["crypto"]). Positions/orders themselves carry no tag, so slot
            accounting below is intentionally account-wide, not per-bot —
            see the "DEDUP ACROSS BOTS" note in the module docs.
        win_rate_reload_interval: Seconds between win-rate reloads from disk.
    """

    def __init__(
        self,
        client: PolymarketTradingClient,
        strategy: MispricingStrategy,
        poll_interval: int = 30,
        max_open_positions: int = 10,
        balance_fraction: float = 1.0,
        tag_slugs: Optional[list[str]] = None,
        win_rate_reload_interval: int = 200,
        tracker: Optional[PerformanceTracker] = None,
    ):
        self.client = client
        self.strategy = strategy
        self.poll_interval = poll_interval
        self.max_open_positions = max_open_positions
        self.balance_fraction = balance_fraction
        self.tag_slugs = tag_slugs
        self.win_rate_reload_interval = win_rate_reload_interval
        self.tracker = tracker
        self._last_reload = 0.0
        self._dead_ids: set[str] = set()
        self._last_daily_summary = 0.0

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _outcome_side(self, opp: TradeOpp) -> str:
        """'yes' if betting the market's first outcome, else 'no'."""
        return "yes" if opp.outcome == opp.market.outcomes[0] else "no"

    def _cancel_stale(self, open_orders: list[Order], opps: list[TradeOpp]):
        live = {(o.market.condition_id, o.outcome) for o in opps}
        for order in open_orders:
            if (order.condition_id, order.outcome) not in live:
                logger.info("Cancelling stale order %s on %s", order.order_id, order.condition_id)
                if not self.client.cancel_order(order.order_id):
                    logger.error(
                        "Failed to cancel stale order %s on %s — may still be resting",
                        order.order_id, order.condition_id,
                    )

    def _limit_price(self, opp: TradeOpp) -> int:
        """Return a maker limit price at the bid/ask midpoint.

        Posting at the midpoint instead of hitting the ask captures half the
        spread instead of paying it.
        """
        market = opp.market
        if self._outcome_side(opp) == "yes":
            mid = (market.yes_bid + market.yes_ask) // 2
        else:
            no_bid_est = 100 - market.yes_ask
            mid = (no_bid_est + market.no_ask) // 2
        return max(1, min(99, mid))

    def _place_orders(
        self, opps: list[TradeOpp], held: set[str], slots: int, budget_cents: int
    ) -> int:
        placed = 0
        remaining = budget_cents
        for opp in opps:
            if placed >= slots:
                break
            if opp.market.condition_id in held:
                continue

            limit_cents = self._limit_price(opp)
            cost = limit_cents * opp.count
            if cost > remaining:
                logger.debug(
                    "Skipping %s: cost %d¢ > remaining budget %d¢",
                    opp.market.condition_id, cost, remaining,
                )
                continue

            try:
                order = self.client.place_order(
                    condition_id=opp.market.condition_id,
                    outcome=opp.outcome,
                    size=opp.count,
                    price=limit_cents,
                    side="BUY",
                )
                logger.info(
                    "Placed BUY %s on %s (%s): %sx @ %d¢ (mid, ask was %d¢)  edge=%.1fpp  id=%s",
                    opp.outcome, opp.market.condition_id, opp.market.question[:60],
                    opp.count, limit_cents, opp.entry_price, opp.edge, order.order_id,
                )
                if self.tracker:
                    self.tracker.record_bet(
                        condition_id=opp.market.condition_id,
                        series=self.strategy._market_series(opp.market),
                        side=self._outcome_side(opp),
                        entry_price=limit_cents,
                        predicted_win_rate=opp.win_rate,
                        count=opp.count,
                        close_time=opp.market.end_time,
                    )
                held.add(opp.market.condition_id)
                remaining -= cost
                placed += 1
            except Exception as exc:
                msg = str(exc)
                if "not found" in msg.lower() or "404" in msg:
                    self._dead_ids.add(opp.market.condition_id)
                    logger.debug("Market gone, blacklisting: %s", opp.market.condition_id)
                elif "insufficient" in msg.lower() or "balance" in msg.lower():
                    logger.debug("Insufficient balance for %s, stopping placement", opp.market.condition_id)
                    break
                else:
                    logger.warning("Order failed on %s: %s", opp.market.condition_id, exc)
        return remaining

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _scan_once(self):
        now = time.monotonic()

        if now - self._last_reload >= self.win_rate_reload_interval:
            logger.info("Reloading win rates from disk...")
            self.strategy.reload_win_rates()
            self._last_reload = now

        if not self.strategy.win_rates:
            logger.warning("No historical data yet — run the indexer first.")
            return

        if self.tracker and now - self._last_daily_summary >= 86400:
            logger.info(self.tracker.daily_summary())
            self._last_daily_summary = now

        balance_cents = self.client.get_balance()
        budget_cents = int(balance_cents * self.balance_fraction)

        # Fetch positions and orders once — account-wide (not scoped to this
        # bot's tag_slugs, since Position/Order carry no tag). Two bots on the
        # same account naturally see each other's positions this way.
        positions = self.client.get_positions()
        open_orders = self.client.get_open_orders()
        held = {p.condition_id for p in positions} | {o.condition_id for o in open_orders}

        slots = max(0, self.max_open_positions - len(positions) - len(open_orders))

        markets = self.client.list_liquid_markets(tag_slugs=self.tag_slugs)
        open_ids = {m.condition_id for m in markets}

        if self.tracker:
            n_settled = self.tracker.check_settlements(self.client, open_ids)
            if n_settled:
                logger.info("Learning: settled %d bet(s) this cycle", n_settled)

        opps = self.strategy.find_opportunities(markets, held | self._dead_ids)

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

        if markets:
            self._cancel_stale(open_orders, opps)

        if slots > 0 and opps and budget_cents > 0:
            self._place_orders(opps, held, slots, budget_cents)

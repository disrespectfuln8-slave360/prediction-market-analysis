from __future__ import annotations

import logging
import signal
import threading
import time
from datetime import datetime

from src.trading.kalshi.strategy import MispricingStrategy, Opportunity
from src.trading.kalshi.trading_client import KalshiTradingClient, Order

logger = logging.getLogger(__name__)


def _run_indexer_background(stop_event: threading.Event, interval: int = 1800):
    """Runs KalshiTradesIndexer incrementally every `interval` seconds."""
    from src.indexers.kalshi.trades import KalshiTradesIndexer
    while not stop_event.wait(timeout=interval):
        try:
            logger.info("Indexer: fetching new trades...")
            KalshiTradesIndexer().run()
            logger.info("Indexer: done.")
        except Exception as exc:
            logger.warning("Indexer error: %s", exc)


class TradingBot:
    """Polls Kalshi every poll_interval seconds, places bets where edge exists,
    and cancels stale limit orders that have drifted out of the opportunity set."""

    def __init__(
        self,
        client: KalshiTradingClient,
        strategy: MispricingStrategy,
        poll_interval: int = 60,
        max_open_positions: int = 20,
        win_rate_reload_interval: int = 3600,
    ):
        self.client = client
        self.strategy = strategy
        self.poll_interval = poll_interval
        self.max_open_positions = max_open_positions
        self.win_rate_reload_interval = win_rate_reload_interval
        self._running = False
        self._last_reload = 0.0

    # ── Helpers ──────────────────────────────────────────────────────────────

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

    def _place_orders(self, opps: list[Opportunity], held: set[str], slots: int):
        placed = 0
        for opp in opps:
            if placed >= slots:
                break
            if opp.market.ticker in held:
                continue

            # Kalshi always takes yes_price; for NO orders pass the YES-equivalent
            yes_price = opp.entry_price if opp.side == "yes" else (100 - opp.entry_price)

            try:
                order = self.client.place_order(
                    ticker=opp.market.ticker,
                    side=opp.side,
                    count=opp.count,
                    yes_price=yes_price,
                )
                logger.info(
                    "Placed %s order on %s: %dx @ %d¢  edge=%.1fpp  id=%s",
                    opp.side.upper(),
                    opp.market.ticker,
                    opp.count,
                    opp.entry_price,
                    opp.edge,
                    order.order_id,
                )
                held.add(opp.market.ticker)
                placed += 1
            except Exception as exc:
                logger.warning("Order failed on %s: %s", opp.market.ticker, exc)

    # ── Main loop ────────────────────────────────────────────────────────────

    def _scan_once(self):
        now = time.monotonic()

        if now - self._last_reload >= self.win_rate_reload_interval:
            logger.info("Reloading historical win rates...")
            self.strategy.reload_win_rates()
            self._last_reload = now

        if not self.strategy.win_rates:
            logger.warning("No historical data found. Run the Kalshi indexer first.")
            return

        held = self._held_tickers()
        open_orders = self.client.get_open_orders()
        positions = self.client.get_positions()
        slots = max(0, self.max_open_positions - len(positions))

        markets = self.client.list_all_open_markets()
        opps = self.strategy.find_opportunities(markets, held)

        logger.info(
            "[%s] markets=%d  opps=%d  positions=%d  open_orders=%d  slots=%d",
            datetime.now().strftime("%H:%M:%S"),
            len(markets),
            len(opps),
            len(positions),
            len(open_orders),
            slots,
        )

        self._cancel_stale(open_orders, opps)

        if slots > 0 and opps:
            self._place_orders(opps, held, slots)

    def run(self):
        """Start the polling loop. Blocks until SIGINT/SIGTERM.

        Also runs the trades indexer in a background thread every 30 minutes
        so win-rate data keeps improving while the bot is live.
        """
        self._running = True
        self._last_reload = 0.0
        stop_event = threading.Event()

        def _stop(sig, frame):
            logger.info("Shutting down...")
            self._running = False
            stop_event.set()

        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)

        indexer_thread = threading.Thread(
            target=_run_indexer_background,
            args=(stop_event,),
            daemon=True,
            name="indexer",
        )
        indexer_thread.start()

        logger.info("Bot started. Poll interval: %ds  Max positions: %d", self.poll_interval, self.max_open_positions)

        while self._running:
            try:
                self._scan_once()
            except Exception as exc:
                logger.error("Scan error: %s", exc, exc_info=True)

            if self._running:
                time.sleep(self.poll_interval)

        indexer_thread.join(timeout=5)
        logger.info("Bot stopped.")

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional

from src.trading.polymarket_us.strategy import LadderLock, find_ladder_locks

logger = logging.getLogger(__name__)


class LadderArbBot:
    """Scans polymarket.us for ladder-monotonicity violations and places
    guaranteed-profit lock trades (see strategy.py for why this needs no
    historical calibration — it's a math constraint, not a probability
    model). Both legs are taker-priced (cross the spread) so they fill
    immediately — resting maker orders would risk one leg filling without
    the other, leaving a naked position.
    """

    def __init__(
        self,
        client,
        poll_interval: int = 60,
        max_open_legs: int = 20,
    ):
        self.client = client
        self.poll_interval = poll_interval
        self.max_open_legs = max_open_legs

    def _held_slugs(self, positions, open_orders) -> set:
        return {p.market_slug for p in positions} | {o.market_slug for o in open_orders}

    def _place_lock(self, lock: LadderLock) -> None:
        """Place both legs of a lock. cancel_order never raises (it returns a
        bool) — its return value must be checked directly, not caught as an
        exception. And once a taker-priced leg reports status="filled",
        there is nothing left to "cancel": real money already moved, so a
        failure on the second leg from that point on is a naked position
        that needs a human, not a cancel call — logged CRITICAL rather than
        silently treated as recovered.
        """
        buy_order = None
        try:
            buy_order = self.client.place_order(
                market_slug=lock.buy_slug, side="yes",
                count=lock.count, yes_price=lock.buy_price,
            )
            if buy_order.status != "filled":
                # Didn't fill immediately — the bulk-scan quote it was priced
                # against had already moved. Nothing to hedge yet, so pull it
                # rather than risk it filling later at an unknown price.
                cancelled = self.client.cancel_order(buy_order.order_id, lock.buy_slug)
                if cancelled:
                    logger.warning(
                        "YES leg on %s (order %s) didn't fill immediately, cancelled — skipping lock",
                        lock.buy_slug, buy_order.order_id,
                    )
                else:
                    logger.error(
                        "CRITICAL: YES leg on %s order %s didn't fill and cancel failed — "
                        "may still be resting live, needs manual check",
                        lock.buy_slug, buy_order.order_id,
                    )
                return

            sell_order = self.client.place_order(
                market_slug=lock.sell_slug, side="no",
                count=lock.count, yes_price=lock.sell_price,
            )
            if sell_order.status != "filled":
                cancelled = self.client.cancel_order(sell_order.order_id, lock.sell_slug)
                logger.error(
                    "CRITICAL: NO leg on %s (order %s) did not fully fill (cancel %s) — "
                    "YES leg on %s already filled %dx, naked position, needs manual flatten",
                    lock.sell_slug, sell_order.order_id,
                    "succeeded" if cancelled else "FAILED — may still be resting live",
                    lock.buy_slug, lock.count,
                )
                return

            logger.info(
                "LOCK: %dx YES %s@%d¢ + %dx NO %s@%d¢  edge=+%d¢/ct guaranteed",
                lock.count, lock.buy_slug, lock.buy_price,
                lock.count, lock.sell_slug, 100 - lock.sell_price,
                lock.edge_cents,
            )
        except Exception as exc:
            if buy_order is not None and buy_order.status == "filled":
                logger.error(
                    "CRITICAL: YES leg on %s (order %s) filled but placing/checking the NO leg "
                    "on %s raised — naked position, needs manual flatten: %s",
                    lock.buy_slug, buy_order.order_id, lock.sell_slug, exc,
                )
            elif buy_order is not None:
                cancelled = self.client.cancel_order(buy_order.order_id, lock.buy_slug)
                logger.warning(
                    "Lock failed on %s (order %s, cancel %s): %s",
                    lock.buy_slug, buy_order.order_id,
                    "succeeded" if cancelled else "FAILED — may still be resting live, needs manual check",
                    exc,
                )
            else:
                logger.warning("Lock YES leg failed on %s: %s", lock.buy_slug, exc)

    def _scan_once(self):
        positions = self.client.get_positions()
        open_orders = self.client.get_open_orders()
        held = self._held_slugs(positions, open_orders)

        open_legs = len(positions) + len(open_orders)
        slots = max(0, (self.max_open_legs - open_legs) // 2)

        markets = self.client.list_all_markets(active=True, closed=False)
        locks = find_ladder_locks(markets, held)

        logger.info(
            "[%s] markets=%d  ladder_locks_found=%d  open_legs=%d  slots=%d",
            datetime.now().strftime("%H:%M:%S"), len(markets), len(locks), open_legs, slots,
        )

        for lock in locks[:slots]:
            self._place_lock(lock)

    def run_forever(self, stop_event=None):
        while True:
            try:
                self._scan_once()
            except Exception as exc:
                logger.warning("Ladder bot scan error: %s", exc)
            if stop_event is not None:
                if stop_event.wait(timeout=self.poll_interval):
                    break
            else:
                time.sleep(self.poll_interval)

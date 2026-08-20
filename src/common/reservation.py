"""Thread-safe ticker reservation shared across bots in one process.

Independent bot threads (e.g. the crypto TradingBot and ConvergenceBot in
run_all.py) each compute their own "held" set from a positions/orders
snapshot taken once per scan cycle. If their market universes overlap, two
threads can both see the same ticker as unheld and place competing orders on
it before either thread's next snapshot would reveal the conflict. A shared
TickerReservation closes that window: whichever thread reaches a ticker
first wins it for the rest of that ticker's lifetime as a live bet.
"""

from __future__ import annotations

import threading


class TickerReservation:
    def __init__(self):
        self._lock = threading.Lock()
        self._reserved: set[str] = set()

    def try_reserve(self, ticker: str) -> bool:
        """Atomically claim `ticker`. Returns True if newly claimed by this
        call, False if another thread already holds it."""
        with self._lock:
            if ticker in self._reserved:
                return False
            self._reserved.add(ticker)
            return True

    def release(self, ticker: str) -> None:
        """Give up a claim, e.g. after a failed order placement."""
        with self._lock:
            self._reserved.discard(ticker)

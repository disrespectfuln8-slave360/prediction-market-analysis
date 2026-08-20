"""Thread-safe per-event daily exposure cap, shared across bots in one process.

A single real-world event (one game, one tournament bracket, one crypto
close-time, one weather day) can surface as many separate markets — price
brackets, player props, spreads/totals, different threshold questions. Each
individual market can look independently attractive to a mispricing
strategy, but they are not independent bets: if they all resolve on the same
underlying outcome, betting several of them stacks correlated risk that no
existing per-ticker/per-bot check catches (TickerReservation only stops two
threads from both betting the *same* ticker, not different tickers tied to
the same event).

Caps total spend per event_key per calendar day (UTC), across every bot
sharing this tracker. Resets automatically at day rollover.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone


class EventExposureTracker:
    def __init__(self, max_event_cents: int):
        self.max_event_cents = max_event_cents
        self._lock = threading.Lock()
        self._spent: dict[str, int] = {}
        self._day = self._today()

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def _roll_day_if_needed_locked(self) -> None:
        """Caller must hold self._lock."""
        today = self._today()
        if today != self._day:
            self._day = today
            self._spent = {}

    def try_reserve(self, event_key: str, cost_cents: int) -> bool:
        """Atomically claim `cost_cents` of today's budget for `event_key`.

        Returns True if the event's daily cap has room for this cost (and
        reserves it), False if adding it would exceed max_event_cents.
        """
        with self._lock:
            self._roll_day_if_needed_locked()
            spent = self._spent.get(event_key, 0)
            if spent + cost_cents > self.max_event_cents:
                return False
            self._spent[event_key] = spent + cost_cents
            return True

    def release(self, event_key: str, cost_cents: int) -> None:
        """Give back reserved budget, e.g. after a failed order placement."""
        with self._lock:
            self._roll_day_if_needed_locked()
            remaining = self._spent.get(event_key, 0) - cost_cents
            self._spent[event_key] = max(0, remaining)

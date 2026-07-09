"""Adaptive learning tracker for the trading bot.

Records every placed bet and its eventual outcome. Computes per-series,
per-side (YES/NO), per-price-bucket performance deviations from predicted
win rates. When a combination underperforms, adds extra required edge for
future bets in that bucket — conservative by design (never loosens thresholds).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATA_PATH = Path("data/kalshi/bot_learning.json")

# Minimum settled bets before applying any adjustment to a bucket.
# Increased from 10 to 25 to reduce statistical noise (std error ~5.5% vs 14%).
MIN_BETS_FOR_ADJUSTMENT = 25

# Cap on how much extra edge we require (pp). Allows larger corrections when needed.
# Increased from 6.0 to 10.0 to match true underperformance (don't leave money on table).
MAX_EDGE_ADJUSTMENT_PP = 10.0


@dataclass
class BetRecord:
    ticker: str
    series: str
    side: str              # "yes" or "no"
    entry_price: int       # cents paid per contract
    predicted_win_rate: float
    count: int
    placed_at: str         # ISO timestamp
    close_time: Optional[str] = None
    settled_at: Optional[str] = None
    result: Optional[str] = None    # "win", "loss", or None (pending)
    profit_cents: Optional[int] = None


class PerformanceTracker:
    """Tracks bet outcomes and returns adaptive edge adjustments.

    Edge adjustments are keyed by (series, side, price_bucket) where
    price_bucket is the entry_price floored to the nearest 10¢.
    Adjustment = max(0, underperformance * 0.5), capped at MAX_EDGE_ADJUSTMENT_PP.
    """

    def __init__(self, path: Path = DATA_PATH):
        self.path = path
        self._bets: list[BetRecord] = []
        self._lock = threading.Lock()
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self):
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
                self._bets = [BetRecord(**b) for b in raw.get("bets", [])]
                logger.info("Learning: loaded %d bet records (%d pending, %d settled)",
                            len(self._bets),
                            sum(1 for b in self._bets if b.result is None),
                            sum(1 for b in self._bets if b.result is not None))
            except Exception as exc:
                logger.warning("Learning: failed to load data: %s", exc)
                self._bets = []

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"bets": [asdict(b) for b in self._bets]}, indent=2))
            os.replace(tmp, self.path)

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_bet(
        self,
        ticker: str,
        series: str,
        side: str,
        entry_price: int,
        predicted_win_rate: float,
        count: int,
        close_time: Optional[datetime] = None,
    ):
        rec = BetRecord(
            ticker=ticker,
            series=series,
            side=side,
            entry_price=entry_price,
            predicted_win_rate=predicted_win_rate,
            count=count,
            placed_at=datetime.now(timezone.utc).isoformat(),
            close_time=close_time.isoformat() if close_time else None,
        )
        with self._lock:
            self._bets.append(rec)
        self.save()
        logger.debug("Learning: recorded %s %s@%d¢ (predicted %.1f%%)",
                     ticker, side.upper(), entry_price, predicted_win_rate)

    # ── Settlement detection ──────────────────────────────────────────────────

    def check_settlements(self, client, current_open_tickers: set[str]) -> int:
        """Check pending bets that are no longer in the open market list.

        Only makes API calls for bets whose tickers have disappeared from the
        current scan (meaning the market closed/settled). This avoids burning
        rate-limit budget on still-open markets.
        """
        pending = [b for b in self._bets if b.result is None]
        if not pending:
            return 0

        # Also check bets where close_time is in the past by > 5 min,
        # even if we don't have a full market scan to compare against.
        now = datetime.now(timezone.utc)
        to_check = [
            b for b in pending
            if b.ticker not in current_open_tickers
            or (
                b.close_time
                and (now - datetime.fromisoformat(b.close_time)).total_seconds() > 300
            )
        ]
        if not to_check:
            return 0

        # Collect outcomes without holding the lock — API calls are slow and
        # the sibling bot thread may call record_bet() concurrently.
        updates: list[tuple] = []  # (bet, won: bool)
        for bet in to_check:
            try:
                market = client.get_market(bet.ticker)
                if market.status == "finalized" and market.result in ("yes", "no"):
                    updates.append((bet, market.result == bet.side))
            except Exception as exc:
                logger.debug("Settlement check failed for %s: %s", bet.ticker, exc)

        if not updates:
            return 0

        # Apply mutations atomically under the lock.
        with self._lock:
            for bet, won in updates:
                bet.result = "win" if won else "loss"
                bet.settled_at = now.isoformat()
                bet.profit_cents = (
                    (100 - bet.entry_price) * bet.count if won else -bet.entry_price * bet.count
                )

        settled = len(updates)
        for bet, _ in updates:
            logger.info(
                "Learning: %s %s@%d¢ → %s  P&L=%+d¢  (predicted %.1f%%)",
                bet.ticker, bet.side.upper(), bet.entry_price,
                bet.result.upper(), bet.profit_cents, bet.predicted_win_rate,
            )

        self.save()
        self._log_adjustment_table()
        return settled

    # ── Adaptive edge adjustment ──────────────────────────────────────────────

    def get_edge_adjustment(self, series: str, side: str, price: int) -> float:
        """Return extra edge (pp) required based on past performance in this bucket.

        Only returns positive values — never loosens the threshold.
        Returns 0.0 if fewer than MIN_BETS_FOR_ADJUSTMENT settled bets exist.
        """
        bucket = (price // 10) * 10

        relevant = [
            b for b in self._bets
            if b.result is not None
            and b.series == series
            and b.side == side
            and (b.entry_price // 10) * 10 == bucket
        ]

        if len(relevant) < MIN_BETS_FOR_ADJUSTMENT:
            return 0.0

        n_wins = sum(1 for b in relevant if b.result == "win")
        actual_wr = (n_wins / len(relevant)) * 100
        avg_predicted = sum(b.predicted_win_rate for b in relevant) / len(relevant)
        underperformance = avg_predicted - actual_wr

        if underperformance <= 0:
            return 0.0

        # Apply 85% of observed underperformance (conservative but realistic).
        # Was 0.5 (50%), which under-corrected — with N≥25, we can be more aggressive.
        adjustment = min(MAX_EDGE_ADJUSTMENT_PP, underperformance * 0.85)
        return round(adjustment, 2)

    # ── Summaries ─────────────────────────────────────────────────────────────

    def _log_adjustment_table(self):
        """Log current adjustments after any settlement."""
        settled = [b for b in self._bets if b.result is not None]
        if not settled:
            return

        # Build unique (series, side, bucket) combos with enough data
        seen: dict[tuple, list] = {}
        for b in settled:
            key = (b.series, b.side, (b.entry_price // 10) * 10)
            seen.setdefault(key, []).append(b)

        adjustments = []
        for (series, side, bucket), bets in sorted(seen.items()):
            if len(bets) < MIN_BETS_FOR_ADJUSTMENT:
                continue
            n_wins = sum(1 for b in bets if b.result == "win")
            actual_wr = (n_wins / len(bets)) * 100
            avg_pred = sum(b.predicted_win_rate for b in bets) / len(bets)
            adj = self.get_edge_adjustment(series, side, bucket)
            adjustments.append(
                f"  {series} {side.upper()} {bucket}-{bucket+9}¢: "
                f"pred={avg_pred:.1f}% actual={actual_wr:.1f}% "
                f"n={len(bets)} adj=+{adj:.1f}pp"
            )

        if adjustments:
            logger.info("Learning adjustments:\n%s", "\n".join(adjustments))

    def daily_summary(self) -> str:
        settled = [b for b in self._bets if b.result is not None]
        if not settled:
            return "Learning: no settled bets yet."

        now = datetime.now(timezone.utc)
        recent = [
            b for b in settled
            if b.settled_at
            and (now - datetime.fromisoformat(b.settled_at)).total_seconds() < 86400
        ]

        lines = [f"=== Daily Learning Summary ({len(recent)} bets settled today, {len(settled)} total) ==="]

        for series in sorted(set(b.series for b in recent)):
            sb = [b for b in recent if b.series == series]
            wins = sum(1 for b in sb if b.result == "win")
            pnl = sum(b.profit_cents for b in sb if b.profit_cents is not None)
            yes_b = [b for b in sb if b.side == "yes"]
            no_b = [b for b in sb if b.side == "no"]
            yes_w = sum(1 for b in yes_b if b.result == "win")
            no_w = sum(1 for b in no_b if b.result == "win")
            lines.append(
                f"  {series}: {wins}/{len(sb)} wins  P&L={pnl:+d}¢  "
                f"YES={yes_w}/{len(yes_b)}  NO={no_w}/{len(no_b)}"
            )

        return "\n".join(lines)

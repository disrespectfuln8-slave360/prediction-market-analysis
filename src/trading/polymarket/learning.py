"""Adaptive learning tracker for the Polymarket trading bot.

Identical design to src/trading/kalshi/learning.py: records every placed bet
and its eventual outcome, then computes per-series/per-side/per-price-bucket
performance deviations from predicted win rates so future bets in an
underperforming bucket require extra edge. Conservative by design — never
loosens thresholds.
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

DATA_PATH = Path("data/polymarket/learning.json")

MIN_BETS_FOR_ADJUSTMENT = 10
MAX_EDGE_ADJUSTMENT_PP = 6.0

# ── Category circuit-breaker (whole-series, coarser than the per-bucket
# edge adjustment above) ──────────────────────────────────────────────────
CATEGORY_ALL_LOSS_MIN = 4    # this many CONSECUTIVE (most recent) losses -> early block
CATEGORY_MIN_SAMPLE = 15     # settled bets needed before trusting a category's return rate
# A block has no other way to lift (no new bets get placed/recorded for a
# blocked category, so its stats can never improve on their own) — after
# this long with no new resolved bet, allow one fresh probe instead of
# blocking forever. If the underlying condition is still bad, a new losing
# streak re-blocks it almost immediately; if it's improved, real data can
# finally accumulate again.
CATEGORY_BLOCK_COOLDOWN_DAYS = 3

# ── Confidence-weighted sizing ────────────────────────────────────────────
CONFIDENCE_MIN_SAMPLE = 10   # settled bets needed before nudging size at all
UNTESTED_MULT = 0.90         # haircut for a category with zero resolved history
MIN_MULT = 0.75              # worst-case size haircut for a bad track record
MAX_MULT = 1.20              # best-case size boost for a good track record


@dataclass
class BetRecord:
    condition_id: str
    series: str
    side: str              # "yes" or "no"
    entry_price: int       # cents paid per share
    predicted_win_rate: float
    count: int
    placed_at: str          # ISO timestamp
    close_time: Optional[str] = None
    settled_at: Optional[str] = None
    result: Optional[str] = None    # "win", "loss", or None (pending)
    profit_cents: Optional[int] = None


class PerformanceTracker:
    """Tracks bet outcomes and returns adaptive edge adjustments.

    Edge adjustments are keyed by (series, side, price_bucket) where
    price_bucket is the entry_price floored to the nearest 10¢.
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
        condition_id: str,
        series: str,
        side: str,
        entry_price: int,
        predicted_win_rate: float,
        count: int,
        close_time: Optional[datetime] = None,
    ):
        rec = BetRecord(
            condition_id=condition_id,
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
                     condition_id, side.upper(), entry_price, predicted_win_rate)

    # ── Settlement detection ──────────────────────────────────────────────────

    def check_settlements(self, client, current_open_ids: set[str]) -> int:
        """Check pending bets whose markets are no longer in the open list.

        Only makes API calls for bets whose condition_id has disappeared from
        the current scan (meaning the market closed), or whose close_time is
        well in the past — avoids burning rate-limit budget on still-open markets.
        """
        pending = [b for b in self._bets if b.result is None]
        if not pending:
            return 0

        now = datetime.now(timezone.utc)
        to_check = [
            b for b in pending
            if b.condition_id not in current_open_ids
            or (
                b.close_time
                and (now - datetime.fromisoformat(b.close_time)).total_seconds() > 300
            )
        ]
        if not to_check:
            return 0

        # Collect outcomes without holding the lock — API calls are slow and
        # sibling bot threads may call record_bet() concurrently.
        updates: list[tuple] = []  # (bet, won: bool)
        for bet in to_check:
            try:
                data = client.get_market_resolution(bet.condition_id)
                if not data or not data.get("closed"):
                    continue
                tokens = data.get("tokens", []) or []
                if len(tokens) < 2:
                    continue
                won = (
                    (bet.side == "yes" and tokens[0].get("winner"))
                    or (bet.side == "no" and tokens[1].get("winner"))
                )
                resolved = any(t.get("winner") for t in tokens)
                if resolved:
                    updates.append((bet, bool(won)))
            except Exception as exc:
                logger.debug("Settlement check failed for %s: %s", bet.condition_id, exc)

        if not updates:
            return 0

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
                "Learning: %s %s@%d¢ -> %s  P&L=%+d¢  (predicted %.1f%%)",
                bet.condition_id, bet.side.upper(), bet.entry_price,
                bet.result.upper(), bet.profit_cents, bet.predicted_win_rate,
            )

        self.save()
        self._log_adjustment_table()
        return settled

    # ── Adaptive edge adjustment ──────────────────────────────────────────────

    def get_edge_adjustment(self, series: str, side: str, price: int) -> float:
        """Extra edge (pp) required based on past performance in this bucket.

        Only returns positive values — never loosens the threshold. Returns
        0.0 if fewer than MIN_BETS_FOR_ADJUSTMENT settled bets exist.
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

        adjustment = min(MAX_EDGE_ADJUSTMENT_PP, underperformance * 0.5)
        return round(adjustment, 2)

    # ── Category circuit-breaker ──────────────────────────────────────────────

    def is_category_blocked(self, series: str) -> Optional[str]:
        """Return a rejection reason if `series` has a proven bad track record
        across ALL sides/prices, or None if clear to bet.

        Coarser than get_edge_adjustment (whole series, not per side/price
        bucket) — catches a category that's losing everywhere, not just in
        one price band, which raising the required edge in individual
        buckets wouldn't necessarily stop.
        """
        relevant = [b for b in self._bets if b.result is not None and b.series == series]
        n = len(relevant)
        if n == 0:
            return None

        # Chronological order so "most recent" checks below look at the tail,
        # not an arbitrary dict/list order.
        ordered = sorted(relevant, key=lambda b: b.settled_at or b.placed_at)

        # No new bets get placed/recorded for an already-blocked category, so
        # without this, a block could never lift itself even after whatever
        # caused it is fixed. After a long enough gap since the last
        # settlement, allow one fresh probe instead of blocking forever.
        last_settled = ordered[-1].settled_at or ordered[-1].placed_at
        if last_settled:
            age_days = (
                datetime.now(timezone.utc) - datetime.fromisoformat(last_settled)
            ).total_seconds() / 86400
            if age_days >= CATEGORY_BLOCK_COOLDOWN_DAYS:
                return None

        # Most recent CATEGORY_ALL_LOSS_MIN results, all losses — a fast
        # early block on a live losing streak, regardless of any older win
        # elsewhere in the category's history.
        recent = ordered[-CATEGORY_ALL_LOSS_MIN:]
        if len(recent) >= CATEGORY_ALL_LOSS_MIN and all(b.result == "loss" for b in recent):
            return (
                f"{series}: last {len(recent)} resolved bets are all losses "
                f"— blocked"
            )

        if n >= CATEGORY_MIN_SAMPLE:
            total_pnl = sum(b.profit_cents or 0 for b in relevant)
            total_staked = sum(b.entry_price * b.count for b in relevant)
            return_rate = (total_pnl / total_staked) if total_staked > 0 else 0.0
            if return_rate < 0:
                return (
                    f"{series}: {n} resolved bets, {return_rate:+.1%} return "
                    f"— proven negative, blocked"
                )
        return None

    # ── Confidence-weighted sizing ────────────────────────────────────────────

    def confidence_multiplier(self, series: str) -> float:
        """Position-size multiplier (MIN_MULT-MAX_MULT) from this category's
        realized return rate. Asymmetric on purpose — punishes a bad track
        record harder than it rewards a good one, consistent with
        capital-preservation over chasing a hot streak.

        Independent of is_category_blocked — a caller should check both;
        this only ever nudges size, never gates whether to bet at all.
        """
        relevant = [b for b in self._bets if b.result is not None and b.series == series]
        n = len(relevant)
        if n == 0:
            return UNTESTED_MULT
        if n < CONFIDENCE_MIN_SAMPLE:
            return 1.0

        total_pnl = sum(b.profit_cents or 0 for b in relevant)
        total_staked = sum(b.entry_price * b.count for b in relevant)
        return_rate = (total_pnl / total_staked) if total_staked > 0 else 0.0

        if return_rate >= 0:
            mult = 1.0 + min(return_rate, 1.0) * (MAX_MULT - 1.0)
        else:
            mult = 1.0 + max(return_rate, -1.0) * (1.0 - MIN_MULT)
        return max(MIN_MULT, min(MAX_MULT, mult))

    # ── Summaries ─────────────────────────────────────────────────────────────

    def _log_adjustment_table(self):
        settled = [b for b in self._bets if b.result is not None]
        if not settled:
            return

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
                f"  {series} {side.upper()} {bucket}-{bucket+9}c: "
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
                f"  {series}: {wins}/{len(sb)} wins  P&L={pnl:+d}c  "
                f"YES={yes_w}/{len(yes_b)}  NO={no_w}/{len(no_b)}"
            )

        return "\n".join(lines)

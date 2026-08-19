"""Decile-bucketed price-level calibration for polymarket.us tennis
spread/totals markets.

Same architecture and reasoning as sports_prop_calibration.py (see that
module's docstring for the full case for decile-not-exact-cent bucketing) --
tennis has no independent forecast to calibrate against either, so this is
price-level calibration: "when this category has historically priced a
bucket at X, how often did it resolve YES?"

Scoped independently per category (never pooled across categories, never
sharing one trustworthiness flag between a bucket's two sides) -- category
here already encodes tour x {game,set} x {spreads,totals}, eight values,
see tennis_prop_quote_logger.py's docstring for the real evidence behind
that split (a 2026-08-18 finding: the original four-category version of
that module was itself silently pooling game- and set-based bets, the same
mistake this scoping exists to avoid, just not caught until now).

Data sources (written by tennis_prop_quote_logger.py):
    data/polymarket_us/tennis_prop_quotes.jsonl
    data/polymarket_us/tennis_prop_settlements.jsonl
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_QUOTES_PATH = Path("data/polymarket_us/tennis_prop_quotes.jsonl")
DEFAULT_SETTLEMENTS_PATH = Path("data/polymarket_us/tennis_prop_settlements.jsonl")

# Inherited from sports_prop_calibration.py's own validated figures --
# tennis is a new, smaller dataset with no independent validation of its
# own yet, so start from the same conservative defaults rather than guess.
MIN_SAMPLE_FLOOR = 30
MAX_CALIBRATION_ERROR_PCT = 10.0
STALE_DATA_DAYS = 3


def _decile_low(price_cents: int) -> int:
    """Same decile bucketing as sports_prop_calibration.py's own."""
    return min((max(0, min(100, price_cents)) // 10) * 10, 90)


@dataclass(frozen=True)
class TennisPropGrounding:
    category: str
    side: str  # "yes" or "no"
    price_cents: int
    implied_prob: float
    calibration_trustworthy: bool
    calibration_note: str
    calibration_sample_count: int
    calibration_actual_rate: Optional[float]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _build_calibration_table(quotes: list[dict], settlements: list[dict]) -> dict[str, dict[str, dict[int, dict]]]:
    """category -> {"yes": {decile: {count, wins, actual_rate}}, "no": {...}}
    Priced off each slug's FIRST recorded ask/bid, same proxy-for-entry-price
    reasoning as sports_prop_calibration.py's identical function."""
    first_ask: dict[str, int] = {}
    first_bid: dict[str, int] = {}
    for q in quotes:
        slug = q.get("slug")
        if slug is None:
            continue
        if q.get("ask") is not None and slug not in first_ask:
            first_ask[slug] = q["ask"]
        if q.get("bid") is not None and slug not in first_bid:
            first_bid[slug] = q["bid"]

    settle_by_slug = {s["slug"]: s for s in settlements if "slug" in s}

    table: dict[str, dict[str, dict[int, dict]]] = {}

    def _bucket(category: str, side: str, price: int, won: bool) -> None:
        cat = table.setdefault(category, {"yes": {}, "no": {}})
        decile = _decile_low(price)
        d = cat[side].setdefault(decile, {"count": 0, "wins": 0})
        d["count"] += 1
        d["wins"] += 1 if won else 0

    for slug, s in settle_by_slug.items():
        category = s.get("category", "")
        outcome_yes = s.get("outcome_yes")
        if outcome_yes is None:
            continue

        ask = first_ask.get(slug)
        if ask is not None:
            _bucket(category, "yes", ask, won=outcome_yes)

        bid = first_bid.get(slug)
        if bid is not None:
            no_price = 100 - bid
            _bucket(category, "no", no_price, won=not outcome_yes)

    for cat in table.values():
        for side_table in (cat["yes"], cat["no"]):
            for decile, d in side_table.items():
                d["actual_rate"] = 100.0 * d["wins"] / d["count"] if d["count"] else 0.0

    return table


class TennisPropCalibrationEngine:
    def __init__(
        self,
        quotes_path: Path = DEFAULT_QUOTES_PATH,
        settlements_path: Path = DEFAULT_SETTLEMENTS_PATH,
    ):
        self.quotes_path = quotes_path
        self.settlements_path = settlements_path
        self._table: dict[str, dict[str, dict[int, dict]]] = {}
        self._last_settlement_ts: Optional[str] = None
        self.reload()

    def reload(self) -> None:
        quotes = _read_jsonl(self.quotes_path)
        settlements = _read_jsonl(self.settlements_path)
        self._table = _build_calibration_table(quotes, settlements)
        timestamps = [s["resolved_ts"] for s in settlements if s.get("resolved_ts")]
        self._last_settlement_ts = max(timestamps) if timestamps else None
        logger.info(
            "[tennis_prop_calibration] Loaded %d settlements across %d categories, latest %s",
            len(settlements), len(self._table), self._last_settlement_ts,
        )

    def data_is_stale(self, now: Optional[datetime] = None) -> bool:
        if self._last_settlement_ts is None:
            return True
        now = now or datetime.now(timezone.utc)
        try:
            latest = datetime.fromisoformat(self._last_settlement_ts.replace("Z", "+00:00"))
        except ValueError:
            return True
        return (now - latest).days > STALE_DATA_DAYS

    def _trustworthiness(self, category: str, side: str, price_cents: int) -> tuple[bool, str, int, Optional[float]]:
        cat = self._table.get(category)
        decile = _decile_low(price_cents)
        if cat is None:
            return False, f"no calibration data at all for {category}", 0, None
        info = cat[side].get(decile)
        if info is None or info["count"] < MIN_SAMPLE_FLOOR:
            count = info["count"] if info else 0
            return False, (
                f"only {count} historical {side} sample(s) in the {decile}-{decile + 10}% "
                f"range for {category} (need {MIN_SAMPLE_FLOOR})"
            ), count, (info["actual_rate"] if info else None)
        midpoint = decile + 5
        error = abs(info["actual_rate"] - midpoint)
        if error > MAX_CALIBRATION_ERROR_PCT:
            return False, (
                f"{category} {side} measured {error:.1f}pp off the {decile}-{decile + 10}% "
                f"range's midpoint ({info['count']} samples) — exceeds the "
                f"{MAX_CALIBRATION_ERROR_PCT:.0f}pp tolerance"
            ), info["count"], info["actual_rate"]
        return True, (
            f"calibration OK: {info['count']} samples, actual rate {info['actual_rate']:.1f}% "
            f"in the {decile}-{decile + 10}% range"
        ), info["count"], info["actual_rate"]

    def evaluate(self, category: str, side: str, price_cents: int) -> TennisPropGrounding:
        """side is "yes" (cost = price_cents) or "no" (cost = 100 - yes_bid,
        already converted by the caller before price_cents is passed in)."""
        trustworthy, note, count, actual_rate = self._trustworthiness(category, side, price_cents)
        return TennisPropGrounding(
            category=category, side=side, price_cents=price_cents,
            implied_prob=price_cents / 100.0,
            calibration_trustworthy=trustworthy, calibration_note=note,
            calibration_sample_count=count, calibration_actual_rate=actual_rate,
        )

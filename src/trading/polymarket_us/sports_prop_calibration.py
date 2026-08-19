"""Decile-bucketed price-level calibration for polymarket.us MLB/NFL player
props.

Weather has NOAA as an independent forecast to calibrate against; nothing
equivalent exists for player-prop outcomes (see sports_prop_quote_logger.py's
module docstring). The only calibration this platform can support is
price-level: "when this category has historically been priced around X%,
how often did it actually resolve YES?" -- the same shape as Kalshi's
MispricingStrategy win-rate table, not Apex's NOAA-grounded weather_model.py.

Decile buckets (10-wide price bands), not exact price points, because
per-cent granularity needs far more data than a few hours of collection
produces. Checked live 2026-08-15 against sports_prop_quote_logger.py's
first night of real data: decile bucketing already cleared the sample floor
for several MLB categories (hits: 5/10 deciles, hrr: 6/10, total bases:
4/10); exact-price-point bucketing cleared at most 2/10 for any category.
Kalshi itself uses exact-cent buckets because its public trade history is
large enough to fill them -- this dataset isn't there yet, and may never
need to be if deciles keep working.

Scoped independently per (sport, stat), and independently per side (YES vs
NO) within each -- never pooled across stat types, and never sharing one
trustworthiness flag between a bucket's two sides. Both are lessons from
real bugs found this same session: pooling MLB totals/spreads into Kalshi's
"macro" category hid a 15-31pp miscalibration in both halves, and Apex's
per-side gating bug let a NO-side bet run on an unverified decile because
the YES side's decile happened to be trustworthy. Building this system
without either mistake from the start, now that both are known.

Data sources (written by sports_prop_quote_logger.py):
    data/polymarket_us/sports_prop_quotes.jsonl
    data/polymarket_us/sports_prop_settlements.jsonl
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_QUOTES_PATH = Path("data/polymarket_us/sports_prop_quotes.jsonl")
DEFAULT_SETTLEMENTS_PATH = Path("data/polymarket_us/sports_prop_settlements.jsonl")

# Matches the floor already checked live against this dataset (see module
# docstring) -- not Apex's MIN_SAMPLE_FLOOR=25, this is a different data
# source with different density, and 30 is what the real numbers above were
# validated against.
MIN_SAMPLE_FLOOR = 30

# Same tolerance Apex uses for weather -- not re-derived for this dataset
# yet, just inherited as the same conservative default until there's reason
# to tune it differently.
MAX_CALIBRATION_ERROR_PCT = 10.0

# If neither file has been touched in this many days, treat calibration as
# stale rather than trade on a table that may no longer reflect current
# game/pricing conditions. Mirrors Apex's STALE_HISTORY_DAYS.
STALE_DATA_DAYS = 3


def _decile_low(price_cents: int) -> int:
    """Same decile bucketing as Apex's weather_model.py._bucket_low_pct."""
    return min((max(0, min(100, price_cents)) // 10) * 10, 90)


@dataclass(frozen=True)
class SportsPropGrounding:
    sport: str
    stat: str
    side: str  # "yes" or "no"
    price_cents: int
    implied_prob: float  # price_cents / 100 (yes) or (100 - price_cents) / 100 (no) -- same as the side's cost fraction, ungrossed of fees
    calibration_trustworthy: bool
    calibration_note: str
    calibration_sample_count: int
    calibration_actual_rate: Optional[float]  # measured win rate in this decile, 0-100, for display


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


def _build_calibration_table(
    quotes: list[dict], settlements: list[dict]
) -> dict[tuple[str, str], dict[str, dict[int, dict]]]:
    """(sport, stat) -> {"yes": {decile: {count, actual_rate}}, "no": {...}}

    Priced off each slug's FIRST recorded ask (yes cost) / first recorded
    bid (used to derive no cost = 100 - bid) -- a simple proxy for entry
    price, not a true weighted-average or game-start-relative price. Good
    enough to validate the bucketing approach; worth refining once there's
    a reason to believe entry-price choice is the binding constraint rather
    than sample count.
    """
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

    table: dict[tuple[str, str], dict[str, dict[int, dict]]] = {}

    def _bucket(key: tuple[str, str], side: str, price: int, won: bool) -> None:
        cat = table.setdefault(key, {"yes": {}, "no": {}})
        decile = _decile_low(price)
        d = cat[side].setdefault(decile, {"count": 0, "wins": 0})
        d["count"] += 1
        d["wins"] += 1 if won else 0

    for slug, s in settle_by_slug.items():
        key = (s.get("sport", ""), s.get("stat", ""))
        outcome_yes = s.get("outcome_yes")
        if outcome_yes is None:
            continue

        ask = first_ask.get(slug)
        if ask is not None:
            _bucket(key, "yes", ask, won=outcome_yes)

        bid = first_bid.get(slug)
        if bid is not None:
            no_price = 100 - bid
            _bucket(key, "no", no_price, won=not outcome_yes)

    # Derive actual_rate from count/wins.
    for cat in table.values():
        for side_table in (cat["yes"], cat["no"]):
            for decile, d in side_table.items():
                d["actual_rate"] = 100.0 * d["wins"] / d["count"] if d["count"] else 0.0

    return table


class SportsPropCalibrationEngine:
    def __init__(
        self,
        quotes_path: Path = DEFAULT_QUOTES_PATH,
        settlements_path: Path = DEFAULT_SETTLEMENTS_PATH,
        extra_sources: tuple[tuple[Path, Path], ...] = (),
    ):
        """extra_sources: additional (quotes_path, settlements_path) pairs to
        merge in alongside the primary pair -- e.g. nfl_prop_quote_logger.py's
        own dedicated files, split out from the shared sports_prop_*.jsonl on
        2026-08-17 so NFL has its own data (see that module's docstring).
        Safe to merge: _build_calibration_table keys purely on each ROW's own
        (sport, stat) fields, never on which file it came from, so combining
        sources here doesn't re-introduce the pooling this class exists to
        avoid -- it's the same table a single logger covering every sport
        would have produced.

        Staleness (data_is_stale) is computed across the POOLED settlement
        timestamps -- stale only if EVERY source has gone quiet, not each
        independently. Simple on purpose: this flag exists to catch "is the
        collection pipeline broadly alive", not to track each sport's own
        cadence separately, and a sport with genuinely no games running
        (e.g. NFL in the off-season) isn't "stale", it's just quiet."""
        self.quotes_path = quotes_path
        self.settlements_path = settlements_path
        self.extra_sources = extra_sources
        self._table: dict[tuple[str, str], dict[str, dict[int, dict]]] = {}
        self._last_settlement_ts: Optional[str] = None
        self.reload()

    def reload(self) -> None:
        quotes = _read_jsonl(self.quotes_path)
        settlements = _read_jsonl(self.settlements_path)
        for extra_quotes_path, extra_settlements_path in self.extra_sources:
            quotes += _read_jsonl(extra_quotes_path)
            settlements += _read_jsonl(extra_settlements_path)
        self._table = _build_calibration_table(quotes, settlements)
        timestamps = [s["resolved_ts"] for s in settlements if s.get("resolved_ts")]
        self._last_settlement_ts = max(timestamps) if timestamps else None
        n_settlements = len(settlements)
        n_categories = len(self._table)
        logger.info(
            "[sports_prop_calibration] Loaded %d settlements across %d (sport, stat) categories, latest %s",
            n_settlements, n_categories, self._last_settlement_ts,
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

    def _trustworthiness(self, sport: str, stat: str, side: str, price_cents: int) -> tuple[bool, str, int, Optional[float]]:
        cat = self._table.get((sport, stat))
        decile = _decile_low(price_cents)
        if cat is None:
            return False, f"no calibration data at all for {sport}/{stat}", 0, None
        info = cat[side].get(decile)
        if info is None or info["count"] < MIN_SAMPLE_FLOOR:
            count = info["count"] if info else 0
            return False, (
                f"only {count} historical {side} sample(s) in the {decile}-{decile + 10}% "
                f"range for {sport}/{stat} (need {MIN_SAMPLE_FLOOR})"
            ), count, (info["actual_rate"] if info else None)
        # Bucket midpoint is the best single number to compare the measured
        # rate against, same convention as Apex's decile calibration check.
        midpoint = decile + 5
        error = abs(info["actual_rate"] - midpoint)
        if error > MAX_CALIBRATION_ERROR_PCT:
            return False, (
                f"{sport}/{stat} {side} measured {error:.1f}pp off the {decile}-{decile + 10}% "
                f"range's midpoint ({info['count']} samples) — exceeds the "
                f"{MAX_CALIBRATION_ERROR_PCT:.0f}pp tolerance"
            ), info["count"], info["actual_rate"]
        return True, (
            f"calibration OK: {info['count']} samples, actual rate {info['actual_rate']:.1f}% "
            f"in the {decile}-{decile + 10}% range"
        ), info["count"], info["actual_rate"]

    def evaluate(self, sport: str, stat: str, side: str, price_cents: int) -> SportsPropGrounding:
        """side is "yes" (cost = price_cents) or "no" (cost = 100 - yes_bid,
        already converted by the caller before price_cents is passed in)."""
        trustworthy, note, count, actual_rate = self._trustworthiness(sport, stat, side, price_cents)
        return SportsPropGrounding(
            sport=sport, stat=stat, side=side, price_cents=price_cents,
            implied_prob=price_cents / 100.0,
            calibration_trustworthy=trustworthy, calibration_note=note,
            calibration_sample_count=count, calibration_actual_rate=actual_rate,
        )

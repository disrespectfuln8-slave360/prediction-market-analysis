"""Tests for the decile-bucketed sports-prop calibration engine.

Real-data sanity check (2026-08-15, not asserted here since it depends on
live collection): decile bucketing cleared the 30-sample floor for several
MLB categories after one night of collection; exact-price-point bucketing
cleared at most 2/10 buckets for any category. These tests cover the
mechanics with synthetic data instead.
"""

from __future__ import annotations

import json

from src.trading.polymarket_us.sports_prop_calibration import (
    MIN_SAMPLE_FLOOR,
    SportsPropCalibrationEngine,
    _decile_low,
)


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _quote(slug, sport="mlb", stat="hits", bid=None, ask=None, player=None):
    return {
        "ts": "2026-08-15T00:00:00Z",
        "slug": slug,
        "sport": sport,
        "stat": stat,
        "matchup": "aaa-bbb",
        "date": "2026-08-15",
        # Defaults to the slug itself, not a fixed "player1" -- distinct_games
        # is keyed on (matchup, date, player), so every fixture row sharing
        # one hardcoded player would collapse into a single "game" no matter
        # how many rows a test generates, undercounting real evidence the
        # same way the correlated-threshold-slugs bug did in production.
        "player": player if player is not None else slug,
        "threshold": 1,
        "bid": bid,
        "ask": ask,
        "bid_depth": 10,
        "ask_depth": 10,
    }


def _settlement(slug, sport="mlb", stat="hits", outcome_yes=True, resolved_ts="2026-08-15T00:00:00+00:00", player=None):
    return {
        "resolved_ts": resolved_ts,
        "slug": slug,
        "sport": sport,
        "stat": stat,
        "matchup": "aaa-bbb",
        "date": "2026-08-15",
        "player": player if player is not None else slug,
        "threshold": 1,
        "outcome_yes": outcome_yes,
    }


# ── _decile_low ──────────────────────────────────────────────────────────────


def test_decile_low_buckets_correctly():
    assert _decile_low(0) == 0
    assert _decile_low(9) == 0
    assert _decile_low(10) == 10
    assert _decile_low(65) == 60
    assert _decile_low(95) == 90
    assert _decile_low(100) == 90  # clamped, matching Apex's convention


# ── Engine: sample floor ─────────────────────────────────────────────────────


def test_below_sample_floor_is_untrustworthy(tmp_path):
    quotes_path = tmp_path / "quotes.jsonl"
    settlements_path = tmp_path / "settlements.jsonl"

    n = MIN_SAMPLE_FLOOR - 5
    quotes = [_quote(f"slug-{i}", ask=65) for i in range(n)]
    settlements = [_settlement(f"slug-{i}", outcome_yes=True) for i in range(n)]
    _write_jsonl(quotes_path, quotes)
    _write_jsonl(settlements_path, settlements)

    engine = SportsPropCalibrationEngine(quotes_path=quotes_path, settlements_path=settlements_path)
    grounding = engine.evaluate("mlb", "hits", "yes", price_cents=65)

    assert not grounding.calibration_trustworthy
    assert grounding.calibration_sample_count == n


def test_well_calibrated_bucket_above_floor_is_trustworthy(tmp_path):
    quotes_path = tmp_path / "quotes.jsonl"
    settlements_path = tmp_path / "settlements.jsonl"

    # 65% decile (60-70%, midpoint 65) -- win rate matching the midpoint closely.
    n = MIN_SAMPLE_FLOOR + 10
    quotes = [_quote(f"slug-{i}", ask=65) for i in range(n)]
    wins = int(n * 0.65)
    settlements = [_settlement(f"slug-{i}", outcome_yes=(i < wins)) for i in range(n)]
    _write_jsonl(quotes_path, quotes)
    _write_jsonl(settlements_path, settlements)

    engine = SportsPropCalibrationEngine(quotes_path=quotes_path, settlements_path=settlements_path)
    grounding = engine.evaluate("mlb", "hits", "yes", price_cents=65)

    assert grounding.calibration_trustworthy
    assert grounding.calibration_sample_count == n


def test_badly_miscalibrated_bucket_above_floor_is_untrustworthy(tmp_path):
    """Regression case for the exact failure shape found live in Kalshi's
    MLB-totals/spreads category: plenty of samples, but the measured rate
    is far from the bucket's own price range."""
    quotes_path = tmp_path / "quotes.jsonl"
    settlements_path = tmp_path / "settlements.jsonl"

    n = MIN_SAMPLE_FLOOR + 10
    quotes = [_quote(f"slug-{i}", ask=65) for i in range(n)]
    # Priced at 65% but only wins 20% of the time -- 45pp off, way past tolerance.
    wins = int(n * 0.20)
    settlements = [_settlement(f"slug-{i}", outcome_yes=(i < wins)) for i in range(n)]
    _write_jsonl(quotes_path, quotes)
    _write_jsonl(settlements_path, settlements)

    engine = SportsPropCalibrationEngine(quotes_path=quotes_path, settlements_path=settlements_path)
    grounding = engine.evaluate("mlb", "hits", "yes", price_cents=65)

    assert not grounding.calibration_trustworthy
    assert "exceeds" in grounding.calibration_note


# ── Engine: per-side independence ────────────────────────────────────────────


def test_yes_and_no_sides_are_evaluated_independently(tmp_path):
    """The NO side's true price is (100 - bid), which can land in a
    completely different decile than the YES side's ask -- and needs its
    own sample count and calibration check, not a shared flag. Same lesson
    as Apex's per-side gating fix earlier this session. Here every quote
    shares the same ask (concentrating YES samples into one decile) but a
    scattered bid (spreading NO samples thinly across many deciles), so
    YES clears the floor while no single NO decile does."""
    quotes_path = tmp_path / "quotes.jsonl"
    settlements_path = tmp_path / "settlements.jsonl"

    n = MIN_SAMPLE_FLOOR + 10
    # ask=65 for every row -> all n samples land in the YES 60-70% decile.
    # bid cycles 5..95 -> no_price = 100-bid scatters across many NO deciles,
    # none of which accumulates anywhere near MIN_SAMPLE_FLOOR on its own.
    quotes = [_quote(f"slug-{i}", bid=5 + (i % 19) * 5, ask=65) for i in range(n)]
    wins = int(n * 0.65)
    settlements = [_settlement(f"slug-{i}", outcome_yes=(i < wins)) for i in range(n)]
    _write_jsonl(quotes_path, quotes)
    _write_jsonl(settlements_path, settlements)

    engine = SportsPropCalibrationEngine(quotes_path=quotes_path, settlements_path=settlements_path)
    yes_grounding = engine.evaluate("mlb", "hits", "yes", price_cents=65)
    no_grounding = engine.evaluate("mlb", "hits", "no", price_cents=100 - 5)  # decile of the first row's no-price

    assert yes_grounding.calibration_trustworthy  # all n samples concentrated in one decile
    assert not no_grounding.calibration_trustworthy  # same n underlying trades, but scattered thin across NO's deciles
    assert no_grounding.calibration_sample_count < MIN_SAMPLE_FLOOR


# ── Engine: category independence ────────────────────────────────────────────


def test_categories_are_never_pooled(tmp_path):
    """Mirrors the Kalshi MLB-totals/macro pooling bug found live this
    session -- a stat type with plenty of well-calibrated data must not
    lend its trustworthiness to a different, unrelated stat type."""
    quotes_path = tmp_path / "quotes.jsonl"
    settlements_path = tmp_path / "settlements.jsonl"

    n = MIN_SAMPLE_FLOOR + 10
    quotes = [_quote(f"hits-{i}", stat="hits", ask=50) for i in range(n)]
    wins = int(n * 0.50)
    settlements = [_settlement(f"hits-{i}", stat="hits", outcome_yes=(i < wins)) for i in range(n)]
    _write_jsonl(quotes_path, quotes)
    _write_jsonl(settlements_path, settlements)

    engine = SportsPropCalibrationEngine(quotes_path=quotes_path, settlements_path=settlements_path)
    hits_grounding = engine.evaluate("mlb", "hits", "yes", price_cents=50)
    hr_grounding = engine.evaluate("mlb", "hr", "yes", price_cents=50)  # no data seeded for "hr" at all

    assert hits_grounding.calibration_trustworthy
    assert not hr_grounding.calibration_trustworthy
    assert hr_grounding.calibration_sample_count == 0


# ── Staleness ────────────────────────────────────────────────────────────────


def test_no_data_at_all_is_stale(tmp_path):
    engine = SportsPropCalibrationEngine(
        quotes_path=tmp_path / "missing_quotes.jsonl",
        settlements_path=tmp_path / "missing_settlements.jsonl",
    )
    assert engine.data_is_stale()


def test_recent_settlement_is_not_stale(tmp_path):
    from datetime import datetime, timezone

    quotes_path = tmp_path / "quotes.jsonl"
    settlements_path = tmp_path / "settlements.jsonl"
    now_iso = datetime.now(timezone.utc).isoformat()
    _write_jsonl(quotes_path, [_quote("slug-0", ask=50)])
    _write_jsonl(settlements_path, [_settlement("slug-0", resolved_ts=now_iso)])

    engine = SportsPropCalibrationEngine(quotes_path=quotes_path, settlements_path=settlements_path)
    assert not engine.data_is_stale()


# ── extra_sources (NFL's own files, merged alongside the primary MLB pair) ────


def test_extra_sources_are_merged_into_the_same_table(tmp_path):
    """NFL's own dedicated files (nfl_prop_quote_logger.py) merge in
    alongside the primary pair -- a bucket sourced entirely from an extra
    source must be gradeable exactly like one from the primary pair."""
    quotes_path = tmp_path / "quotes.jsonl"
    settlements_path = tmp_path / "settlements.jsonl"
    nfl_quotes_path = tmp_path / "nfl_quotes.jsonl"
    nfl_settlements_path = tmp_path / "nfl_settlements.jsonl"

    n = MIN_SAMPLE_FLOOR + 10
    quotes = [_quote(f"slug-{i}", sport="nfl", stat="td", ask=65) for i in range(n)]
    wins = int(n * 0.65)
    settlements = [_settlement(f"slug-{i}", sport="nfl", stat="td", outcome_yes=(i < wins)) for i in range(n)]
    _write_jsonl(nfl_quotes_path, quotes)
    _write_jsonl(nfl_settlements_path, settlements)
    # Primary pair stays empty -- everything here comes from the extra source.
    _write_jsonl(quotes_path, [])
    _write_jsonl(settlements_path, [])

    engine = SportsPropCalibrationEngine(
        quotes_path=quotes_path,
        settlements_path=settlements_path,
        extra_sources=((nfl_quotes_path, nfl_settlements_path),),
    )
    grounding = engine.evaluate("nfl", "td", "yes", price_cents=65)

    assert grounding.calibration_trustworthy
    assert grounding.calibration_sample_count == n


def test_extra_sources_never_pool_with_the_primary_sports_stat_categories(tmp_path):
    """Merging file sources must not blur (sport, stat) scoping -- an MLB
    "hits" bucket in the primary pair and an NFL "td" bucket in an extra
    source must stay fully independent, same discipline as
    test_categories_are_never_pooled."""
    quotes_path = tmp_path / "quotes.jsonl"
    settlements_path = tmp_path / "settlements.jsonl"
    nfl_quotes_path = tmp_path / "nfl_quotes.jsonl"
    nfl_settlements_path = tmp_path / "nfl_settlements.jsonl"

    n = MIN_SAMPLE_FLOOR + 10
    mlb_quotes = [_quote(f"mlb-{i}", sport="mlb", stat="hits", ask=50) for i in range(n)]
    # ask=50 -> the 50-60% decile, midpoint 55 -- win rate has to land near
    # that to be trustworthy, not just clear the sample floor.
    mlb_wins = int(n * 0.55)
    mlb_settlements = [_settlement(f"mlb-{i}", sport="mlb", stat="hits", outcome_yes=(i < mlb_wins)) for i in range(n)]
    _write_jsonl(quotes_path, mlb_quotes)
    _write_jsonl(settlements_path, mlb_settlements)
    # Only 5 NFL samples in the extra source -- well under the floor.
    nfl_quotes = [_quote(f"nfl-{i}", sport="nfl", stat="td", ask=50) for i in range(5)]
    nfl_settlements = [_settlement(f"nfl-{i}", sport="nfl", stat="td", outcome_yes=True) for i in range(5)]
    _write_jsonl(nfl_quotes_path, nfl_quotes)
    _write_jsonl(nfl_settlements_path, nfl_settlements)

    engine = SportsPropCalibrationEngine(
        quotes_path=quotes_path,
        settlements_path=settlements_path,
        extra_sources=((nfl_quotes_path, nfl_settlements_path),),
    )
    mlb_grounding = engine.evaluate("mlb", "hits", "yes", price_cents=50)
    nfl_grounding = engine.evaluate("nfl", "td", "yes", price_cents=50)

    assert mlb_grounding.calibration_trustworthy  # well past the floor on its own
    assert not nfl_grounding.calibration_trustworthy  # NFL's thin sample doesn't borrow MLB's floor
    assert nfl_grounding.calibration_sample_count == 5

"""Tests for the pure stat-extraction logic in mlb_stats_verification.py.

The network-calling functions (get_boxscore_for, verify_settlement) are
verified live against the real MLB Stats API rather than unit-tested here
-- no httpx-mocking precedent exists elsewhere in this codebase, and these
functions are thin orchestration around real_stat_value/find_player, which
carry the actual logic and are fully testable with plain fixture data.
"""

from __future__ import annotations

from src.trading.polymarket_us.mlb_stats_verification import (
    find_player,
    real_stat_value,
    verify_settlement,
)


def _player(batting=None, pitching=None, name="Jake Cronenworth"):
    return {
        "person": {"fullName": name},
        "stats": {"batting": batting or {}, "pitching": pitching or {}},
    }


# ── real_stat_value: batter stats ───────────────────────────────────────────


def test_total_bases_reads_the_real_value():
    p = _player(batting={"totalBases": 2, "hits": 1})
    assert real_stat_value(p, "tb") == 2


def test_hits_reads_the_real_value():
    p = _player(batting={"hits": 3})
    assert real_stat_value(p, "hits") == 3


def test_home_runs_reads_the_real_value():
    p = _player(batting={"homeRuns": 0})
    assert real_stat_value(p, "hr") == 0


def test_empty_batting_stats_is_zero_not_none():
    """A player IN the box score with an empty batting dict genuinely
    recorded zero of everything that game (did not play/bat) -- this is
    real signal, not missing data. This exact distinction is what
    polymarket.us's own settlement pipeline gets wrong (confirmed live:
    Nick Loftin, isOnBench, empty batting stats, still settled YES on a
    "1+ hits" prop)."""
    p = _player(batting={})
    assert real_stat_value(p, "tb") == 0
    assert real_stat_value(p, "hits") == 0
    assert real_stat_value(p, "hr") == 0


# ── real_stat_value: hrr (hits + runs + rbi) ────────────────────────────────


def test_hrr_sums_hits_runs_and_rbi():
    # Real example verified live: Ke'Bryan Hayes, 1 hit + 0 runs + 0 rbi = 1,
    # matching a recorded gte2 -> NO settlement.
    p = _player(batting={"hits": 1, "runs": 0, "rbi": 0})
    assert real_stat_value(p, "hrr") == 1


def test_hrr_empty_batting_is_zero():
    p = _player(batting={})
    assert real_stat_value(p, "hrr") == 0


# ── real_stat_value: pitcher stats ──────────────────────────────────────────


def test_outs_converts_innings_pitched_to_three_per_inning():
    p = _player(pitching={"inningsPitched": "6.1"})
    assert real_stat_value(p, "outs") == 19  # 6 full innings (18) + 1 out


def test_outs_handles_a_whole_number_of_innings():
    p = _player(pitching={"inningsPitched": "5.0"})
    assert real_stat_value(p, "outs") == 15


def test_empty_pitching_stats_is_zero_not_none():
    p = _player(pitching={})
    assert real_stat_value(p, "outs") == 0
    assert real_stat_value(p, "k") == 0
    assert real_stat_value(p, "er") == 0
    assert real_stat_value(p, "ha") == 0
    assert real_stat_value(p, "wa") == 0


def test_strikeouts_hits_allowed_walks_allowed_earned_runs():
    p = _player(pitching={"strikeOuts": 6, "hits": 8, "baseOnBalls": 1, "earnedRuns": 3})
    assert real_stat_value(p, "k") == 6
    assert real_stat_value(p, "ha") == 8
    assert real_stat_value(p, "wa") == 1
    assert real_stat_value(p, "er") == 3


def test_unknown_stat_key_returns_none():
    p = _player(batting={"hits": 1})
    assert real_stat_value(p, "not_a_real_stat") is None


# ── find_player ──────────────────────────────────────────────────────────────


def test_find_player_matches_case_insensitively():
    box = {
        "teams": {
            "away": {"players": {"12": _player(name="Jake Cronenworth")}},
            "home": {"players": {}},
        }
    }
    assert find_player(box, "JAKE CRONENWORTH") is not None


def test_find_player_returns_none_when_absent():
    box = {"teams": {"away": {"players": {}}, "home": {"players": {}}}}
    assert find_player(box, "Nobody Real") is None


def test_find_player_returns_none_for_a_missing_boxscore():
    assert find_player(None, "Anyone") is None


# ── verify_settlement: fallback behavior on failure ─────────────────────────


class _RaisingClient:
    def get_market(self, slug):
        raise RuntimeError("network down")


def test_verify_settlement_returns_none_when_the_market_lookup_fails():
    """Undeterminable must return None (caller falls back to trusting the
    exchange), never guess a direction."""
    meta = {"matchup": "sd-nym", "date": "2026-08-17", "stat": "tb", "threshold": 4}
    result = verify_settlement(_RaisingClient(), "some-slug", meta, {}, {})
    assert result is None

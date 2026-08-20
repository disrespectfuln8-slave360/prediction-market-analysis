"""Tests for the pure stat-extraction logic in nfl_stats_verification.py.
Same testing philosophy as test_mlb_stats_verification.py -- network-calling
functions verified live, pure logic unit-tested with fixture data.
"""

from __future__ import annotations

from src.trading.polymarket_us.nfl_stats_verification import real_stat_value, verify_settlement


def _boxscore(passing=None, rushing=None, receiving=None):
    stats = []
    if passing is not None:
        stats.append({"name": "passing", "labels": ["TD", "YDS"], "athletes": passing})
    if rushing is not None:
        stats.append({"name": "rushing", "labels": ["YDS", "TD"], "athletes": rushing})
    if receiving is not None:
        stats.append({"name": "receiving", "labels": ["YDS", "TD"], "athletes": receiving})
    return {"boxscore": {"players": [{"statistics": stats}]}}


def _athlete(name, stat_values):
    return {"athlete": {"displayName": name}, "stats": stat_values}


# ── real_stat_value: passing/receiving/rushing yards and passing TDs ───────


def test_passing_yards_and_touchdowns():
    box = _boxscore(passing=[_athlete("Quinn Ewers", ["0", "27"])])
    assert real_stat_value(box, "Quinn Ewers", "pyd") == 27
    assert real_stat_value(box, "Quinn Ewers", "ptd") == 0


def test_receiving_yards():
    box = _boxscore(receiving=[_athlete("Garrett Wilson", ["32", "0"])])
    assert real_stat_value(box, "Garrett Wilson", "recyd") == 32


def test_rushing_yards():
    box = _boxscore(rushing=[_athlete("Breece Hall", ["16", "0"])])
    assert real_stat_value(box, "Breece Hall", "ryd") == 16


def test_player_not_in_the_relevant_category_returns_none():
    box = _boxscore(passing=[_athlete("Quinn Ewers", ["0", "27"])])
    assert real_stat_value(box, "Someone Else", "pyd") is None


def test_missing_boxscore_returns_none():
    assert real_stat_value(None, "Anyone", "pyd") is None


# ── real_stat_value: td (rushing + receiving, excluding passing TDs) ───────


def test_td_sums_rushing_and_receiving_touchdowns():
    box = _boxscore(
        rushing=[_athlete("Jonah Coleman", ["10", "1"])],
        receiving=[_athlete("Jonah Coleman", ["5", "1"])],
    )
    assert real_stat_value(box, "Jonah Coleman", "td") == 2


def test_td_excludes_a_quarterbacks_own_passing_touchdowns():
    """A QB's passing TD count is not "this player's" td prop -- e.g. Josh
    Allen throwing 2 TDs is not "Josh Allen scores 2 touchdowns himself"."""
    box = _boxscore(passing=[_athlete("Josh Allen", ["3", "250"])])
    assert real_stat_value(box, "Josh Allen", "td") == 0


def test_td_returns_none_when_the_player_never_appears_anywhere():
    box = _boxscore(rushing=[_athlete("Someone Else", ["10", "1"])])
    assert real_stat_value(box, "Not In This Game", "td") is None


def test_unrecognized_stat_key_returns_none():
    box = _boxscore(passing=[_athlete("Quinn Ewers", ["0", "27"])])
    assert (
        real_stat_value(box, "not_a_real_stat", "pyd") is None
        or real_stat_value(box, "Quinn Ewers", "not_a_real_stat") is None
    )


# ── verify_settlement: fallback behavior on failure ─────────────────────────


class _RaisingClient:
    def get_market(self, slug):
        raise RuntimeError("network down")


def test_verify_settlement_returns_none_when_the_market_lookup_fails():
    meta = {"matchup": "den-atl", "date": "2026-08-14", "stat": "ptd", "threshold": 1}
    result = verify_settlement(_RaisingClient(), "some-slug", meta, {})
    assert result is None

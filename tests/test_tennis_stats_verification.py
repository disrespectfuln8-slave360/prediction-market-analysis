"""Tests for the pure logic in tennis_stats_verification.py.

Network-calling functions (get_events_for, find_match) verified live
against the real ESPN API rather than unit-tested here, same philosophy as
the MLB/NFL verification test files. The real, live-verified example this
module's own docstring cites (Alex de Minaur bt Arthur Fery 7-5 7-6(7-3),
Cincinnati Open 2026-08-17: 25 real total games, 2 real sets) is used
directly as fixture data below so the pure math is checked against a case
that's already been confirmed correct against the live API.
"""

from __future__ import annotations

from src.trading.polymarket_us.tennis_stats_verification import (
    find_match,
    player_code,
    real_total_games,
    real_total_sets,
    verify_settlement,
)

# The real, live-confirmed de Minaur vs Fery match (Cincinnati Open, 2026-08-17).
DEMINAUR_FERY_MATCH = {
    "competitors": [
        {
            "athlete": {"fullName": "Alex de Minaur"},
            "winner": True,
            "linescores": [{"value": 7.0, "winner": True}, {"value": 7.0, "tiebreak": 7, "winner": True}],
        },
        {
            "athlete": {"fullName": "Arthur Fery"},
            "winner": False,
            "linescores": [{"value": 5.0, "winner": False}, {"value": 6.0, "tiebreak": 3, "winner": False}],
        },
    ],
}


# ── player_code ──────────────────────────────────────────────────────────────


def test_player_code_uses_first_and_last_token():
    assert player_code("Arthur Fery") == "artfer"
    assert player_code("Alex de Minaur") == "alemin"  # middle token "de" skipped


def test_player_code_handles_a_hyphenated_surname():
    assert player_code("Felix Auger-Aliassime") == "felaug"


def test_player_code_handles_a_three_token_name_with_no_hyphen():
    # Real example verified live: matches the "juacer" code on a real
    # polymarket.us slug.
    assert player_code("Juan Manuel Cerundolo") == "juacer"


def test_player_code_returns_empty_for_a_single_token_name():
    assert player_code("Cher") == ""


# ── real_total_games / real_total_sets ──────────────────────────────────────


def test_real_total_games_sums_every_linescore_both_competitors():
    # 7+7 (de Minaur) + 5+6 (Fery) = 25, confirmed live against tg-22.5=True,
    # tg-25.5=False on the real settled ladder for this exact match.
    assert real_total_games(DEMINAUR_FERY_MATCH) == 25


def test_real_total_sets_counts_linescores():
    assert real_total_sets(DEMINAUR_FERY_MATCH) == 2


def test_real_total_games_none_when_a_linescore_value_is_missing():
    match = {
        "competitors": [
            {"athlete": {"fullName": "A B"}, "linescores": [{"winner": True}]},
            {"athlete": {"fullName": "C D"}, "linescores": [{"value": 4.0, "winner": False}]},
        ]
    }
    assert real_total_games(match) is None


def test_real_total_sets_none_when_competitors_disagree_on_set_count():
    """Malformed/incomplete data -- must not guess."""
    match = {
        "competitors": [
            {"athlete": {"fullName": "A B"}, "linescores": [{"value": 6.0}, {"value": 4.0}]},
            {"athlete": {"fullName": "C D"}, "linescores": [{"value": 4.0}]},
        ]
    }
    assert real_total_sets(match) is None


def test_real_total_games_none_for_the_wrong_number_of_competitors():
    assert real_total_games({"competitors": []}) is None


# ── find_match ───────────────────────────────────────────────────────────────


def test_find_match_matches_on_the_code_set_regardless_of_order():
    cache = {
        ("2026-08-17", "atp"): [
            {"groupings": [{"competitions": [DEMINAUR_FERY_MATCH]}]},
        ]
    }
    found = find_match("2026-08-17", "atp", "alemin", "artfer", cache)
    assert found is DEMINAUR_FERY_MATCH
    # Order shouldn't matter -- it's a set comparison.
    found2 = find_match("2026-08-17", "atp", "artfer", "alemin", cache)
    assert found2 is DEMINAUR_FERY_MATCH


def test_find_match_returns_none_when_no_match_found():
    cache = {
        ("2026-08-17", "atp"): [
            {"groupings": [{"competitions": [DEMINAUR_FERY_MATCH]}]},
        ]
    }
    assert find_match("2026-08-17", "atp", "nobody", "atall", cache) is None


# ── verify_settlement: scope and fallback behavior ──────────────────────────


class _StubClient:
    pass  # tennis verification never calls client.get_market -- names come from ESPN


def test_verify_settlement_returns_none_for_spread_categories():
    """Deliberately unsupported -- see module docstring for the real
    ambiguity found trying to verify spreads."""
    meta = {"category": "atp_game_spreads", "matchup": "artfer-alemin", "date": "2026-08-17", "threshold": 1.5}
    assert verify_settlement(_StubClient(), "some-slug", meta, {}) is None


def test_verify_settlement_returns_none_for_a_malformed_matchup():
    meta = {"category": "atp_game_totals", "matchup": "onlyoneplayer", "date": "2026-08-17", "threshold": 20.5}
    assert verify_settlement(_StubClient(), "some-slug", meta, {}) is None


def test_verify_settlement_returns_none_when_no_match_is_found():
    meta = {"category": "atp_game_totals", "matchup": "nobody-atall", "date": "2026-08-17", "threshold": 20.5}
    cache = {("2026-08-17", "atp"): []}
    assert verify_settlement(_StubClient(), "some-slug", meta, cache) is None

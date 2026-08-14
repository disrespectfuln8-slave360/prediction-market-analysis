"""Tests for the MLB/NFL player-prop quote+outcome logger.

Slug parsing is the load-bearing piece: it has to correctly split sport,
stat, player, and threshold apart, and it has to reject anything that
isn't actually this specific ladder shape (e.g. astatc-fwc-* World Cup
markets) rather than mis-parsing them.
"""
from __future__ import annotations

import json

import pytest

from src.trading.polymarket_us.models import Market
from src.trading.polymarket_us.sports_prop_quote_logger import (
    MAX_PENDING_CYCLES,
    SportsPropQuoteLogger,
    parse_prop_slug,
)


def make_market(slug):
    return Market(
        slug=slug, id=slug, title=slug, category="sports", tags=[],
        active=True, closed=False, close_time=None,
        yes_bid=50, yes_ask=52, no_bid=48, no_ask=50,
        volume=0.0, volume_24h=0.0, bid_depth=10, ask_depth=10,
    )


# ── Slug parsing ─────────────────────────────────────────────────────────────

def test_parses_mlb_total_bases_prop():
    meta = parse_prop_slug("astatc-mlb-col-sf-2026-08-14-outs-kylfre-gte16")
    assert meta == {
        "sport": "mlb", "matchup": "col-sf", "date": "2026-08-14",
        "stat": "outs", "player": "kylfre", "threshold": 16,
    }


def test_parses_nfl_touchdown_prop():
    meta = parse_prop_slug("astatc-nfl-lac-hou-2026-08-13-td-trelan-gte1")
    assert meta == {
        "sport": "nfl", "matchup": "lac-hou", "date": "2026-08-13",
        "stat": "td", "player": "trelan", "threshold": 1,
    }


def test_rejects_non_ladder_shaped_astatc_markets():
    """astatc-fwc-* (World Cup) markets don't have the sport-stat-player-gteN
    shape at all -- must be skipped, not mis-parsed."""
    assert parse_prop_slug("astatc-fwc-arg-sui") is None


def test_rejects_sports_outside_the_tracked_set():
    assert parse_prop_slug("astatc-nba-lal-bos-2026-08-14-pts-jamesx-gte20") is None


def test_rejects_non_astatc_slugs():
    assert parse_prop_slug("tc-temp-nychigh-2026-08-14-gte86lt87f") is None


# ── Logger state machine ─────────────────────────────────────────────────────

class FakeClient:
    def __init__(self, markets, bbo=None, settlements=None, settlement_errors=()):
        self._markets = markets
        self._bbo = bbo or {}
        self._settlements = settlements or {}
        self._settlement_errors = set(settlement_errors)

    def list_all_markets(self, active=True, closed=False):
        return self._markets

    def get_bbo(self, slug):
        return self._bbo.get(slug, {"marketData": {}})

    def get_settlement(self, slug):
        if slug in self._settlement_errors or slug not in self._settlements:
            raise RuntimeError("404 not settled yet")
        return self._settlements[slug]


SLUG = "astatc-mlb-col-sf-2026-08-14-outs-kylfre-gte16"


def test_discover_tracks_active_buckets_and_ignores_non_prop_markets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([make_market(SLUG), make_market("astatc-fwc-arg-sui")])
    logger_ = SportsPropQuoteLogger(client)

    logger_._discover()

    assert list(logger_._active) == [SLUG]


def test_quotes_are_logged_with_parsed_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient(
        [make_market(SLUG)],
        bbo={SLUG: {"marketData": {
            "bestBid": {"value": "0.15"}, "bestAsk": {"value": "0.18"},
            "bidDepth": 5, "askDepth": 8,
        }}},
    )
    logger_ = SportsPropQuoteLogger(client)
    logger_.run_cycle()

    from src.trading.polymarket_us.sports_prop_quote_logger import QUOTES_PATH
    rows = [json.loads(line) for line in QUOTES_PATH.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["slug"] == SLUG
    assert rows[0]["sport"] == "mlb"
    assert rows[0]["stat"] == "outs"
    assert rows[0]["bid"] == 15
    assert rows[0]["ask"] == 18


def test_bucket_disappearing_moves_to_pending_settlement(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([make_market(SLUG)])
    logger_ = SportsPropQuoteLogger(client)
    logger_._discover()
    assert SLUG in logger_._active

    client._markets = []  # the bucket closed -- no longer in the active listing
    logger_._discover()

    assert SLUG not in logger_._active
    assert SLUG in logger_._pending_settlement


def test_resolved_settlement_is_logged_and_cleared_from_pending(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient(
        [make_market(SLUG)],
        settlements={SLUG: {"settlement": 1}},  # observed live shape: 1=YES, 0=NO, in dollars
    )
    logger_ = SportsPropQuoteLogger(client)
    logger_._pending_settlement[SLUG] = (parse_prop_slug(SLUG), 0)

    logger_._check_settlements()

    assert SLUG not in logger_._pending_settlement
    from src.trading.polymarket_us.sports_prop_quote_logger import SETTLEMENTS_PATH
    rows = [json.loads(line) for line in SETTLEMENTS_PATH.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["slug"] == SLUG
    assert rows[0]["outcome_yes"] is True


def test_unsettled_bucket_stays_pending_without_writing_a_row(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([make_market(SLUG)], settlement_errors={SLUG})
    logger_ = SportsPropQuoteLogger(client)
    logger_._pending_settlement[SLUG] = (parse_prop_slug(SLUG), 0)

    logger_._check_settlements()

    assert SLUG in logger_._pending_settlement
    meta, cycles = logger_._pending_settlement[SLUG]
    assert cycles == 1
    from src.trading.polymarket_us.sports_prop_quote_logger import SETTLEMENTS_PATH
    assert not SETTLEMENTS_PATH.exists()


def test_pending_settlement_is_dropped_after_the_retry_ceiling(tmp_path, monkeypatch):
    """A bucket that never resolves is more likely dead/cancelled than a
    slow settlement -- must not retry forever."""
    monkeypatch.chdir(tmp_path)
    client = FakeClient([make_market(SLUG)], settlement_errors={SLUG})
    logger_ = SportsPropQuoteLogger(client)
    logger_._pending_settlement[SLUG] = (parse_prop_slug(SLUG), MAX_PENDING_CYCLES - 1)

    logger_._check_settlements()

    assert SLUG not in logger_._pending_settlement

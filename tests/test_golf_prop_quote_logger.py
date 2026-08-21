"""Tests for the PGA round-leader quote+outcome logger.

See golf_prop_quote_logger.py's module docstring for the real market
survey behind the scope: round-leader only, PGA only (LPGA/LIV confirmed
illiquid via real bid/ask/depth/shares-traded checks), outright-winner
excluded (structurally longshot, will never clear this account's 30-70c
no-longshot gate).
"""

from __future__ import annotations

import json

from src.trading.polymarket_us.golf_prop_quote_logger import (
    GolfPropQuoteLogger,
    parse_golf_slug,
)
from src.trading.polymarket_us.models import Market


def make_market(slug):
    return Market(
        slug=slug,
        id=slug,
        title=slug,
        category="sports",
        tags=[],
        active=True,
        closed=False,
        close_time=None,
        yes_bid=40,
        yes_ask=44,
        no_bid=56,
        no_ask=60,
        volume=0.0,
        volume_24h=0.0,
        bid_depth=10,
        ask_depth=10,
    )


# ── Slug parsing ─────────────────────────────────────────────────────────────


def test_parses_a_real_round_leader_slug():
    meta = parse_golf_slug("tec-pga-bmwcham-2026-08-20-r1l-jspa")
    assert meta == {
        "sport": "golf",
        "matchup": "bmwcham",
        "date": "2026-08-20",
        "stat": "r1l",
        "player": "jspa",
    }


def test_parses_round2_and_round3():
    assert parse_golf_slug("tec-pga-ryder-2026-08-20-r2l-ludabe")["stat"] == "r2l"
    assert parse_golf_slug("tec-pga-ryder-2026-08-20-r3l-ludabe")["stat"] == "r3l"


def test_rejects_outright_winner_shape():
    """Confirmed real: median 4c across the field, structurally longshot --
    deliberately excluded, see module docstring."""
    assert parse_golf_slug("tec-pga-bmwcham-2026-08-20-w-ludabe") is None


def test_rejects_lpga_and_liv():
    """Confirmed real via BBO: placeholder quotes, zero shares ever traded
    -- deliberately excluded, see module docstring."""
    assert parse_golf_slug("tec-lpga-cpkcwome-2026-08-20-r1l-jenkup") is None
    assert parse_golf_slug("tec-liv-indiindi-2026-08-20-r1l-joanie") is None


def test_rejects_non_golf_slugs():
    assert parse_golf_slug("astatc-mlb-col-sf-2026-08-14-outs-kylfre-gte16") is None
    assert parse_golf_slug("tc-temp-nychigh-2026-08-14-gte86lt87f") is None


# ── Logger state machine ─────────────────────────────────────────────────────


class FakeClient:
    def __init__(self, markets, bbo=None, settlements=None, settlement_errors=()):
        self._markets = markets
        self._bbo = bbo or {}
        self._settlements = settlements or {}
        self._settlement_errors = set(settlement_errors)
        self.bbo_calls = 0

    def list_all_markets(self, active=True, closed=False):
        return self._markets

    def get_bbo(self, slug):
        self.bbo_calls += 1
        return self._bbo.get(slug, {"marketData": {}})

    def get_settlement(self, slug):
        if slug in self._settlement_errors or slug not in self._settlements:
            raise RuntimeError("404 not settled yet")
        return self._settlements[slug]


SLUG = "tec-pga-bmwcham-2026-08-20-r1l-jspa"


def test_discover_tracks_active_buckets_and_ignores_other_sports(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([make_market(SLUG), make_market("astatc-mlb-col-sf-2026-08-14-outs-kylfre-gte16")])
    logger_ = GolfPropQuoteLogger(client)

    logger_._discover()

    assert list(logger_._active) == [SLUG]


def test_discover_ignores_outright_winner_and_lpga_liv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient(
        [
            make_market(SLUG),
            make_market("tec-pga-bmwcham-2026-08-20-w-ludabe"),
            make_market("tec-lpga-cpkcwome-2026-08-20-r1l-jenkup"),
        ]
    )
    logger_ = GolfPropQuoteLogger(client)

    logger_._discover()

    assert list(logger_._active) == [SLUG]


def test_quotes_are_logged_with_parsed_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient(
        [make_market(SLUG)],
        bbo={
            SLUG: {
                "marketData": {
                    "bestBid": {"value": "0.41"},
                    "bestAsk": {"value": "0.44"},
                    "bidDepth": 41,
                    "askDepth": 16,
                }
            }
        },
    )
    logger_ = GolfPropQuoteLogger(client)
    logger_.run_cycle()

    from src.trading.polymarket_us.golf_prop_quote_logger import QUOTES_PATH

    rows = [json.loads(line) for line in QUOTES_PATH.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["slug"] == SLUG
    assert rows[0]["sport"] == "golf"
    assert rows[0]["stat"] == "r1l"
    assert rows[0]["matchup"] == "bmwcham"
    assert rows[0]["bid"] == 41
    assert rows[0]["ask"] == 44


def test_a_slug_with_both_ask_and_bid_is_never_logged_again(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient(
        [make_market(SLUG)],
        bbo={
            SLUG: {
                "marketData": {
                    "bestBid": {"value": "0.41"},
                    "bestAsk": {"value": "0.44"},
                    "bidDepth": 41,
                    "askDepth": 16,
                }
            }
        },
    )
    logger_ = GolfPropQuoteLogger(client)
    logger_._active = {SLUG: parse_golf_slug(SLUG)}

    logger_._log_quotes()
    assert client.bbo_calls == 1
    assert SLUG in logger_._have_ask and SLUG in logger_._have_bid

    logger_._log_quotes()
    assert client.bbo_calls == 1  # never fetched again


def test_capture_state_is_seeded_from_disk_on_startup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from src.trading.polymarket_us.golf_prop_quote_logger import QUOTES_PATH

    QUOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUOTES_PATH.write_text(json.dumps({"slug": SLUG, "bid": 41, "ask": 44}) + "\n")

    client = FakeClient([make_market(SLUG)])
    logger_ = GolfPropQuoteLogger(client)

    assert SLUG in logger_._have_ask
    assert SLUG in logger_._have_bid


def test_bucket_disappearing_moves_to_pending_settlement(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([make_market(SLUG)])
    logger_ = GolfPropQuoteLogger(client)
    logger_._discover()
    assert SLUG in logger_._active

    client._markets = []
    logger_._discover()

    assert SLUG not in logger_._active
    assert SLUG in logger_._pending_settlement


def test_resolved_settlement_is_logged_and_cleared_from_pending(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([make_market(SLUG)], settlements={SLUG: {"settlement": 1}})
    logger_ = GolfPropQuoteLogger(client)
    logger_._discover()
    assert SLUG in logger_._active

    client._markets = []
    logger_._discover()
    assert SLUG in logger_._pending_settlement

    logger_._check_settlements()

    from src.trading.polymarket_us.golf_prop_quote_logger import SETTLEMENTS_PATH

    rows = [json.loads(line) for line in SETTLEMENTS_PATH.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["slug"] == SLUG
    assert rows[0]["outcome_yes"] is True
    assert SLUG not in logger_._pending_settlement


def test_unsettled_bucket_stays_pending_without_writing_a_row(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([make_market(SLUG)], settlement_errors=[SLUG])
    logger_ = GolfPropQuoteLogger(client)
    logger_._pending_settlement = {SLUG: (parse_golf_slug(SLUG), 0)}

    logger_._check_settlements()

    from src.trading.polymarket_us.golf_prop_quote_logger import SETTLEMENTS_PATH

    assert not SETTLEMENTS_PATH.exists()
    assert SLUG in logger_._pending_settlement

"""Tests for the tennis spread/totals quote+outcome logger.

Slug parsing needs to get category (for scoping), matchup (for the trading
bot's one-position-per-match guard), and date (for display/staleness)
right. Category is finer-grained than it looks from the slug prefix alone
-- see module docstring for the real gs/ss/tg/st sub-type evidence found
2026-08-18 (a game-spread and a set-spread share the asc- prefix but are
genuinely different quantities, same for tg/st under tsc-).
"""
from __future__ import annotations

import json

import pytest

from src.trading.polymarket_us.models import Market
from src.trading.polymarket_us.tennis_prop_quote_logger import (
    CATEGORIES,
    TennisPropQuoteLogger,
    parse_tennis_slug,
)


def make_market(slug):
    return Market(
        slug=slug, id=slug, title=slug, category="sports", tags=[],
        active=True, closed=False, close_time=None,
        yes_bid=50, yes_ask=52, no_bid=48, no_ask=50,
        volume=0.0, volume_24h=0.0, bid_depth=10, ask_depth=10,
    )


# ── Slug parsing ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("slug,expected_category", [
    ("asc-atp-tompau-adoval-2026-08-17-gs-neg-1pt5", "atp_game_spreads"),
    ("asc-atp-tompau-adoval-2026-08-17-ss-neg-1pt5", "atp_set_spreads"),
    ("asc-wta-aleeal-amaani-2026-08-16-gs-pos-1pt5", "wta_game_spreads"),
    ("asc-wta-aleeal-amaani-2026-08-16-ss-pos-1pt5", "wta_set_spreads"),
    ("tsc-atp-artfer-alemin-2026-08-17-tg-17pt5", "atp_game_totals"),
    ("tsc-atp-artfer-alemin-2026-08-17-st-2pt5", "atp_set_totals"),
    ("tsc-wta-aleeal-amaani-2026-08-16-tg-22pt5", "wta_game_totals"),
    ("tsc-wta-aleeal-amaani-2026-08-16-st-2pt5", "wta_set_totals"),
])
def test_parse_recognizes_every_tracked_category(slug, expected_category):
    meta = parse_tennis_slug(slug)
    assert meta is not None
    assert meta["category"] == expected_category


def test_parse_extracts_the_embedded_date():
    meta = parse_tennis_slug("tsc-atp-artfer-alemin-2026-08-17-tg-17pt5")
    assert meta["date"] == "2026-08-17"


def test_parse_extracts_the_matchup():
    """Used by the trading bot's one-position-per-match guard -- a spread
    bet and a totals bet on the SAME match are correlated."""
    meta = parse_tennis_slug("tsc-atp-artfer-alemin-2026-08-17-tg-17pt5")
    assert meta["matchup"] == "artfer-alemin"


def test_parse_extracts_an_unsigned_totals_threshold():
    meta = parse_tennis_slug("tsc-atp-artfer-alemin-2026-08-17-tg-17pt5")
    assert meta["threshold"] == 17.5


def test_parse_extracts_a_signed_spread_threshold():
    neg = parse_tennis_slug("asc-atp-tompau-adoval-2026-08-17-gs-neg-1pt5")
    pos = parse_tennis_slug("asc-atp-tompau-adoval-2026-08-17-gs-pos-8pt5")
    assert neg["threshold"] == -1.5
    assert pos["threshold"] == 8.5


def test_game_and_set_variants_are_scored_as_distinct_categories():
    """Real evidence 2026-08-18: ss/st thresholds only ever take one value
    (1.5, 2.5 -- exactly what a best-of-3 set-margin/set-total produces),
    while gs/tg span wide ranges (0.5-10.5, 11.5-30.5) -- genuinely
    different quantities that must never share a calibration bucket, even
    though they share everything else about their slug shape."""
    game_spread = parse_tennis_slug("asc-atp-tompau-adoval-2026-08-17-gs-neg-1pt5")
    set_spread = parse_tennis_slug("asc-atp-tompau-adoval-2026-08-17-ss-neg-1pt5")
    assert game_spread["category"] != set_spread["category"]


def test_parse_rejects_an_unrecognized_subtype_marker():
    """A slug matching the tour prefix but with a marker outside {gs,ss,tg,
    st} must be rejected outright, not silently mis-bucketed."""
    assert parse_tennis_slug("asc-atp-tompau-adoval-2026-08-17-xx-neg-1pt5") is None


def test_parse_rejects_shapes_surveyed_and_excluded_on_purpose():
    """Real tennis inventory the module docstring explains skipping:
    exact-score/set-winner (categorical, not a threshold ladder), match
    moneyline (no threshold at all), and tournament/year-end outrights
    (settle too infrequently to ever calibrate)."""
    assert parse_tennis_slug("astatc-atp-artfer-alemin-2026-08-17-es-2-0") is None
    assert parse_tennis_slug("astatc-wta-mayjoi-tamkor-2026-08-13-sw1") is None
    assert parse_tennis_slug("aec-atp-johnik-henber-2026-08-17") is None
    assert parse_tennis_slug("tec-atp-uso-2026-09-13-w-holrun") is None
    assert parse_tennis_slug("arankc-atp-2026-12-31-no1-tayfri") is None


def test_parse_rejects_unrelated_prefixes():
    assert parse_tennis_slug("astatc-mlb-col-sf-2026-08-14-outs-kylfre-gte16") is None
    assert parse_tennis_slug("tc-temp-nychigh-2026-08-14-gte86lt87f") is None


def test_atp_and_wta_are_scored_as_distinct_categories():
    """The never-pool-categories lesson applied here: ATP (best-of-5 at
    majors) and WTA (always best-of-3) must never share a calibration
    bucket, even though asc-atp/asc-wta share everything else about their
    slug shape."""
    atp = parse_tennis_slug("tsc-atp-artfer-alemin-2026-08-17-tg-17pt5")
    wta = parse_tennis_slug("tsc-wta-aleeal-amaani-2026-08-16-tg-17pt5")
    assert atp["category"] != wta["category"]
    assert set(CATEGORIES) == {
        "atp_game_spreads", "atp_set_spreads", "atp_game_totals", "atp_set_totals",
        "wta_game_spreads", "wta_set_spreads", "wta_game_totals", "wta_set_totals",
    }


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


SLUG = "tsc-atp-artfer-alemin-2026-08-17-tg-17pt5"


def test_discover_tracks_active_buckets_and_ignores_untracked_categories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([make_market(SLUG), make_market("astatc-atp-artfer-alemin-2026-08-17-es-2-0")])
    logger_ = TennisPropQuoteLogger(client)

    logger_._discover()

    assert list(logger_._active) == [SLUG]


def test_quotes_are_logged_with_parsed_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient(
        [make_market(SLUG)],
        bbo={SLUG: {"marketData": {
            "bestBid": {"value": "0.40"}, "bestAsk": {"value": "0.60"},
            "bidDepth": 5, "askDepth": 8,
        }}},
    )
    logger_ = TennisPropQuoteLogger(client)
    logger_.run_cycle()

    from src.trading.polymarket_us.tennis_prop_quote_logger import QUOTES_PATH
    rows = [json.loads(line) for line in QUOTES_PATH.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["slug"] == SLUG
    assert rows[0]["category"] == "atp_game_totals"
    assert rows[0]["bid"] == 40
    assert rows[0]["ask"] == 60


def test_a_slug_with_both_ask_and_bid_is_never_logged_again(tmp_path, monkeypatch):
    """Same regression sports_prop_quote_logger.py hit in production
    (60MB/228k lines in a day) -- built correctly here from the start."""
    monkeypatch.chdir(tmp_path)
    client = FakeClient(
        [make_market(SLUG)],
        bbo={SLUG: {"marketData": {
            "bestBid": {"value": "0.40"}, "bestAsk": {"value": "0.60"}, "bidDepth": 5, "askDepth": 8,
        }}},
    )
    logger_ = TennisPropQuoteLogger(client)

    logger_._discover()
    logger_._log_quotes()
    assert client.bbo_calls == 1

    logger_._log_quotes()
    assert client.bbo_calls == 1  # never fetched again


def test_capture_state_is_seeded_from_disk_on_startup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from src.trading.polymarket_us.tennis_prop_quote_logger import QUOTES_PATH
    QUOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUOTES_PATH.write_text(json.dumps({"slug": SLUG, "bid": 40, "ask": 60}) + "\n")

    client = FakeClient([make_market(SLUG)])
    logger_ = TennisPropQuoteLogger(client)

    assert SLUG in logger_._have_ask
    assert SLUG in logger_._have_bid


def test_bucket_disappearing_moves_to_pending_settlement(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([make_market(SLUG)])
    logger_ = TennisPropQuoteLogger(client)
    logger_._discover()
    assert SLUG in logger_._active

    client._markets = []
    logger_._discover()

    assert SLUG not in logger_._active
    assert SLUG in logger_._pending_settlement


def test_resolved_settlement_is_logged_and_cleared_from_pending(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([make_market(SLUG)], settlements={SLUG: {"settlement": 1}})
    # Real network calls only happen with the default verifier -- see the
    # dedicated settlement-verification tests below for that behavior.
    logger_ = TennisPropQuoteLogger(client, verify_settlement_fn=lambda *a, **k: None)
    logger_._discover()
    assert SLUG in logger_._active

    client._markets = []  # the bucket closed -- no longer in the active listing
    logger_._discover()
    assert SLUG in logger_._pending_settlement

    logger_._check_settlements()

    from src.trading.polymarket_us.tennis_prop_quote_logger import SETTLEMENTS_PATH
    rows = [json.loads(line) for line in SETTLEMENTS_PATH.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["slug"] == SLUG
    assert rows[0]["outcome_yes"] is True
    assert SLUG not in logger_._pending_settlement


def test_unsettled_bucket_stays_pending_without_writing_a_row(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([make_market(SLUG)], settlement_errors={SLUG})
    logger_ = TennisPropQuoteLogger(client)
    logger_._discover()

    client._markets = []
    logger_._discover()

    logger_._check_settlements()

    from src.trading.polymarket_us.tennis_prop_quote_logger import SETTLEMENTS_PATH
    assert not SETTLEMENTS_PATH.exists()
    assert SLUG in logger_._pending_settlement


# ── Settlement verification against real ESPN match data (totals only) ─────
# Real finding 2026-08-18: tennis totals (tg/st) verify cleanly against real
# match data, but spreads (gs/ss) do not resolve under any sign convention
# tried -- verify_settlement deliberately returns None for spreads (see
# tennis_stats_verification.py's docstring). These tests cover the override
# wiring itself with an injected fake; the real verifier's own logic is
# covered by test_tennis_stats_verification.py.

def test_settlement_verification_overrides_a_wrong_exchange_settlement(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([make_market(SLUG)], settlements={SLUG: {"settlement": 1}})
    logger_ = TennisPropQuoteLogger(client, verify_settlement_fn=lambda *a, **k: False)
    logger_._discover()
    client._markets = []
    logger_._discover()

    logger_._check_settlements()

    from src.trading.polymarket_us.tennis_prop_quote_logger import SETTLEMENTS_PATH
    rows = [json.loads(line) for line in SETTLEMENTS_PATH.read_text().splitlines()]
    assert rows[0]["outcome_yes"] is False


def test_settlement_verification_inconclusive_falls_back_to_the_exchange_value(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([make_market(SLUG)], settlements={SLUG: {"settlement": 1}})
    logger_ = TennisPropQuoteLogger(client, verify_settlement_fn=lambda *a, **k: None)
    logger_._discover()
    client._markets = []
    logger_._discover()

    logger_._check_settlements()

    from src.trading.polymarket_us.tennis_prop_quote_logger import SETTLEMENTS_PATH
    rows = [json.loads(line) for line in SETTLEMENTS_PATH.read_text().splitlines()]
    assert rows[0]["outcome_yes"] is True


def test_settlement_verification_raising_falls_back_to_the_exchange_value(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    def _boom(*a, **k):
        raise RuntimeError("ESPN is down")
    client = FakeClient([make_market(SLUG)], settlements={SLUG: {"settlement": 1}})
    logger_ = TennisPropQuoteLogger(client, verify_settlement_fn=_boom)
    logger_._discover()
    client._markets = []
    logger_._discover()

    logger_._check_settlements()

    from src.trading.polymarket_us.tennis_prop_quote_logger import SETTLEMENTS_PATH
    rows = [json.loads(line) for line in SETTLEMENTS_PATH.read_text().splitlines()]
    assert rows[0]["outcome_yes"] is True

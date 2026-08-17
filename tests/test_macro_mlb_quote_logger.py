"""Tests for the macro/MLB-totals-and-spreads quote+outcome logger.

Slug parsing only needs to get category (for scoping) and date (for
display/staleness) right -- unlike sports-props, calibration here never
decodes the bucket's own semantic threshold, so there's no equivalent of
that module's stricter shape validation to test.
"""
from __future__ import annotations

import json

import pytest

from src.trading.polymarket_us.models import Market
from src.trading.polymarket_us.macro_mlb_quote_logger import (
    CATEGORY_PREFIXES,
    MacroMlbQuoteLogger,
    parse_macro_mlb_slug,
)


def make_market(slug):
    return Market(
        slug=slug, id=slug, title=slug, category="macro", tags=[],
        active=True, closed=False, close_time=None,
        yes_bid=50, yes_ask=52, no_bid=48, no_ask=50,
        volume=0.0, volume_24h=0.0, bid_depth=10, ask_depth=10,
    )


# ── Slug parsing ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("slug,expected_category", [
    ("cbpac-usfed-2026-cut", "fed_annual"),
    ("rdc-boc-2026-09-02-hike50", "rate_decision"),
    ("gdpc-us-saa-q3-2026-10-29-gt2pt5", "gdp"),
    ("urc-usunemp-sa-august-2026-09-04-gt4pt5pct", "unemployment"),
    ("nfpc-usnfp-sa-august-2026-09-04-gt150k", "nonfarm_payrolls"),
    ("cpic-uscpi-august-yoy-2026-09-11-gt2pt9pct", "cpi"),
    ("tsc-mlb-stl-cin-2026-08-17-dh1-f5-3pt5", "mlb_totals"),
    ("asc-mlb-az-bos-2026-08-17-neg-2pt5", "mlb_spreads"),
])
def test_parse_recognizes_every_tracked_category(slug, expected_category):
    meta = parse_macro_mlb_slug(slug)
    assert meta is not None
    assert meta["category"] == expected_category


def test_parse_extracts_the_embedded_date():
    meta = parse_macro_mlb_slug("rdc-boc-2026-09-02-hike50")
    assert meta["date"] == "2026-09-02"


def test_parse_tolerates_a_missing_date():
    """cbpac's 2 real markets are irregularly shaped -- one has no
    YYYY-MM-DD segment at all. Must not crash, just report date=None."""
    meta = parse_macro_mlb_slug("cbpac-usfed-2026-cut")
    assert meta is not None
    assert meta["date"] is None


def test_parse_rejects_untracked_prefixes():
    assert parse_macro_mlb_slug("astatc-mlb-col-sf-2026-08-14-outs-kylfre-gte16") is None
    assert parse_macro_mlb_slug("tc-temp-nychigh-2026-08-14-gte86lt87f") is None


def test_mlb_prefixes_are_not_shadowed_by_a_bare_prefix_collision():
    """tsc-mlb/asc-mlb both contain a hyphen -- a naive single-segment
    prefix check (slug.split('-')[0]) would see "tsc"/"asc" and miss the
    category entirely. Longest-prefix-first ordering must catch these."""
    assert parse_macro_mlb_slug("tsc-mlb-az-bos-2026-08-17-3pt5")["category"] == "mlb_totals"
    assert parse_macro_mlb_slug("asc-mlb-az-bos-2026-08-17-neg-1pt5")["category"] == "mlb_spreads"


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


SLUG = "rdc-boc-2026-09-02-hike50"


def test_discover_tracks_active_buckets_and_ignores_untracked_categories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([make_market(SLUG), make_market("astatc-fwc-arg-sui")])
    logger_ = MacroMlbQuoteLogger(client)

    logger_._discover()

    assert list(logger_._active) == [SLUG]


def test_quotes_are_logged_with_parsed_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient(
        [make_market(SLUG)],
        bbo={SLUG: {"marketData": {
            "bestBid": {"value": "0.01"}, "bestAsk": {"value": "0.04"},
            "bidDepth": 5, "askDepth": 8,
        }}},
    )
    logger_ = MacroMlbQuoteLogger(client)
    logger_.run_cycle()

    from src.trading.polymarket_us.macro_mlb_quote_logger import QUOTES_PATH
    rows = [json.loads(line) for line in QUOTES_PATH.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["slug"] == SLUG
    assert rows[0]["category"] == "rate_decision"
    assert rows[0]["bid"] == 1
    assert rows[0]["ask"] == 4


def test_a_slug_with_both_ask_and_bid_is_never_logged_again(tmp_path, monkeypatch):
    """Same regression this exact bug already caused once in
    sports_prop_quote_logger.py (60MB/228k lines in a day) -- built correctly
    here from the start instead of needing a second fix."""
    monkeypatch.chdir(tmp_path)
    client = FakeClient(
        [make_market(SLUG)],
        bbo={SLUG: {"marketData": {
            "bestBid": {"value": "0.01"}, "bestAsk": {"value": "0.04"}, "bidDepth": 5, "askDepth": 8,
        }}},
    )
    logger_ = MacroMlbQuoteLogger(client)

    logger_._discover()
    logger_._log_quotes()
    assert client.bbo_calls == 1

    logger_._log_quotes()
    assert client.bbo_calls == 1  # never fetched again


def test_capture_state_is_seeded_from_disk_on_startup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from src.trading.polymarket_us.macro_mlb_quote_logger import QUOTES_PATH
    QUOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUOTES_PATH.write_text(json.dumps({"slug": SLUG, "bid": 1, "ask": 4}) + "\n")

    client = FakeClient([make_market(SLUG)])
    logger_ = MacroMlbQuoteLogger(client)

    assert SLUG in logger_._have_ask
    assert SLUG in logger_._have_bid


def test_bucket_disappearing_moves_to_pending_settlement(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([make_market(SLUG)])
    logger_ = MacroMlbQuoteLogger(client)
    logger_._discover()
    assert SLUG in logger_._active

    client._markets = []
    logger_._discover()

    assert SLUG not in logger_._active
    assert SLUG in logger_._pending_settlement


def test_resolved_settlement_is_logged_and_cleared_from_pending(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([make_market(SLUG)], settlements={SLUG: {"settlement": 1}})
    logger_ = MacroMlbQuoteLogger(client)
    logger_._discover()
    assert SLUG in logger_._active

    client._markets = []  # the bucket closed -- no longer in the active listing
    logger_._discover()
    assert SLUG in logger_._pending_settlement

    logger_._check_settlements()

    from src.trading.polymarket_us.macro_mlb_quote_logger import SETTLEMENTS_PATH
    rows = [json.loads(line) for line in SETTLEMENTS_PATH.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["slug"] == SLUG
    assert rows[0]["outcome_yes"] is True
    assert SLUG not in logger_._pending_settlement


def test_unsettled_bucket_stays_pending_without_writing_a_row(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([make_market(SLUG)], settlement_errors={SLUG})
    logger_ = MacroMlbQuoteLogger(client)
    logger_._discover()

    client._markets = []
    logger_._discover()

    logger_._check_settlements()

    from src.trading.polymarket_us.macro_mlb_quote_logger import SETTLEMENTS_PATH
    assert not SETTLEMENTS_PATH.exists()
    assert SLUG in logger_._pending_settlement

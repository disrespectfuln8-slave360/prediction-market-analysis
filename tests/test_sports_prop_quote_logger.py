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


def test_rejects_nfl_now_that_its_tracked_by_its_own_module():
    """NFL split out into nfl_prop_quote_logger.py 2026-08-17 -- this
    module's job is MLB only now, even though the regex shape would
    otherwise match an NFL slug fine."""
    assert parse_prop_slug("astatc-nfl-lac-hou-2026-08-13-td-trelan-gte1") is None


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


def test_a_slug_with_both_ask_and_bid_is_never_logged_again(tmp_path, monkeypatch):
    """Regression: _build_calibration_table only ever reads each slug's
    FIRST recorded ask and FIRST recorded bid -- every quote after that was
    logged but never used for anything. Left unbounded this grew to
    60MB/228k lines in one day (confirmed live 2026-08-15), and re-parsing
    that on every calibration reload was the single largest memory consumer
    on the machine. Once a slug has both, it must stop being fetched
    entirely, not just stop being written."""
    monkeypatch.chdir(tmp_path)
    client = FakeClient(
        [make_market(SLUG)],
        bbo={SLUG: {"marketData": {
            "bestBid": {"value": "0.15"}, "bestAsk": {"value": "0.18"},
            "bidDepth": 5, "askDepth": 8,
        }}},
    )
    logger_ = SportsPropQuoteLogger(client)
    logger_._active = {SLUG: parse_prop_slug(SLUG)}

    logger_._log_quotes()
    assert client.bbo_calls == 1
    assert SLUG in logger_._have_ask and SLUG in logger_._have_bid

    logger_._log_quotes()  # second cycle, same active bucket
    assert client.bbo_calls == 1  # never fetched again


def test_a_slug_with_only_one_side_known_keeps_being_checked(tmp_path, monkeypatch):
    """A one-sided book (bid known, ask still None) must keep polling until
    the missing side actually shows up -- only fully-captured slugs stop."""
    monkeypatch.chdir(tmp_path)
    client = FakeClient(
        [make_market(SLUG)],
        bbo={SLUG: {"marketData": {
            "bestBid": {"value": "0.15"}, "bestAsk": None, "bidDepth": 5, "askDepth": 0,
        }}},
    )
    logger_ = SportsPropQuoteLogger(client)
    logger_._active = {SLUG: parse_prop_slug(SLUG)}

    logger_._log_quotes()
    assert client.bbo_calls == 1
    assert SLUG in logger_._have_bid
    assert SLUG not in logger_._have_ask

    logger_._log_quotes()
    assert client.bbo_calls == 2  # still polled -- ask side still unknown


def test_capture_state_is_seeded_from_disk_on_startup(tmp_path, monkeypatch):
    """A restart must not forget which slugs are already fully captured and
    start re-logging them -- the exact regrowth this whole fix prevents."""
    monkeypatch.chdir(tmp_path)
    from src.trading.polymarket_us.sports_prop_quote_logger import QUOTES_PATH
    QUOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUOTES_PATH.write_text(json.dumps({"slug": SLUG, "bid": 15, "ask": 18}) + "\n")

    client = FakeClient([make_market(SLUG)])
    logger_ = SportsPropQuoteLogger(client)

    assert SLUG in logger_._have_ask
    assert SLUG in logger_._have_bid


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


# ── Settlement verification against the real MLB Stats API ─────────────────
# Real incident 2026-08-18: polymarket.us's own settlement determination was
# confirmed to have a real, ~14% one-directional error rate (players who did
# not play, or played and fell short, still settled YES). These tests cover
# the override wiring itself with an injected fake verifier -- the real
# verification logic (mlb_stats_verification.py) is covered by its own
# dedicated, network-free unit tests in test_mlb_stats_verification.py.

def test_settlement_verification_overrides_a_wrong_exchange_settlement(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([make_market(SLUG)], settlements={SLUG: {"settlement": 1}})  # exchange says YES
    logger_ = SportsPropQuoteLogger(client, verify_settlement_fn=lambda *a, **k: False)  # verified: really NO
    logger_._pending_settlement[SLUG] = (parse_prop_slug(SLUG), 0)

    logger_._check_settlements()

    from src.trading.polymarket_us.sports_prop_quote_logger import SETTLEMENTS_PATH
    rows = [json.loads(line) for line in SETTLEMENTS_PATH.read_text().splitlines()]
    assert rows[0]["outcome_yes"] is False  # the verified value won, not the exchange's


def test_settlement_verification_agreeing_keeps_the_exchange_value(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([make_market(SLUG)], settlements={SLUG: {"settlement": 1}})
    logger_ = SportsPropQuoteLogger(client, verify_settlement_fn=lambda *a, **k: True)
    logger_._pending_settlement[SLUG] = (parse_prop_slug(SLUG), 0)

    logger_._check_settlements()

    from src.trading.polymarket_us.sports_prop_quote_logger import SETTLEMENTS_PATH
    rows = [json.loads(line) for line in SETTLEMENTS_PATH.read_text().splitlines()]
    assert rows[0]["outcome_yes"] is True


def test_settlement_verification_inconclusive_falls_back_to_the_exchange_value(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([make_market(SLUG)], settlements={SLUG: {"settlement": 1}})
    logger_ = SportsPropQuoteLogger(client, verify_settlement_fn=lambda *a, **k: None)  # can't determine
    logger_._pending_settlement[SLUG] = (parse_prop_slug(SLUG), 0)

    logger_._check_settlements()

    from src.trading.polymarket_us.sports_prop_quote_logger import SETTLEMENTS_PATH
    rows = [json.loads(line) for line in SETTLEMENTS_PATH.read_text().splitlines()]
    assert rows[0]["outcome_yes"] is True  # exchange's own settlement, unverified but not overridden


def test_settlement_verification_raising_falls_back_to_the_exchange_value(tmp_path, monkeypatch):
    """A verification bug or transient network failure must never take
    trading data down with it -- same fail-open-to-old-behavior posture as
    every other best-effort check in this codebase."""
    monkeypatch.chdir(tmp_path)
    def _boom(*a, **k):
        raise RuntimeError("MLB Stats API is down")
    client = FakeClient([make_market(SLUG)], settlements={SLUG: {"settlement": 1}})
    logger_ = SportsPropQuoteLogger(client, verify_settlement_fn=_boom)
    logger_._pending_settlement[SLUG] = (parse_prop_slug(SLUG), 0)

    logger_._check_settlements()

    from src.trading.polymarket_us.sports_prop_quote_logger import SETTLEMENTS_PATH
    rows = [json.loads(line) for line in SETTLEMENTS_PATH.read_text().splitlines()]
    assert rows[0]["outcome_yes"] is True


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

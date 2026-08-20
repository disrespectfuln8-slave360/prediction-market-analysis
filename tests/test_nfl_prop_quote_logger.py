"""Tests for the NFL per-game player-prop quote+outcome logger.

Split out of test_sports_prop_quote_logger.py 2026-08-17 -- see
nfl_prop_quote_logger.py's module docstring for why NFL got its own module
and data files, and why only the per-game matchup shape (of three real NFL
shapes surveyed) is tracked at all.
"""

from __future__ import annotations

import json

from src.trading.polymarket_us.models import Market
from src.trading.polymarket_us.nfl_prop_quote_logger import (
    NflPropQuoteLogger,
    parse_nfl_prop_slug,
)


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
        yes_bid=50,
        yes_ask=52,
        no_bid=48,
        no_ask=50,
        volume=0.0,
        volume_24h=0.0,
        bid_depth=10,
        ask_depth=10,
    )


# ── Slug parsing ─────────────────────────────────────────────────────────────


def test_parses_nfl_touchdown_prop():
    meta = parse_nfl_prop_slug("astatc-nfl-lac-hou-2026-08-13-td-trelan-gte1")
    assert meta == {
        "sport": "nfl",
        "matchup": "lac-hou",
        "date": "2026-08-13",
        "stat": "td",
        "player": "trelan",
        "threshold": 1,
    }


def test_parses_real_preseason_slug_shapes():
    """Real rows migrated in from the shared file's 2026-08-14/15 preseason
    collection -- confirms the shape this module inherited still parses."""
    meta = parse_nfl_prop_slug("astatc-nfl-tb-nyj-2026-08-14-pyd-gensmi-gte50")
    assert meta == {
        "sport": "nfl",
        "matchup": "tb-nyj",
        "date": "2026-08-14",
        "stat": "pyd",
        "player": "gensmi",
        "threshold": 50,
    }


def test_rejects_mlb_now_that_its_tracked_by_the_shared_module():
    assert parse_nfl_prop_slug("astatc-mlb-col-sf-2026-08-14-outs-kylfre-gte16") is None


def test_rejects_the_other_two_real_nfl_shapes_surveyed_and_excluded():
    """Real, currently-active inventory this module deliberately does not
    track -- see module docstring. Season-long stat totals have no
    threshold in the slug at all; margin buckets are categorical, not a
    threshold ladder."""
    assert parse_nfl_prop_slug("astatc-nfl-passtds-ou-2027-01-10-tylsho") is None
    assert parse_nfl_prop_slug("astatc-nfl-lv-hou-2026-08-20-margin-long-25-plus") is None


def test_rejects_non_astatc_slugs():
    assert parse_nfl_prop_slug("tc-temp-nychigh-2026-08-14-gte86lt87f") is None


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


SLUG = "astatc-nfl-den-atl-2026-08-14-ptd-jarsti-gte1"


def test_discover_tracks_active_buckets_and_ignores_other_sports(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([make_market(SLUG), make_market("astatc-mlb-col-sf-2026-08-14-outs-kylfre-gte16")])
    logger_ = NflPropQuoteLogger(client)

    logger_._discover()

    assert list(logger_._active) == [SLUG]


def test_quotes_are_logged_with_parsed_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient(
        [make_market(SLUG)],
        bbo={
            SLUG: {
                "marketData": {
                    "bestBid": {"value": "0.15"},
                    "bestAsk": {"value": "0.18"},
                    "bidDepth": 5,
                    "askDepth": 8,
                }
            }
        },
    )
    logger_ = NflPropQuoteLogger(client)
    logger_.run_cycle()

    from src.trading.polymarket_us.nfl_prop_quote_logger import QUOTES_PATH

    rows = [json.loads(line) for line in QUOTES_PATH.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["slug"] == SLUG
    assert rows[0]["sport"] == "nfl"
    assert rows[0]["stat"] == "ptd"
    assert rows[0]["bid"] == 15
    assert rows[0]["ask"] == 18


def test_a_slug_with_both_ask_and_bid_is_never_logged_again(tmp_path, monkeypatch):
    """Same regression sports_prop_quote_logger.py hit in production
    (60MB/228k lines in a day) -- built correctly here from the start."""
    monkeypatch.chdir(tmp_path)
    client = FakeClient(
        [make_market(SLUG)],
        bbo={
            SLUG: {
                "marketData": {
                    "bestBid": {"value": "0.15"},
                    "bestAsk": {"value": "0.18"},
                    "bidDepth": 5,
                    "askDepth": 8,
                }
            }
        },
    )
    logger_ = NflPropQuoteLogger(client)
    logger_._active = {SLUG: parse_nfl_prop_slug(SLUG)}

    logger_._log_quotes()
    assert client.bbo_calls == 1
    assert SLUG in logger_._have_ask and SLUG in logger_._have_bid

    logger_._log_quotes()
    assert client.bbo_calls == 1  # never fetched again


def test_capture_state_is_seeded_from_disk_on_startup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from src.trading.polymarket_us.nfl_prop_quote_logger import QUOTES_PATH

    QUOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUOTES_PATH.write_text(json.dumps({"slug": SLUG, "bid": 15, "ask": 18}) + "\n")

    client = FakeClient([make_market(SLUG)])
    logger_ = NflPropQuoteLogger(client)

    assert SLUG in logger_._have_ask
    assert SLUG in logger_._have_bid


def test_bucket_disappearing_moves_to_pending_settlement(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([make_market(SLUG)])
    logger_ = NflPropQuoteLogger(client)
    logger_._discover()
    assert SLUG in logger_._active

    client._markets = []
    logger_._discover()

    assert SLUG not in logger_._active
    assert SLUG in logger_._pending_settlement


def test_resolved_settlement_is_logged_and_cleared_from_pending(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([make_market(SLUG)], settlements={SLUG: {"settlement": 1}})
    logger_ = NflPropQuoteLogger(client)
    logger_._discover()
    assert SLUG in logger_._active

    client._markets = []
    logger_._discover()
    assert SLUG in logger_._pending_settlement

    logger_._check_settlements()

    from src.trading.polymarket_us.nfl_prop_quote_logger import SETTLEMENTS_PATH

    rows = [json.loads(line) for line in SETTLEMENTS_PATH.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["slug"] == SLUG
    assert rows[0]["outcome_yes"] is True
    assert SLUG not in logger_._pending_settlement


# ── Settlement verification against real ESPN box scores ───────────────────
# Same wiring as sports_prop_quote_logger.py's MLB-side (see that file's
# tests for the fuller rationale). NFL's own spot-check came back clean, but
# the override plumbing itself is tested here regardless.


def test_settlement_verification_overrides_a_wrong_exchange_settlement(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([make_market(SLUG)], settlements={SLUG: {"settlement": 1}})
    logger_ = NflPropQuoteLogger(client, verify_settlement_fn=lambda *a, **k: False)
    logger_._discover()
    client._markets = []
    logger_._discover()

    logger_._check_settlements()

    from src.trading.polymarket_us.nfl_prop_quote_logger import SETTLEMENTS_PATH

    rows = [json.loads(line) for line in SETTLEMENTS_PATH.read_text().splitlines()]
    assert rows[0]["outcome_yes"] is False


def test_settlement_verification_inconclusive_falls_back_to_the_exchange_value(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([make_market(SLUG)], settlements={SLUG: {"settlement": 1}})
    logger_ = NflPropQuoteLogger(client, verify_settlement_fn=lambda *a, **k: None)
    logger_._discover()
    client._markets = []
    logger_._discover()

    logger_._check_settlements()

    from src.trading.polymarket_us.nfl_prop_quote_logger import SETTLEMENTS_PATH

    rows = [json.loads(line) for line in SETTLEMENTS_PATH.read_text().splitlines()]
    assert rows[0]["outcome_yes"] is True


def test_settlement_verification_raising_falls_back_to_the_exchange_value(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def _boom(*a, **k):
        raise RuntimeError("ESPN is down")

    client = FakeClient([make_market(SLUG)], settlements={SLUG: {"settlement": 1}})
    logger_ = NflPropQuoteLogger(client, verify_settlement_fn=_boom)
    logger_._discover()
    client._markets = []
    logger_._discover()

    logger_._check_settlements()

    from src.trading.polymarket_us.nfl_prop_quote_logger import SETTLEMENTS_PATH

    rows = [json.loads(line) for line in SETTLEMENTS_PATH.read_text().splitlines()]
    assert rows[0]["outcome_yes"] is True


def test_unsettled_bucket_stays_pending_without_writing_a_row(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([make_market(SLUG)], settlement_errors={SLUG})
    logger_ = NflPropQuoteLogger(client)
    logger_._discover()

    client._markets = []
    logger_._discover()

    logger_._check_settlements()

    from src.trading.polymarket_us.nfl_prop_quote_logger import SETTLEMENTS_PATH

    assert not SETTLEMENTS_PATH.exists()
    assert SLUG in logger_._pending_settlement

"""Tests for the sports-props scan/decide/execute loop.

Mirrors test_polymarket_us_apex.py's style closely -- same fake client
shape, same FakeGrounding-equivalent pattern -- since sports_props_bot.py
itself reuses apex.risk and apex.memory.ApexMemory directly rather than
re-deriving the same money-critical logic.

Every test that reaches _scan_once() (and so _reconcile_live_positions())
uses monkeypatch.chdir(tmp_path) -- _ladder_bot_held_slugs() reads a real
relative path (bot.py's HELD_SLUGS_PATH) that, left pointed at the actual
repo root, would read whatever the real live ladder bot currently has
published on disk. Added 2026-08-17 alongside that read, to keep tests from
silently depending on live production state.
"""
from __future__ import annotations

import json

import pytest

from src.trading.polymarket_us import sports_props_bot as bot_module
from src.trading.polymarket_us.apex.memory import ApexMemory
from src.trading.polymarket_us.models import Market, Order, Position
from src.trading.polymarket_us.sports_prop_calibration import SportsPropGrounding
from src.trading.polymarket_us.sports_props_bot import SportsPropsBot


def make_market(slug, yes_bid, yes_ask, ask_depth=50, bid_depth=50, closed=False):
    return Market(
        slug=slug, id=slug, title=slug, category="sports", tags=[],
        active=not closed, closed=closed, close_time=None,
        yes_bid=yes_bid, yes_ask=yes_ask,
        no_bid=(100 - yes_ask) if yes_ask is not None else None,
        no_ask=(100 - yes_bid) if yes_bid is not None else None,
        volume=0.0, volume_24h=0.0,
        bid_depth=bid_depth, ask_depth=ask_depth,
    )


def make_grounding(sport, stat, side, price_cents, trustworthy=True, actual_rate=None,
                    note="calibration OK: 40 samples", count=40):
    return SportsPropGrounding(
        sport=sport, stat=stat, side=side, price_cents=price_cents,
        implied_prob=price_cents / 100.0,
        calibration_trustworthy=trustworthy, calibration_note=note,
        calibration_sample_count=count,
        calibration_actual_rate=actual_rate,
    )


class FakeCalibration:
    def __init__(self, groundings, stale=False):
        self._groundings = groundings  # (sport, stat, side, price_cents) -> SportsPropGrounding
        self._stale = stale
        self.reload_calls = 0

    def reload(self):
        self.reload_calls += 1

    def data_is_stale(self):
        return self._stale

    def evaluate(self, sport, stat, side, price_cents):
        key = (sport, stat, side, price_cents)
        if key in self._groundings:
            return self._groundings[key]
        return make_grounding(sport, stat, side, price_cents, trustworthy=False,
                               note="no data", count=0)


class FakeClient:
    def __init__(self, markets, fills=None, balance=100_000, settlements=None, requote_markets=None):
        self._markets = {m.slug: m for m in markets}
        self._fills = fills or {}
        self._balance = balance
        self._settlements = settlements or {}
        # Optional: a slug present here is what get_market() returns instead
        # of the bulk-scan snapshot -- simulates the price having moved
        # between _best_bucket_for_game's scan and _requote_bucket's re-fetch.
        self._requote_markets = requote_markets or {}
        self.positions: dict[str, float] = {}
        self.orders: list[tuple] = []

    def get_balance(self):
        return self._balance

    def get_positions(self):
        return [Position(market_slug=s, position=p) for s, p in self.positions.items() if p != 0]

    def get_market(self, slug):
        return self._requote_markets.get(slug, self._markets[slug])

    def get_settlement(self, slug):
        return self._settlements.get(slug)

    def list_all_markets(self, **kw):
        return list(self._markets.values())

    def place_order(self, market_slug, side, count, yes_price, immediate_or_cancel=False):
        self.orders.append((market_slug, side, count, yes_price, immediate_or_cancel))
        filled = min(count, self._fills.get(market_slug, count))
        signed = filled if side == "yes" else -filled
        self.positions[market_slug] = self.positions.get(market_slug, 0) + signed
        return Order(order_id="o-" + market_slug, market_slug=market_slug, side=side,
                     count=count, price_cents=yes_price, status="unknown")


def make_bot(client, calibration, memory, allowed_sports=("mlb", "nfl")):
    bot_module.FILL_POLL_DELAY = 0
    return SportsPropsBot(client=client, memory=memory, calibration=calibration, poll_interval=1,
                           allowed_sports=allowed_sports)


def _publish_ladder_holdings(slugs) -> None:
    """Write bot.py's HELD_SLUGS_PATH the same shape LadderArbBot._publish_
    held_slugs() does, for tests exercising sports_props_bot's read side.
    Caller must have already monkeypatch.chdir'd into an isolated tmp_path."""
    from datetime import datetime, timezone

    from src.trading.polymarket_us.bot import HELD_SLUGS_PATH
    HELD_SLUGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    HELD_SLUGS_PATH.write_text(json.dumps({
        "updated_ts": datetime.now(timezone.utc).isoformat(),
        "slugs": sorted(slugs),
    }))


SLUG = "astatc-mlb-nyy-bos-2026-08-15-hits-player1-gte1"


def test_scan_buys_yes_when_calibration_shows_a_trustworthy_edge(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = make_grounding("mlb", "hits", "yes", 40, actual_rate=70.0)
    calibration = FakeCalibration({("mlb", "hits", "yes", 40): grounding})
    client = FakeClient([market], fills={SLUG: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert len(client.orders) == 1
    slug, side, count, yes_price, ioc = client.orders[0]
    assert (slug, side, yes_price, ioc) == (SLUG, "yes", 40, True)
    assert len(memory.open_trades()) == 1
    assert memory.open_trades()[0]["side"] == "yes"


def test_scan_buys_no_at_the_yes_equivalent_price(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    market = make_market(SLUG, yes_bid=70, yes_ask=95)
    # NO costs 100-70=30; grounding keyed on the NO cost price.
    grounding = make_grounding("mlb", "hits", "no", 30, actual_rate=70.0)
    calibration = FakeCalibration({("mlb", "hits", "no", 30): grounding})
    client = FakeClient([market], fills={SLUG: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert len(client.orders) == 1
    slug, side, count, yes_price, ioc = client.orders[0]
    assert (slug, side, yes_price) == (SLUG, "no", 70)
    trade = memory.open_trades()[0]
    assert trade["side"] == "no"
    assert trade["price_cents"] == 30


def test_scan_reevaluates_calibration_at_the_requoted_price_not_the_stale_scan_price(tmp_path, monkeypatch):
    """Regression: model_prob used to come from the grounding computed
    against the bulk-scan price, never re-checked against the price the
    order actually requotes to. calibration_actual_rate is scoped to a
    10-cent decile, so a price that moved deciles between the scan and the
    requote (thin sports-prop books move fast) would size a real trade off
    a probability measured for a decile it isn't even buying into anymore.
    Here the bulk scan sees yes_ask=40 (a great, trustworthy 90% decile),
    but the price has actually moved to 70 by requote time, where there's
    no calibration data at all -- the trade must be skipped, not placed
    using the stale 40c grounding."""
    monkeypatch.chdir(tmp_path)
    scan_market = make_market(SLUG, yes_bid=38, yes_ask=40)
    requoted_market = make_market(SLUG, yes_bid=68, yes_ask=70)
    grounding_at_scan_price = make_grounding("mlb", "hits", "yes", 40, actual_rate=90.0)
    calibration = FakeCalibration({("mlb", "hits", "yes", 40): grounding_at_scan_price})
    client = FakeClient([scan_market], fills={SLUG: 99}, requote_markets={SLUG: requoted_market})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert client.orders == []  # no calibration data at 70c -- must not trade on the stale 40c grounding
    assert memory.open_trades() == []


def test_scan_passes_when_calibration_is_untrustworthy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    calibration = FakeCalibration({})  # nothing trustworthy anywhere
    client = FakeClient([market], fills={SLUG: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert client.orders == []
    assert memory.open_trades() == []


def test_scan_never_stacks_a_second_position_in_the_same_game(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    held_slug = "astatc-mlb-nyy-bos-2026-08-15-tb-player2-gte1"
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = make_grounding("mlb", "hits", "yes", 40, actual_rate=70.0)
    calibration = FakeCalibration({("mlb", "hits", "yes", 40): grounding})
    client = FakeClient([market, make_market(held_slug, yes_bid=50, yes_ask=52)], fills={SLUG: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)
    memory.record_open(
        market_slug=held_slug, side="yes", count=1.0, price_cents=52, cost_cents=52,
        model_prob=0.6, market_implied_prob=0.52, edge_pct=0.08, kelly_fraction_used=0.05,
    )

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert client.orders == []  # held_slug is the SAME game (nyy-bos, 2026-08-15) as SLUG
    assert len(memory.open_trades()) == 1


def test_different_games_can_both_be_traded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    other_slug = "astatc-mlb-lad-sfo-2026-08-15-hits-player3-gte1"
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    # A different price decile than SLUG's (70c, not 40c) -- otherwise this
    # is exactly the bucket-collision the guard below exists to catch, which
    # isn't what THIS test is checking (see test_scan_blocks_a_second_game_
    # landing_in_the_same_calibration_bucket for that). actual_rate=90 at
    # 70c keeps a clear, tradeable edge (a 70/70 combination would price in
    # ~zero edge and get skipped for an unrelated reason).
    other_market = make_market(other_slug, yes_bid=68, yes_ask=70)
    grounding = make_grounding("mlb", "hits", "yes", 40, actual_rate=70.0)
    other_grounding = make_grounding("mlb", "hits", "yes", 70, actual_rate=90.0)
    calibration = FakeCalibration({
        ("mlb", "hits", "yes", 40): grounding,
        ("mlb", "hits", "yes", 70): other_grounding,
    })
    client = FakeClient([market, other_market], fills={SLUG: 99, other_slug: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)
    memory.record_open(
        # Different DATE, different game, AND a third, distinct price decile
        # (90c -- not SLUG's 40c or other_slug's 70c) -- same reasoning as
        # above, stay out of both candidates' buckets.
        market_slug="astatc-mlb-nyy-bos-2026-08-14-hits-playerx-gte1",
        side="yes", count=1.0, price_cents=90, cost_cents=90,
        model_prob=0.9, market_implied_prob=0.9, edge_pct=0.3, kelly_fraction_used=0.05,
    )

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert len(client.orders) == 2


# ── Bucket-correlation guard ──────────────────────────────────────────────────
# Real incident 2026-08-17: 4 of 6 real trades that day all drew on the
# identical (mlb, tb, yes, 10-20%) calibration bucket and lost together --
# not four independent bets, the same underlying signal fired four times.
# Orthogonal to the per-game guard above: these are DIFFERENT games.

def test_scan_blocks_a_second_game_landing_in_the_same_calibration_bucket(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    other_slug = "astatc-mlb-lad-sfo-2026-08-15-tb-player3-gte1"
    # Same stat/side/price on two DIFFERENT games -- same bucket_key.
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    other_market = make_market(other_slug, yes_bid=38, yes_ask=40)
    grounding = make_grounding("mlb", "hits", "yes", 40, actual_rate=70.0)
    calibration = FakeCalibration({("mlb", "hits", "yes", 40): grounding})
    client = FakeClient([market, other_market], fills={SLUG: 99, other_slug: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert len(client.orders) == 1  # the second game's identical-bucket candidate was skipped


def test_an_already_open_position_blocks_a_new_bucket_match(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    held_slug = "astatc-mlb-lad-sfo-2026-08-14-hits-player2-gte1"  # different game, different day
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = make_grounding("mlb", "hits", "yes", 40, actual_rate=70.0)
    calibration = FakeCalibration({("mlb", "hits", "yes", 40): grounding})
    client = FakeClient([market], fills={SLUG: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)
    memory.record_open(
        market_slug=held_slug, side="yes", count=1.0, price_cents=40, cost_cents=40,
        model_prob=0.7, market_implied_prob=0.4, edge_pct=0.3, kelly_fraction_used=0.05,
    )

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert client.orders == []  # SLUG is a different game, but the same (mlb, hits, yes, 40c-decile) bucket
    assert len(memory.open_trades()) == 1


def test_different_deciles_of_the_same_stat_are_independent_bets(tmp_path, monkeypatch):
    """The guard must not overreach -- two candidates that happen to share a
    (sport, stat, side) but land in DIFFERENT price deciles are genuinely
    different calibration signals and must both be tradeable."""
    monkeypatch.chdir(tmp_path)
    other_slug = "astatc-mlb-lad-sfo-2026-08-15-hits-player3-gte1"
    market = make_market(SLUG, yes_bid=38, yes_ask=40)  # 40c -> 40-50% decile
    other_market = make_market(other_slug, yes_bid=68, yes_ask=70)  # 70c -> 70-80% decile
    grounding = make_grounding("mlb", "hits", "yes", 40, actual_rate=70.0)
    other_grounding = make_grounding("mlb", "hits", "yes", 70, actual_rate=75.0)
    calibration = FakeCalibration({
        ("mlb", "hits", "yes", 40): grounding,
        ("mlb", "hits", "yes", 70): other_grounding,
    })
    client = FakeClient([market, other_market], fills={SLUG: 99, other_slug: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert len(client.orders) == 2


def test_terminated_bot_only_observes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = make_grounding("mlb", "hits", "yes", 40, actual_rate=70.0)
    calibration = FakeCalibration({("mlb", "hits", "yes", 40): grounding})
    client = FakeClient([market], fills={SLUG: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)
    memory.terminated_reason = "bankroll $0.50 at or below the $1.00 floor"

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert client.orders == []


def test_stale_calibration_blocks_trading(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = make_grounding("mlb", "hits", "yes", 40, actual_rate=70.0)
    calibration = FakeCalibration({("mlb", "hits", "yes", 40): grounding}, stale=True)
    client = FakeClient([market], fills={SLUG: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert client.orders == []
    assert calibration.reload_calls == 1


def test_settlement_reconciliation_pays_out_and_logs_a_miss_lesson(tmp_path):
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)
    memory.record_open(
        market_slug=SLUG, side="yes", count=1.0, price_cents=40, cost_cents=40,
        model_prob=0.9, market_implied_prob=0.4, edge_pct=0.4, kelly_fraction_used=0.05,
    )
    market = make_market(SLUG, yes_bid=0, yes_ask=100, closed=True)
    client = FakeClient([market], settlements={SLUG: {"settlement": 0}})

    bot = SportsPropsBot(client=client, memory=memory, calibration=FakeCalibration({}))
    bot._reconcile_settlements()

    assert memory.open_trades() == []
    settled = memory.trade_log[-1]
    assert settled["status"] == "settled"
    assert settled["pnl_cents"] == -40
    assert memory.bankroll_cents == 960
    assert len(memory.lessons) == 1


def test_untracked_live_position_halts_before_trading(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stray_slug = "astatc-mlb-lad-sfo-2026-08-14-tb-playerz-gte1"
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = make_grounding("mlb", "hits", "yes", 40, actual_rate=70.0)
    calibration = FakeCalibration({("mlb", "hits", "yes", 40): grounding})
    client = FakeClient([market], fills={SLUG: 99})
    client.positions[stray_slug] = 2.0
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert client.orders == []
    assert memory.is_halted()
    assert stray_slug in memory.anomaly_halted_reason


def test_scan_recognizes_nfl_slugs_via_the_dedicated_nfl_parser(tmp_path, monkeypatch):
    """NFL split into its own nfl_prop_quote_logger.py 2026-08-17 -- this
    bot's own market-scanning must still recognize NFL per-game matchup
    slugs via _parse_any_prop_slug, even though parse_prop_slug (the shared
    MLB-only module) no longer does."""
    monkeypatch.chdir(tmp_path)
    nfl_slug = "astatc-nfl-den-atl-2026-08-14-ptd-jarsti-gte1"
    market = make_market(nfl_slug, yes_bid=38, yes_ask=40)
    grounding = make_grounding("nfl", "ptd", "yes", 40, actual_rate=70.0)
    calibration = FakeCalibration({("nfl", "ptd", "yes", 40): grounding})
    client = FakeClient([market], fills={nfl_slug: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert len(client.orders) == 1
    slug, side, count, yes_price, ioc = client.orders[0]
    assert (slug, side, yes_price) == (nfl_slug, "yes", 40)
    assert memory.open_trades()[0]["side"] == "yes"


def test_mlb_and_nfl_games_on_the_same_day_are_independent_slots(tmp_path, monkeypatch):
    """The one-position-per-game guard is keyed on (sport, matchup, date) --
    an NFL game must never be treated as "the same game" as an MLB one just
    because both happen to share a date."""
    monkeypatch.chdir(tmp_path)
    nfl_slug = "astatc-nfl-den-atl-2026-08-15-ptd-jarsti-gte1"
    mlb_market = make_market(SLUG, yes_bid=38, yes_ask=40)  # SLUG's date is 2026-08-15
    nfl_market = make_market(nfl_slug, yes_bid=38, yes_ask=40)
    grounding = make_grounding("mlb", "hits", "yes", 40, actual_rate=70.0)
    nfl_grounding = make_grounding("nfl", "ptd", "yes", 40, actual_rate=70.0)
    calibration = FakeCalibration({
        ("mlb", "hits", "yes", 40): grounding,
        ("nfl", "ptd", "yes", 40): nfl_grounding,
    })
    client = FakeClient([mlb_market, nfl_market], fills={SLUG: 99, nfl_slug: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert len(client.orders) == 2


def test_allowed_sports_filters_other_sports_out_of_trading(tmp_path, monkeypatch):
    """As of 2026-08-19, MLB and NFL each run as their own SportsPropsBot
    instance via allowed_sports. An MLB-only instance must never trade an
    NFL candidate even if it's the only market it sees, and vice versa."""
    monkeypatch.chdir(tmp_path)
    nfl_slug = "astatc-nfl-den-atl-2026-08-14-ptd-jarsti-gte1"
    market = make_market(nfl_slug, yes_bid=38, yes_ask=40)
    grounding = make_grounding("nfl", "ptd", "yes", 40, actual_rate=70.0)
    calibration = FakeCalibration({("nfl", "ptd", "yes", 40): grounding})
    client = FakeClient([market], fills={nfl_slug: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, calibration, memory, allowed_sports=("mlb",))
    bot._scan_once()

    assert len(client.orders) == 0
    assert memory.open_trades() == []


def test_allowed_sports_scopes_the_stray_position_check(tmp_path, monkeypatch):
    """A real MLB position on the shared account belongs to the sibling
    MLB-only SportsPropsBot instance, not this NFL-only one -- it must
    never register as stray/untracked here, the same way the ladder bot's
    and Apex's positions already don't."""
    monkeypatch.chdir(tmp_path)
    nfl_slug = "astatc-nfl-den-atl-2026-08-14-ptd-jarsti-gte1"
    market = make_market(nfl_slug, yes_bid=38, yes_ask=40)
    grounding = make_grounding("nfl", "ptd", "yes", 40, actual_rate=70.0)
    calibration = FakeCalibration({("nfl", "ptd", "yes", 40): grounding})
    client = FakeClient([market], fills={nfl_slug: 99})
    client.positions[SLUG] = 5.0  # a real MLB position, owned by the sibling bot
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, calibration, memory, allowed_sports=("nfl",))
    bot._scan_once()

    assert not memory.is_halted()
    assert len(client.orders) == 1


def test_reconciliation_ignores_other_bots_live_positions(tmp_path, monkeypatch):
    """Shared account with the ladder bot (astatc- arb locks that don't
    match this parser's shape) and Apex (tc-temp-* weather) -- neither
    should ever trigger a halt here."""
    monkeypatch.chdir(tmp_path)
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = make_grounding("mlb", "hits", "yes", 40, actual_rate=70.0)
    calibration = FakeCalibration({("mlb", "hits", "yes", 40): grounding})
    client = FakeClient([market], fills={SLUG: 99})
    client.positions["astatc-fwc-arg-sui"] = 5.0  # not this parser's shape
    client.positions["tc-temp-nychigh-2026-08-14-gte86lt87f"] = 3.0  # Apex's
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert not memory.is_halted()
    assert len(client.orders) == 1


# ── Cross-bot ambiguity: ladder bot positions in the same slug family ────────
# Real incident 2026-08-17: sports_props_bot halted on 6 of the ladder bot's
# own astatc-mlb-* lock legs, since (unlike Apex's tc-temp-* prefix) the
# ladder bot trades the identical slug family this bot does, and nothing
# distinguished "my own untracked trade" from "a sibling bot's legitimate
# one". Fixed by having the ladder bot publish its own held slugs (bot.py's
# HELD_SLUGS_PATH) for this bot to subtract before deciding what's stray.

def test_position_matching_the_ladder_bots_published_holdings_does_not_halt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ladder_slug = "astatc-mlb-az-bos-2026-08-17-er-mitbra-gte1"
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = make_grounding("mlb", "hits", "yes", 40, actual_rate=70.0)
    calibration = FakeCalibration({("mlb", "hits", "yes", 40): grounding})
    client = FakeClient([market], fills={SLUG: 99})
    client.positions[ladder_slug] = 1.0
    _publish_ladder_holdings({ladder_slug})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert not memory.is_halted()
    assert len(client.orders) == 1  # still trades its own genuine candidate


def test_a_position_not_in_the_ladder_bots_published_holdings_still_halts(tmp_path, monkeypatch):
    """The fix must not blind this check entirely -- a position in the same
    astatc-mlb-* family that the ladder bot does NOT claim is still a real,
    unexplained anomaly and must still halt."""
    monkeypatch.chdir(tmp_path)
    ladder_slug = "astatc-mlb-az-bos-2026-08-17-er-mitbra-gte1"
    genuinely_stray_slug = "astatc-mlb-lad-sfo-2026-08-14-tb-playerz-gte1"
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = make_grounding("mlb", "hits", "yes", 40, actual_rate=70.0)
    calibration = FakeCalibration({("mlb", "hits", "yes", 40): grounding})
    client = FakeClient([market], fills={SLUG: 99})
    client.positions[genuinely_stray_slug] = 2.0
    _publish_ladder_holdings({ladder_slug})  # doesn't mention genuinely_stray_slug
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert memory.is_halted()
    assert genuinely_stray_slug in memory.anomaly_halted_reason


def test_stale_ladder_publish_file_falls_back_to_the_cautious_default(tmp_path, monkeypatch):
    """A ladder-bot publish file that hasn't been updated recently (crashed,
    stopped, scheduled task disabled) must not be trusted forever -- falls
    back to treating the position as potentially stray, same as if the file
    didn't exist at all."""
    monkeypatch.chdir(tmp_path)
    from datetime import datetime, timedelta, timezone

    from src.trading.polymarket_us.bot import HELD_SLUGS_PATH
    from src.trading.polymarket_us.sports_props_bot import LADDER_HELD_SLUGS_STALE_SECS

    ladder_slug = "astatc-mlb-az-bos-2026-08-17-er-mitbra-gte1"
    HELD_SLUGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    stale_ts = datetime.now(timezone.utc) - timedelta(seconds=LADDER_HELD_SLUGS_STALE_SECS + 60)
    HELD_SLUGS_PATH.write_text(json.dumps({"updated_ts": stale_ts.isoformat(), "slugs": [ladder_slug]}))

    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = make_grounding("mlb", "hits", "yes", 40, actual_rate=70.0)
    calibration = FakeCalibration({("mlb", "hits", "yes", 40): grounding})
    client = FakeClient([market], fills={SLUG: 99})
    client.positions[ladder_slug] = 1.0
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert memory.is_halted()  # stale publish file -- not trusted, falls back to halting
    assert ladder_slug in memory.anomaly_halted_reason


def test_missing_ladder_publish_file_falls_back_to_the_cautious_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stray_slug = "astatc-mlb-lad-sfo-2026-08-14-tb-playerz-gte1"
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = make_grounding("mlb", "hits", "yes", 40, actual_rate=70.0)
    calibration = FakeCalibration({("mlb", "hits", "yes", 40): grounding})
    client = FakeClient([market], fills={SLUG: 99})
    client.positions[stray_slug] = 2.0  # no ladder publish file written at all
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert memory.is_halted()

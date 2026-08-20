"""Tests for the tennis-props scan/decide/execute loop.

Mirrors test_sports_props_bot.py's style closely -- same fake client shape,
same FakeCalibration pattern -- since tennis_props_bot.py itself reuses
apex.risk and apex.memory.ApexMemory directly rather than re-deriving the
same money-critical logic. Every test that reaches _scan_once() uses
monkeypatch.chdir(tmp_path) since this bot has no cross-bot file reads of
its own, but the pattern is kept for consistency with the other bot test
files and in case that ever changes.
"""

from __future__ import annotations

from src.trading.polymarket_us import tennis_props_bot as bot_module
from src.trading.polymarket_us.apex.memory import ApexMemory
from src.trading.polymarket_us.models import Market, Order, Position
from src.trading.polymarket_us.tennis_prop_calibration import TennisPropGrounding
from src.trading.polymarket_us.tennis_props_bot import TennisPropsBot


def make_market(slug, yes_bid, yes_ask, ask_depth=50, bid_depth=50, closed=False):
    return Market(
        slug=slug,
        id=slug,
        title=slug,
        category="sports",
        tags=[],
        active=not closed,
        closed=closed,
        close_time=None,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=(100 - yes_ask) if yes_ask is not None else None,
        no_ask=(100 - yes_bid) if yes_bid is not None else None,
        volume=0.0,
        volume_24h=0.0,
        bid_depth=bid_depth,
        ask_depth=ask_depth,
    )


def make_grounding(
    category, side, price_cents, trustworthy=True, actual_rate=None, note="calibration OK: 40 samples", count=40
):
    return TennisPropGrounding(
        category=category,
        side=side,
        price_cents=price_cents,
        implied_prob=price_cents / 100.0,
        calibration_trustworthy=trustworthy,
        calibration_note=note,
        calibration_sample_count=count,
        calibration_actual_rate=actual_rate,
    )


class FakeCalibration:
    def __init__(self, groundings, stale=False):
        self._groundings = groundings  # (category, side, price_cents) -> TennisPropGrounding
        self._stale = stale
        self.reload_calls = 0

    def reload(self):
        self.reload_calls += 1

    def data_is_stale(self):
        return self._stale

    def evaluate(self, category, side, price_cents):
        key = (category, side, price_cents)
        if key in self._groundings:
            return self._groundings[key]
        return make_grounding(category, side, price_cents, trustworthy=False, note="no data", count=0)


class FakeClient:
    def __init__(self, markets, fills=None, balance=100_000, settlements=None, requote_markets=None):
        self._markets = {m.slug: m for m in markets}
        self._fills = fills or {}
        self._balance = balance
        self._settlements = settlements or {}
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
        return Order(
            order_id="o-" + market_slug,
            market_slug=market_slug,
            side=side,
            count=count,
            price_cents=yes_price,
            status="unknown",
        )


def make_bot(client, calibration, memory):
    bot_module.FILL_POLL_DELAY = 0
    return TennisPropsBot(client=client, memory=memory, calibration=calibration, poll_interval=1)


SLUG = "tsc-atp-artfer-alemin-2026-08-17-tg-17pt5"


def test_scan_buys_yes_when_calibration_shows_a_trustworthy_edge(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = make_grounding("atp_game_totals", "yes", 40, actual_rate=70.0)
    calibration = FakeCalibration({("atp_game_totals", "yes", 40): grounding})
    client = FakeClient([market], fills={SLUG: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert len(client.orders) == 1
    slug, side, count, yes_price, ioc = client.orders[0]
    assert (slug, side, yes_price, ioc) == (SLUG, "yes", 40, True)
    assert len(memory.open_trades()) == 1


def test_scan_buys_no_at_the_yes_equivalent_price(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    market = make_market(SLUG, yes_bid=70, yes_ask=95)
    grounding = make_grounding("atp_game_totals", "no", 30, actual_rate=70.0)
    calibration = FakeCalibration({("atp_game_totals", "no", 30): grounding})
    client = FakeClient([market], fills={SLUG: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert len(client.orders) == 1
    slug, side, count, yes_price = client.orders[0][:4]
    assert (slug, side, yes_price) == (SLUG, "no", 70)
    trade = memory.open_trades()[0]
    assert trade["side"] == "no"
    assert trade["price_cents"] == 30


def test_scan_ignores_spread_categories_entirely(tmp_path, monkeypatch):
    """Real finding 2026-08-18: spreads don't verify cleanly, so this bot
    must never consider them a candidate at all, even with an
    (artificially) trustworthy grounding available."""
    monkeypatch.chdir(tmp_path)
    spread_slug = "asc-atp-artfer-alemin-2026-08-17-gs-pos-1pt5"
    market = make_market(spread_slug, yes_bid=38, yes_ask=40)
    grounding = make_grounding("atp_game_spreads", "yes", 40, actual_rate=90.0)
    calibration = FakeCalibration({("atp_game_spreads", "yes", 40): grounding})
    client = FakeClient([market], fills={spread_slug: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert client.orders == []


def test_scan_never_stacks_a_second_position_in_the_same_match(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    held_slug = "tsc-atp-artfer-alemin-2026-08-17-st-2pt5"  # same match, different totals line
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = make_grounding("atp_game_totals", "yes", 40, actual_rate=70.0)
    calibration = FakeCalibration({("atp_game_totals", "yes", 40): grounding})
    client = FakeClient([market], fills={SLUG: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)
    memory.record_open(
        market_slug=held_slug,
        side="yes",
        count=1.0,
        price_cents=80,
        cost_cents=80,
        model_prob=0.6,
        market_implied_prob=0.8,
        edge_pct=0.08,
        kelly_fraction_used=0.05,
    )

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert client.orders == []  # held_slug is the SAME match (artfer-alemin, 2026-08-17) as SLUG
    assert len(memory.open_trades()) == 1


def test_different_matches_can_both_be_traded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    other_slug = "tsc-atp-juacer-felaug-2026-08-17-tg-20pt5"
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    other_market = make_market(other_slug, yes_bid=68, yes_ask=70)  # different decile, avoids the bucket guard too
    grounding = make_grounding("atp_game_totals", "yes", 40, actual_rate=70.0)
    other_grounding = make_grounding("atp_game_totals", "yes", 70, actual_rate=90.0)
    calibration = FakeCalibration(
        {
            ("atp_game_totals", "yes", 40): grounding,
            ("atp_game_totals", "yes", 70): other_grounding,
        }
    )
    client = FakeClient([market, other_market], fills={SLUG: 99, other_slug: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert len(client.orders) == 2


# ── Bucket-correlation guard (built in from day one, not retrofitted) ──────


def test_scan_blocks_a_second_match_landing_in_the_same_calibration_bucket(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    other_slug = "tsc-atp-juacer-felaug-2026-08-17-tg-17pt5"  # different match, same price -> same bucket
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    other_market = make_market(other_slug, yes_bid=38, yes_ask=40)
    grounding = make_grounding("atp_game_totals", "yes", 40, actual_rate=70.0)
    calibration = FakeCalibration({("atp_game_totals", "yes", 40): grounding})
    client = FakeClient([market, other_market], fills={SLUG: 99, other_slug: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert len(client.orders) == 1  # the second match's identical-bucket candidate was skipped


def test_an_already_open_position_blocks_a_new_bucket_match(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    held_slug = "tsc-atp-juacer-felaug-2026-08-16-tg-17pt5"  # different match, different day, same bucket
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = make_grounding("atp_game_totals", "yes", 40, actual_rate=70.0)
    calibration = FakeCalibration({("atp_game_totals", "yes", 40): grounding})
    client = FakeClient([market], fills={SLUG: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)
    memory.record_open(
        market_slug=held_slug,
        side="yes",
        count=1.0,
        price_cents=40,
        cost_cents=40,
        model_prob=0.7,
        market_implied_prob=0.4,
        edge_pct=0.3,
        kelly_fraction_used=0.05,
    )

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert client.orders == []
    assert len(memory.open_trades()) == 1


def test_game_totals_and_set_totals_are_independent_buckets(tmp_path, monkeypatch):
    """Different subtype (game vs set totals) on the SAME match is still a
    different calibration category -- the bucket guard must not conflate
    them just because they share a match. (The match-level guard still
    blocks this at the match level -- this test checks the bucket key
    itself is correctly scoped, using two different matches to isolate it.)"""
    other_slug = "tsc-atp-juacer-felaug-2026-08-17-st-2pt5"
    market = make_market(SLUG, yes_bid=38, yes_ask=40)  # atp_game_totals
    other_market = make_market(other_slug, yes_bid=38, yes_ask=40)  # atp_set_totals, same price
    grounding = make_grounding("atp_game_totals", "yes", 40, actual_rate=70.0)
    other_grounding = make_grounding("atp_set_totals", "yes", 40, actual_rate=70.0)
    calibration = FakeCalibration(
        {
            ("atp_game_totals", "yes", 40): grounding,
            ("atp_set_totals", "yes", 40): other_grounding,
        }
    )
    client = FakeClient([market, other_market], fills={SLUG: 99, other_slug: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert len(client.orders) == 2  # different category (game_totals vs set_totals) -- not the same bucket


def test_terminated_bot_only_observes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = make_grounding("atp_game_totals", "yes", 40, actual_rate=70.0)
    calibration = FakeCalibration({("atp_game_totals", "yes", 40): grounding})
    client = FakeClient([market], fills={SLUG: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)
    memory.terminated_reason = "bankroll $0.50 at or below the $1.00 floor"

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert client.orders == []


def test_stale_calibration_blocks_trading(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = make_grounding("atp_game_totals", "yes", 40, actual_rate=70.0)
    calibration = FakeCalibration({("atp_game_totals", "yes", 40): grounding}, stale=True)
    client = FakeClient([market], fills={SLUG: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert client.orders == []
    assert calibration.reload_calls == 1


def test_settlement_reconciliation_pays_out_and_logs_a_miss_lesson(tmp_path):
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)
    memory.record_open(
        market_slug=SLUG,
        side="yes",
        count=1.0,
        price_cents=40,
        cost_cents=40,
        model_prob=0.9,
        market_implied_prob=0.4,
        edge_pct=0.4,
        kelly_fraction_used=0.05,
    )
    market = make_market(SLUG, yes_bid=0, yes_ask=100, closed=True)
    client = FakeClient([market], settlements={SLUG: {"settlement": 0}})

    bot = TennisPropsBot(client=client, memory=memory, calibration=FakeCalibration({}))
    bot._reconcile_settlements()

    assert memory.open_trades() == []
    settled = memory.trade_log[-1]
    assert settled["status"] == "settled"
    assert settled["pnl_cents"] == -40
    assert memory.bankroll_cents == 960
    assert len(memory.lessons) == 1


def test_untracked_live_position_halts_before_trading(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stray_slug = "tsc-atp-juacer-felaug-2026-08-14-tg-20pt5"
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = make_grounding("atp_game_totals", "yes", 40, actual_rate=70.0)
    calibration = FakeCalibration({("atp_game_totals", "yes", 40): grounding})
    client = FakeClient([market], fills={SLUG: 99})
    client.positions[stray_slug] = 2.0
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert client.orders == []
    assert memory.is_halted()
    assert stray_slug in memory.anomaly_halted_reason


def test_reconciliation_ignores_other_bots_live_positions(tmp_path, monkeypatch):
    """Shared account with the ladder bot and Apex -- neither trades
    asc-*/tsc-* slugs (confirmed live, see module docstring), so their
    positions never even parse as a tennis game_key and must never trigger
    a halt here."""
    monkeypatch.chdir(tmp_path)
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = make_grounding("atp_game_totals", "yes", 40, actual_rate=70.0)
    calibration = FakeCalibration({("atp_game_totals", "yes", 40): grounding})
    client = FakeClient([market], fills={SLUG: 99})
    client.positions["astatc-mlb-lad-sfo-2026-08-14-tb-playerz-gte1"] = 5.0  # sports-props/ladder's
    client.positions["tc-temp-nychigh-2026-08-14-gte86lt87f"] = 3.0  # Apex's
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, calibration, memory)
    bot._scan_once()

    assert not memory.is_halted()
    assert len(client.orders) == 1

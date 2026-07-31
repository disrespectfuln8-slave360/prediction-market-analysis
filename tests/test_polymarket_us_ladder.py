"""Regression tests for the polymarket.us ladder-arb execution path.

The cases that matter here all come from a live incident on 2026-07-30: the
API reported that two YES legs had not filled, the bot cancelled them and
recorded nothing, and both had actually filled — leaving real unhedged
positions. Every test below that mentions "silent fill" is modelling that
exact API behaviour: an order response that admits nothing, and a position
that says otherwise.
"""
from __future__ import annotations

from src.trading.polymarket_us.bot import LadderArbBot
from src.trading.polymarket_us.models import Market, Order, Position
from src.trading.polymarket_us.strategy import (
    MAX_SPREAD_CENTS,
    _size_lock,
    find_ladder_locks,
)


def make_market(slug, yes_bid, yes_ask):
    return Market(
        slug=slug, id=slug, title=slug, category="sports", tags=[],
        active=True, closed=False, close_time=None,
        yes_bid=yes_bid, yes_ask=yes_ask,
        no_bid=(100 - yes_ask) if yes_ask is not None else None,
        no_ask=(100 - yes_bid) if yes_bid is not None else None,
        volume=0.0, volume_24h=0.0,
    )


# ── Sizing ───────────────────────────────────────────────────────────────────

def test_size_lock_returns_zero_when_budget_cannot_cover_one_contract():
    # 99c/contract against a 48c budget. Rounding up to 1 here is what buys a
    # YES leg with no money left to hedge it.
    assert _size_lock(buy_price=40, sell_price=41, budget_cents=48) == 0


def test_size_lock_never_exceeds_budget():
    for edge in range(1, 21):
        buy, sell = 40, 40 + edge
        count = _size_lock(buy, sell, budget_cents=448)
        assert count * (buy + (100 - sell)) <= 448


# ── Liquidity filter ─────────────────────────────────────────────────────────

def test_wide_spread_rungs_are_not_traded():
    # gte180 from the incident: 52/99, a 47c book nobody will trade at.
    markets = [
        make_market("x-gte1", yes_bid=52, yes_ask=99),
        make_market("x-gte2", yes_bid=78, yes_ask=88),
    ]
    assert find_ladder_locks(markets, set(), budget_cents=500) == []


def test_one_sided_book_is_not_traded():
    markets = [
        make_market("x-gte1", yes_bid=None, yes_ask=40),
        make_market("x-gte2", yes_bid=50, yes_ask=55),
    ]
    assert find_ladder_locks(markets, set(), budget_cents=500) == []


def test_before_round_ladder_runs_the_other_direction():
    """"Fight ends before Round N" nests the opposite way from "N+ goals".

    before(2) implies before(3), so the LOWER rung is the subset and price
    must rise with N. A violation is the low rung bid ABOVE the high rung
    ask — the mirror image of the gte case.
    """
    markets = [
        make_market("ufc-x-rof-before-r2", yes_bid=50, yes_ask=52),
        make_market("ufc-x-rof-before-r3", yes_bid=38, yes_ask=40),
    ]
    locks = find_ladder_locks(markets, set(), budget_cents=500)
    assert len(locks) == 1
    lock = locks[0]
    assert lock.family == "before_round"
    # YES goes on r3 (the implied, superset event), NO on r2 (the implier).
    assert lock.buy_slug == "ufc-x-rof-before-r3"
    assert lock.sell_slug == "ufc-x-rof-before-r2"
    assert lock.edge_cents == 10


def test_before_round_correct_ordering_is_not_traded():
    """Prices already rising with N is coherent — there is no arb."""
    markets = [
        make_market("ufc-x-rof-before-r2", yes_bid=38, yes_ask=40),
        make_market("ufc-x-rof-before-r3", yes_bid=50, yes_ask=52),
    ]
    assert find_ladder_locks(markets, set(), budget_cents=500) == []


def test_win_in_round_markets_are_never_traded_as_a_ladder():
    """`-rov-f1-rN` are mutually exclusive outcomes, not nested ones.

    Winning in round 1 does not imply winning in round 2, so pricing them
    as a ladder would be a directional bet dressed up as arbitrage. These
    must not match any family no matter how the prices sit.
    """
    markets = [
        make_market("ufc-x-rov-f1-r1", yes_bid=50, yes_ask=52),
        make_market("ufc-x-rov-f1-r2", yes_bid=38, yes_ask=40),
        make_market("ufc-x-rov-f1-r3", yes_bid=20, yes_ask=22),
    ]
    assert find_ladder_locks(markets, set(), budget_cents=500) == []


def test_families_are_never_paired_with_each_other():
    """Two families under one prefix must not be cross-matched."""
    markets = [
        make_market("evt-gte1", yes_bid=38, yes_ask=40),
        make_market("evt-before-r1", yes_bid=90, yes_ask=92),
    ]
    assert find_ladder_locks(markets, set(), budget_cents=500) == []


def test_tight_book_violation_is_still_found():
    markets = [
        make_market("x-gte1", yes_bid=38, yes_ask=40),
        make_market("x-gte2", yes_bid=50, yes_ask=52),
    ]
    locks = find_ladder_locks(markets, set(), budget_cents=500)
    assert len(locks) == 1
    assert locks[0].edge_cents == 10
    assert locks[0].total_cost_cents <= 500
    assert MAX_SPREAD_CENTS >= 2  # sanity: the fixture is inside the filter


# ── Fake client modelling the real API's unreliable fill reporting ───────────

class FakeClient:
    """Reports whatever `report_status` says, while positions tell the truth."""

    def __init__(self, markets, fills, report_status="unknown", balance=500,
                 sell_fails=False):
        self._markets = {m.slug: m for m in markets}
        self._fills = fills  # slug -> contracts that actually fill per order
        self._report_status = report_status
        self._balance = balance
        self._sell_fails = sell_fails
        self.positions: dict[str, float] = {}
        self.orders: list[tuple] = []
        self.sells: list[tuple] = []

    def get_balance(self):
        return self._balance

    def get_positions(self):
        return [Position(market_slug=s, position=p)
                for s, p in self.positions.items() if p != 0]

    def get_open_orders(self):
        return []

    def get_market(self, slug):
        return self._markets[slug]

    def list_all_markets(self, **kw):
        return list(self._markets.values())

    def place_order(self, market_slug, side, count, yes_price,
                    immediate_or_cancel=False):
        self.orders.append((market_slug, side, count, yes_price, immediate_or_cancel))
        filled = min(count, self._fills.get(market_slug, 0))
        signed = filled if side == "yes" else -filled
        self.positions[market_slug] = self.positions.get(market_slug, 0) + signed
        return Order(order_id="o-" + market_slug, market_slug=market_slug, side=side,
                     count=count, price_cents=yes_price, status=self._report_status)

    def close_long(self, market_slug, count, price_cents):
        self.sells.append((market_slug, count, price_cents))
        if self._sell_fails:
            return "sell-fail", 0
        self.positions[market_slug] = self.positions.get(market_slug, 0) - count
        return "sell-ok", count


def make_bot(client):
    bot = LadderArbBot(client=client, poll_interval=1, max_open_legs=20)
    # Position readback is immediate in these fakes; don't sleep through the
    # real propagation allowance.
    import src.trading.polymarket_us.bot as bot_module
    bot_module.FILL_POLL_DELAY = 0
    return bot


LADDER = [
    make_market("p-gte1", yes_bid=38, yes_ask=40),
    make_market("p-gte2", yes_bid=50, yes_ask=52),
]


def test_silent_fill_is_detected_and_hedged():
    """The incident: response admits nothing, both legs actually fill.

    Old code cancelled and recorded nothing. Correct behaviour is to notice
    the position and hedge it.
    """
    client = FakeClient(LADDER, fills={"p-gte1": 5, "p-gte2": 5}, report_status="unknown")
    bot = make_bot(client)
    locks = find_ladder_locks(LADDER, set(), budget_cents=500)
    spent = bot._place_lock(locks[0], budget_cents=500)

    assert client.positions["p-gte1"] > 0      # long the low rung
    assert client.positions["p-gte2"] < 0      # short the high rung
    assert abs(client.positions["p-gte1"]) == abs(client.positions["p-gte2"])
    assert spent > 0
    assert bot._halted_reason is None
    assert client.sells == []                  # nothing to unwind


def test_silent_no_fill_takes_no_position():
    """Nothing fills. The bot must not invent a position or a spend."""
    client = FakeClient(LADDER, fills={}, report_status="filled")  # lies the other way
    bot = make_bot(client)
    locks = find_ladder_locks(LADDER, set(), budget_cents=500)
    spent = bot._place_lock(locks[0], budget_cents=500)

    assert spent == 0
    assert client.get_positions() == []
    assert len(client.orders) == 1             # never sent the hedge
    assert bot._halted_reason is None


def test_partial_first_leg_is_hedged_at_its_real_size():
    client = FakeClient(LADDER, fills={"p-gte1": 2, "p-gte2": 99}, report_status="unknown")
    bot = make_bot(client)
    locks = find_ladder_locks(LADDER, set(), budget_cents=500)
    bot._place_lock(locks[0], budget_cents=500)

    hedge = [o for o in client.orders if o[1] == "no"][0]
    assert hedge[2] == 2                       # hedged the fill, not the intent
    assert client.positions["p-gte1"] == 2
    assert client.positions["p-gte2"] == -2
    assert bot._halted_reason is None


def test_unhedgeable_leg_is_flattened():
    """YES fills, NO cannot — the long must be sold back, not kept."""
    client = FakeClient(LADDER, fills={"p-gte1": 5, "p-gte2": 0}, report_status="unknown")
    bot = make_bot(client)
    locks = find_ladder_locks(LADDER, set(), budget_cents=500)
    bot._place_lock(locks[0], budget_cents=500)

    assert client.sells == [("p-gte1", 5, 38)]  # sold into the bid
    assert client.positions.get("p-gte1", 0) == 0
    assert bot._halted_reason is None


def test_failed_flatten_halts_trading():
    client = FakeClient(LADDER, fills={"p-gte1": 5, "p-gte2": 0},
                        report_status="unknown", sell_fails=True)
    bot = make_bot(client)
    locks = find_ladder_locks(LADDER, set(), budget_cents=500)
    bot._place_lock(locks[0], budget_cents=500)

    assert bot._halted_reason is not None
    assert client.positions["p-gte1"] == 5     # still naked, and known to be

    # A halted bot keeps reporting but stops committing money.
    before = len(client.orders)
    bot._scan_once()
    assert len(client.orders) == before


def test_stale_edge_is_rejected_at_requote():
    """Candidate came from the bulk scan; the live book no longer supports it."""
    stale = [
        make_market("p-gte1", yes_bid=38, yes_ask=40),
        make_market("p-gte2", yes_bid=50, yes_ask=52),
    ]
    locks = find_ladder_locks(stale, set(), budget_cents=500)

    moved = [
        make_market("p-gte1", yes_bid=48, yes_ask=50),   # ask rose above the
        make_market("p-gte2", yes_bid=50, yes_ask=52),   # other rung's bid
    ]
    client = FakeClient(moved, fills={"p-gte1": 5, "p-gte2": 5}, report_status="unknown")
    bot = make_bot(client)
    spent = bot._place_lock(locks[0], budget_cents=500)

    assert spent == 0
    assert client.orders == []                 # never sent a thing

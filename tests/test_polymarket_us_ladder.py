"""Regression tests for the polymarket.us ladder-arb execution path.

The cases that matter here all come from a live incident on 2026-07-30: the
API reported that two YES legs had not filled, the bot cancelled them and
recorded nothing, and both had actually filled — leaving real unhedged
positions. Every test below that mentions "silent fill" is modelling that
exact API behaviour: an order response that admits nothing, and a position
that says otherwise.
"""
from __future__ import annotations

from dataclasses import replace

from src.trading.polymarket_us.bot import MIN_BALANCE_CENTS, SPORTS_PROP_DEPTH_MARGIN, LadderArbBot
from src.trading.polymarket_us.models import Market, Order, Position
from src.trading.polymarket_us.strategy import (
    MAX_SPREAD_CENTS,
    _size_lock,
    find_ladder_locks,
    taker_fee_cents,
)


def make_market(slug, yes_bid, yes_ask, bid_depth=50, ask_depth=50, close_time=None):
    return Market(
        slug=slug, id=slug, title=slug, category="sports", tags=[],
        active=True, closed=False, close_time=close_time,
        yes_bid=yes_bid, yes_ask=yes_ask,
        no_bid=(100 - yes_ask) if yes_ask is not None else None,
        no_ask=(100 - yes_bid) if yes_bid is not None else None,
        volume=0.0, volume_24h=0.0,
        bid_depth=bid_depth, ask_depth=ask_depth,
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


def test_by_date_ladder_uses_earlier_date_as_the_subset():
    """"Released by <date>": earlier deadline implies later one."""
    markets = [
        make_market("prdc-gtavi-11-19-2026", yes_bid=92, yes_ask=93),
        make_market("prdc-gtavi-12-31-2026", yes_bid=85, yes_ask=86),
    ]
    locks = find_ladder_locks(markets, set(), budget_cents=500)
    assert len(locks) == 1
    lock = locks[0]
    assert lock.family == "by_date"
    assert lock.buy_slug == "prdc-gtavi-12-31-2026"   # YES on the later (implied) date
    assert lock.sell_slug == "prdc-gtavi-11-19-2026"  # NO on the earlier (implier)
    assert lock.edge_cents == 6


def test_by_date_handles_both_slug_date_formats():
    """MM-DD-YYYY and YYYY-MM-DD both occur and must sort together."""
    markets = [
        make_market("mowc-nato-us-11-19-2026", yes_bid=92, yes_ask=93),
        make_market("mowc-nato-us-2026-12-31", yes_bid=85, yes_ask=86),
    ]
    locks = find_ladder_locks(markets, set(), budget_cents=500)
    assert len(locks) == 1
    assert locks[0].sell_slug == "mowc-nato-us-11-19-2026"  # earlier date is the subset


def test_fee_estimate_never_understates_observed_commissions():
    """Every (price, count, commission) point is a real fill from this
    account's /v1/portfolio/activities on 2026-08-01. The estimate may
    overstate (a floor that is too strict skips a marginal trade) but must
    never understate (a floor that is too loose books losers as winners).
    """
    observed = [
        (17, 1, 1), (20, 1, 1), (18, 2, 2), (88, 2, 1), (87, 2, 1),
        (11, 1, 1), (4, 2, 0), (3, 2, 0), (17, 2, 2),
    ]
    for price, count, commission in observed:
        assert taker_fee_cents(price, count) >= commission


def test_one_cent_gross_edge_is_rejected_as_fee_negative():
    """The live GTA VI lock: 88c ask vs 89c bid, +1c gross. Entry fees are
    ~2c/contract, so the "guaranteed profit" is a guaranteed loss. The old
    code took this trade with real money.
    """
    markets = [
        make_market("prdc-gtavi-11-19-2026", yes_bid=89, yes_ask=90),
        make_market("prdc-gtavi-12-31-2026", yes_bid=87, yes_ask=88),
    ]
    assert find_ladder_locks(markets, set(), budget_cents=500) == []


def test_long_dated_lock_needs_more_edge_than_same_day():
    """+6c gross (~+4c net) clears the fee floor same-day but not four
    months out, where the lock-up hurdle demands more.
    """
    from datetime import datetime, timedelta, timezone
    far = datetime.now(timezone.utc) + timedelta(days=120)
    near = datetime.now(timezone.utc) + timedelta(hours=6)

    def ladder(close):
        return [
            make_market("prdc-gtavi-11-19-2026", yes_bid=92, yes_ask=93, close_time=close),
            make_market("prdc-gtavi-12-31-2026", yes_bid=85, yes_ask=86, close_time=close),
        ]

    assert len(find_ladder_locks(ladder(near), set(), budget_cents=500)) == 1
    assert find_ladder_locks(ladder(far), set(), budget_cents=500) == []


def test_paper_ledger_charges_fees(tmp_path):
    """A fee-free paper ledger overstated every thin-edge lock — the +$38
    paper record was earned with simulated free trading.
    """
    from src.trading.polymarket_us.paper_client import PaperPolymarketUSTradingClient

    class _StubReal:
        def close(self):
            pass

    client = PaperPolymarketUSTradingClient(
        real_client=_StubReal(), ledger_path=tmp_path / "ledger.json",
        starting_balance_cents=10_000,
    )
    client.place_order(market_slug="m", side="yes", count=5, yes_price=40)
    fee = taker_fee_cents(40, 5)
    assert fee > 0
    assert client.get_balance() == 10_000 - 5 * 40 - fee


def test_paper_ledger_accepts_a_fractional_contract_count(tmp_path):
    """Regression: Kelly-sized bots (Apex, sports-props) size count as
    stake_cents / price_cents and deliberately never int() it, but
    _fee_cents_exact used to do `Decimal * count` with that raw float and
    crash with `unsupported operand type(s) for *: 'decimal.Decimal' and
    'float'` -- caught live 2026-08-15 when the sports-props bot's first
    paper-mode order hit this exact path and silently rejected every
    single trade it tried to place.
    """
    from src.trading.polymarket_us.paper_client import PaperPolymarketUSTradingClient

    class _StubReal:
        def close(self):
            pass

    client = PaperPolymarketUSTradingClient(
        real_client=_StubReal(), ledger_path=tmp_path / "ledger.json",
        starting_balance_cents=10_000,
    )
    order = client.place_order(market_slug="m", side="yes", count=1.25, yes_price=40)
    assert order.status == "filled"
    fee = taker_fee_cents(40, 1.25)
    assert fee > 0
    assert client.get_balance() == 10_000 - round(40 * 1.25) - fee


def test_by_date_correct_ordering_is_not_traded():
    markets = [
        make_market("prdc-gtavi-11-19-2026", yes_bid=80, yes_ask=82),
        make_market("prdc-gtavi-12-31-2026", yes_bid=88, yes_ask=90),
    ]
    assert find_ladder_locks(markets, set(), budget_cents=500) == []


def test_game_winner_markets_are_never_traded_as_a_date_ladder():
    """`aec-*` is 29 of 36 date-suffixed families and nests not at all.

    The trailing date is the fixture date, so two rungs are two unrelated
    baseball games. Shape matches the date pattern exactly; only the prefix
    allowlist keeps them out.
    """
    markets = [
        make_market("aec-mlb-stl-tor-2026-07-31", yes_bid=89, yes_ask=90),
        make_market("aec-mlb-stl-tor-2026-08-01", yes_bid=40, yes_ask=42),
    ]
    assert find_ladder_locks(markets, set(), budget_cents=500) == []


def test_unverified_date_prefix_is_ignored():
    """A new date family must be read and allowlisted before it can trade."""
    markets = [
        make_market("newthing-x-2026-07-31", yes_bid=89, yes_ask=90),
        make_market("newthing-x-2026-12-31", yes_bid=40, yes_ask=42),
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


def test_fractional_fill_is_hedged_not_ignored():
    """The incident: a 5-contract IOC buy on a player-prop market (0.01
    minimum trade size there) partial-filled at 0.3. Old code computed
    yes_filled = int(0.3) = 0, logged "did not fill", and walked away from
    a real unhedged position that later lost its full stake at settlement.
    Fill sizes must survive as fractions all the way through the hedge.
    """
    client = FakeClient(LADDER, fills={"p-gte1": 0.3, "p-gte2": 0.3}, report_status="unknown")
    bot = make_bot(client)
    locks = find_ladder_locks(LADDER, set(), budget_cents=500)
    spent = bot._place_lock(locks[0], budget_cents=500)

    assert len(client.orders) == 2              # the NO leg must actually be attempted
    assert client.positions["p-gte1"] == 0.3
    assert client.positions["p-gte2"] == -0.3
    assert spent > 0
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


def test_failed_pair_is_not_retried_next_scan():
    """The 2026-07-31 loss: same pair re-attempted every scan for 35 minutes.

    Each cycle filled the YES leg, failed the hedge, and paid the spread to
    unwind — about twenty times, for 79% of the account. One attempt is a
    cost of doing business; twenty is a bug.
    """
    client = FakeClient(LADDER, fills={"p-gte1": 5, "p-gte2": 0}, report_status="unknown")
    bot = make_bot(client)
    locks = find_ladder_locks(LADDER, set(), budget_cents=500)

    bot._place_lock(locks[0], budget_cents=500)
    assert client.sells, "first attempt should have unwound"
    assert bot._in_cooldown(locks[0])

    orders_before = len(client.orders)
    bot._scan_once()
    assert len(client.orders) == orders_before, "pair must not be retried while cooling off"


def test_repeated_hedge_failures_halt_the_bot():
    """Three strikes across pairs means conditions are broadly wrong."""
    client = FakeClient(LADDER, fills={"p-gte1": 5, "p-gte2": 0}, report_status="unknown")
    bot = make_bot(client)
    lock = find_ladder_locks(LADDER, set(), budget_cents=500)[0]

    for _ in range(3):
        bot._cooldown.clear()  # simulate three *different* pairs failing
        bot._place_lock(lock, budget_cents=500)

    assert bot._consecutive_failures >= 3
    assert bot._halted_reason is not None


def test_successful_lock_resets_the_failure_counter():
    client = FakeClient(LADDER, fills={"p-gte1": 5, "p-gte2": 5}, report_status="unknown")
    bot = make_bot(client)
    bot._consecutive_failures = 2
    lock = find_ladder_locks(LADDER, set(), budget_cents=500)[0]
    bot._place_lock(lock, budget_cents=500)
    assert bot._consecutive_failures == 0
    assert bot._halted_reason is None


def test_order_is_capped_to_book_depth():
    """4 contracts against a 3-deep book is what caused the unwind loop."""
    shallow = [
        make_market("p-gte1", yes_bid=38, yes_ask=40, ask_depth=3, bid_depth=50),
        make_market("p-gte2", yes_bid=50, yes_ask=52, bid_depth=3, ask_depth=50),
    ]
    client = FakeClient(shallow, fills={"p-gte1": 99, "p-gte2": 99}, report_status="unknown")
    bot = make_bot(client)
    lock = find_ladder_locks(shallow, set(), budget_cents=500)[0]
    assert lock.count > 3, "budget alone would have sized past the book"

    bot._place_lock(lock, budget_cents=500)
    for order in client.orders:
        slug, count = order[0], order[2]
        assert count <= 3, f"{slug} ordered {count} against a 3-deep book"


def test_sports_prop_pairs_require_a_depth_margin():
    """Regression test: every CRITICAL halt to date (2026-08-04 through
    2026-08-13) was on an astatc-* pair where a depth snapshot showing
    exactly enough size let a real fill through, then had nothing left for
    the hedge or flatten a moment later. A book at 4-deep should only size
    an astatc-* lock to floor(4 / SPORTS_PROP_DEPTH_MARGIN) contracts, not
    the full 4 an identical non-sports book would get."""
    sports = [
        make_market("astatc-mlb-nyy-bos-gte1", yes_bid=38, yes_ask=40, ask_depth=4, bid_depth=50),
        make_market("astatc-mlb-nyy-bos-gte2", yes_bid=50, yes_ask=52, bid_depth=4, ask_depth=50),
    ]
    client = FakeClient(sports, fills={"astatc-mlb-nyy-bos-gte1": 99, "astatc-mlb-nyy-bos-gte2": 99},
                         report_status="unknown")
    bot = make_bot(client)
    lock = find_ladder_locks(sports, set(), budget_cents=500)[0]
    assert lock.count > 4 // SPORTS_PROP_DEPTH_MARGIN, "budget alone would have sized past the margin"

    bot._place_lock(lock, budget_cents=500)
    for order in client.orders:
        slug, count = order[0], order[2]
        assert count <= 4 // SPORTS_PROP_DEPTH_MARGIN, f"{slug} ordered {count} past the sports depth margin"


def test_non_sports_pairs_are_unaffected_by_the_sports_depth_margin():
    """Same 4-deep book, non-astatc slugs -- the margin must not apply
    outside the category it was built for (see test_sports_prop_pairs_...)."""
    non_sports = [
        make_market("p-gte1", yes_bid=38, yes_ask=40, ask_depth=4, bid_depth=50),
        make_market("p-gte2", yes_bid=50, yes_ask=52, bid_depth=4, ask_depth=50),
    ]
    client = FakeClient(non_sports, fills={"p-gte1": 99, "p-gte2": 99}, report_status="unknown")
    bot = make_bot(client)
    lock = find_ladder_locks(non_sports, set(), budget_cents=500)[0]

    bot._place_lock(lock, budget_cents=500)
    assert any(order[2] == 4 for order in client.orders), "non-sports book should size up to the full depth"


def test_unknown_depth_falls_back_to_one_contract():
    blind = [
        make_market("p-gte1", yes_bid=38, yes_ask=40, ask_depth=None, bid_depth=None),
        make_market("p-gte2", yes_bid=50, yes_ask=52, ask_depth=None, bid_depth=None),
    ]
    client = FakeClient(blind, fills={"p-gte1": 99, "p-gte2": 99}, report_status="unknown")
    bot = make_bot(client)
    lock = find_ladder_locks(blind, set(), budget_cents=500)[0]
    bot._place_lock(lock, budget_cents=500)
    assert all(o[2] == 1 for o in client.orders)


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


# ── Stale NO-side price at placement (2026-08-05 zero-fill incidents) ────────

class MovingBookFakeClient(FakeClient):
    """A NO-side book whose bid ticks down between the requote and the
    moment the NO leg is actually sent — reproducing the timing race behind
    the 2026-08-05 incidents (get_market() on live astatc-* markets moved
    inside single-digit-second poll gaps, well within
    _requote() -> YES leg -> FILL_POLL -> NO leg).

    get_market(sell_slug) returns `initial_bid` on its first call (the
    _requote() read) and `moved_bid` on every call after. place_order's
    NO-leg fill is decided against the true live bid (`moved_bid` — time has
    passed by send time regardless of whether the bot re-checked), not a
    fixed fill dict, so a stale price genuinely misses and a fresh price
    genuinely crosses.
    """

    def __init__(self, markets, sell_slug, buy_slug, initial_bid, moved_bid,
                 yes_fill_count, balance=500, moved_bid_depth=None):
        super().__init__(markets, fills={buy_slug: yes_fill_count}, balance=balance)
        self.sell_slug = sell_slug
        self._bid = initial_bid
        self._moved_bid = moved_bid
        self._moved_bid_depth = moved_bid_depth
        self._sell_slug_reads = 0

    def get_market(self, slug):
        if slug == self.sell_slug:
            self._sell_slug_reads += 1
            overrides = {"yes_bid": self._bid}
            # Only the refresh read (the second one) sees a depth override —
            # the first (_requote's) read must stay unaffected so sizing
            # isn't disturbed by a test targeting the later re-check.
            if self._sell_slug_reads > 1 and self._moved_bid_depth is not None:
                overrides["bid_depth"] = self._moved_bid_depth
            current = replace(self._markets[slug], **overrides)
            self._bid = self._moved_bid  # subsequent reads see the moved book
            return current
        return self._markets[slug]

    def place_order(self, market_slug, side, count, yes_price, immediate_or_cancel=False):
        self.orders.append((market_slug, side, count, yes_price, immediate_or_cancel))
        if market_slug == self.sell_slug and side == "no":
            filled = count if yes_price <= self._moved_bid else 0  # ground truth at send time
        else:
            filled = min(count, self._fills.get(market_slug, 0))
        signed = filled if side == "yes" else -filled
        self.positions[market_slug] = self.positions.get(market_slug, 0) + signed
        return Order(order_id="o-" + market_slug, market_slug=market_slug, side=side,
                     count=count, price_cents=yes_price, status="unknown")


def test_no_leg_repriced_when_book_moves_after_requote():
    """The 2026-08-05 incidents: bid=50 at requote time, ticked to 47 by
    the time the NO leg is actually sent (after the YES leg + FILL_POLL).
    Old code sends yes_price=50 (stale) -> 50 <= 47 is False -> 0 filled ->
    forced flatten of the YES leg, a realized loss. Fixed code re-reads
    the book immediately before sending -> yes_price=47 -> crosses -> hedged.
    """
    markets = [
        make_market("p-gte1", yes_bid=38, yes_ask=40),
        make_market("p-gte2", yes_bid=50, yes_ask=52),  # requote() sees bid=50
    ]
    client = MovingBookFakeClient(
        markets, sell_slug="p-gte2", buy_slug="p-gte1",
        initial_bid=50, moved_bid=47, yes_fill_count=5,
    )
    bot = make_bot(client)
    lock = find_ladder_locks(markets, set(), budget_cents=500)[0]

    bot._place_lock(lock, budget_cents=500)

    no_order = [o for o in client.orders if o[1] == "no"][0]
    assert no_order[3] == 47, "NO leg must be priced off the fresh bid, not the stale requote"
    assert client.positions["p-gte1"] == 5     # hedged, not flattened
    assert client.positions["p-gte2"] == -5
    assert client.sells == []                  # no forced unwind


def test_no_leg_skipped_when_book_empties_before_placement():
    """Depth is refreshed on the same call as price. A read of bid_depth<=0
    right before placement means there's truly nothing to hit — skip the
    order outright (straight to flatten) rather than sending a guaranteed
    miss."""
    markets = [
        make_market("p-gte1", yes_bid=38, yes_ask=40),
        make_market("p-gte2", yes_bid=50, yes_ask=52),
    ]
    client = MovingBookFakeClient(
        markets, sell_slug="p-gte2", buy_slug="p-gte1",
        initial_bid=50, moved_bid=50, yes_fill_count=5, moved_bid_depth=0,
    )
    bot = make_bot(client)
    lock = find_ladder_locks(markets, set(), budget_cents=500)[0]

    bot._place_lock(lock, budget_cents=500)

    assert [o for o in client.orders if o[1] == "no"] == [], \
        "must not send a NO order into a book with zero depth"
    assert client.positions["p-gte1"] == 0     # flattened, not left naked
    assert client.sells


def test_no_leg_skipped_when_refreshed_price_would_lock_in_a_loss():
    """A price move severe enough that hedging at it would lock in a loss
    (rather than cap one) must not be hedged at all — a fresh price is not
    a fresh approval to trade at any price."""
    markets = [
        make_market("p-gte1", yes_bid=38, yes_ask=40),
        make_market("p-gte2", yes_bid=50, yes_ask=52),  # requote() sees bid=50
    ]
    client = MovingBookFakeClient(
        markets, sell_slug="p-gte2", buy_slug="p-gte1",
        initial_bid=50, moved_bid=35, yes_fill_count=5,  # 40 + (100-35) > 100
    )
    bot = make_bot(client)
    lock = find_ladder_locks(markets, set(), budget_cents=500)[0]

    bot._place_lock(lock, budget_cents=500)

    assert [o for o in client.orders if o[1] == "no"] == [], \
        "must not send a NO order priced to lock in a loss"
    assert client.positions["p-gte1"] == 0     # flattened, not left naked
    assert client.sells


def test_no_leg_falls_back_to_last_quote_if_refresh_fails():
    """A transient error on the extra pre-send read must not block the
    hedge attempt outright — fall back to the last-known price, exactly
    as today, and let the existing flatten-or-halt path handle whatever
    happens next."""
    class FlakyRefreshClient(FakeClient):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._sell_calls = 0

        def get_market(self, slug):
            if slug == "p-gte2":
                self._sell_calls += 1
                if self._sell_calls >= 2:
                    raise RuntimeError("boom")
            return super().get_market(slug)

    client = FlakyRefreshClient(LADDER, fills={"p-gte1": 5, "p-gte2": 5}, report_status="unknown")
    bot = make_bot(client)
    lock = find_ladder_locks(LADDER, set(), budget_cents=500)[0]

    bot._place_lock(lock, budget_cents=500)

    no_order = [o for o in client.orders if o[1] == "no"][0]
    assert no_order[3] == 50   # fell back to the original _requote() price
    assert client.positions["p-gte1"] == 5
    assert bot._halted_reason is None


# ── Minimum-balance floor ─────────────────────────────────────────────────────

def test_scan_halts_and_stops_trading_at_the_balance_floor():
    """A bad stretch must stop on its own before the stake is gone, not after."""
    client = FakeClient(LADDER, fills={"p-gte1": 5, "p-gte2": 5}, balance=MIN_BALANCE_CENTS)
    bot = make_bot(client)

    bot._scan_once()

    assert bot._halted_reason is not None
    assert client.orders == []


def test_scan_keeps_trading_one_cent_above_the_floor():
    client = FakeClient(LADDER, fills={"p-gte1": 5, "p-gte2": 5}, balance=MIN_BALANCE_CENTS + 1)
    bot = make_bot(client)

    bot._scan_once()

    assert bot._halted_reason is None


def test_scan_publishes_held_slugs_for_other_live_bots_to_read(tmp_path, monkeypatch):
    """sports_props_bot.py reads this file to tell "the ladder bot legitimately
    holds this astatc-mlb-* position" apart from a real untracked anomaly of
    its own -- real incident 2026-08-17, see sports_props_bot.py's
    _reconcile_live_positions docstring."""
    monkeypatch.chdir(tmp_path)
    client = FakeClient(LADDER, fills={})
    client.positions["astatc-mlb-az-bos-2026-08-17-er-mitbra-gte1"] = 1.0
    bot = make_bot(client)

    bot._scan_once()

    import json

    from src.trading.polymarket_us.bot import HELD_SLUGS_PATH
    payload = json.loads(HELD_SLUGS_PATH.read_text())
    assert payload["slugs"] == ["astatc-mlb-az-bos-2026-08-17-er-mitbra-gte1"]
    assert payload["updated_ts"]

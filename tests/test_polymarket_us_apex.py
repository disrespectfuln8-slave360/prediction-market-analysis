"""Tests for the Apex weather-trading bot: the deterministic risk/sizing
math, the bankroll/lesson ledger, the grounded probability model's
calibration gate, and the scan/decide/execute loop.

Every trade decision in Apex is made by plain code (risk.py, weather_model.py)
so it can be tested exactly like this, without ever needing a live LLM call --
see apex/bot.py's module docstring for why that split exists.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.trading.polymarket_us.apex import bot as bot_module
from src.trading.polymarket_us.apex import risk
from src.trading.polymarket_us.apex.bot import ApexBot
from src.trading.polymarket_us.apex.memory import (
    DAILY_DRAWDOWN_HALT_FRACTION,
    TERMINATION_FLOOR_CENTS,
    ApexMemory,
)
from src.trading.polymarket_us.apex.weather_model import (
    LIVE_SIGMA_SHRINKAGE,
    MAX_CALIBRATION_ERROR_PCT,
    MIN_SAMPLE_FLOOR,
    STALE_HISTORY_DAYS,
    BucketGrounding,
    WeatherGroundingEngine,
    _bucket_low_pct,
    _build_calibration_table,
)
from src.trading.polymarket_us.models import Market, Order, Position
from src.trading.polymarket_us.nbm_forecast import bucket_probability
from src.trading.polymarket_us.strategy import taker_fee_cents


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


# ── risk.py: deterministic Kelly/edge/reserve math ──────────────────────────


def test_raw_kelly_matches_textbook_even_money_formula():
    # p=0.6 at a 50c coin-flip price: textbook Kelly is 2p-1 = 0.2.
    assert risk.raw_kelly_fraction(0.6, 0.5) == pytest.approx(0.2)


def test_raw_kelly_is_never_negative():
    assert risk.raw_kelly_fraction(0.3, 0.5) == 0.0


def test_edge_below_floor_is_a_pass():
    fee = taker_fee_cents(50, 1)
    price_frac = (50 + fee) / 100
    decision = risk.evaluate(
        model_prob=price_frac + 0.01,  # ~1% edge, below the 4% floor
        price_cents=50,
        bankroll_cents=1000,
        cash_cents=1000,
    )
    assert decision.action == "pass"
    assert "floor" in decision.reason


def test_quarter_kelly_binds_below_the_hard_cap():
    decision = risk.evaluate(model_prob=0.60, price_cents=50, bankroll_cents=1000, cash_cents=1000)
    assert decision.action == "buy"
    assert decision.kelly_fraction_used == pytest.approx(decision.kelly_fraction_raw * risk.MAX_KELLY_FRACTION)
    assert decision.kelly_fraction_used < risk.MAX_POSITION_FRACTION
    assert decision.stake_cents == round(decision.kelly_fraction_used * 1000)


def test_hard_position_cap_binds_on_a_large_edge():
    decision = risk.evaluate(model_prob=0.65, price_cents=50, bankroll_cents=1000, cash_cents=1000)
    assert decision.action == "buy"
    assert decision.kelly_fraction_used == pytest.approx(risk.MAX_POSITION_FRACTION)
    assert decision.stake_cents == round(risk.MAX_POSITION_FRACTION * 1000)


def test_cash_reserve_shrinks_a_trade_when_edge_is_below_the_override():
    # bankroll=$10, only $4.40 in cash -- the 40% reserve floor is $4.00, so
    # only $0.40 is actually spendable, well under what quarter/hard-cap
    # sizing would otherwise take (and still above the minimum viable stake).
    decision = risk.evaluate(model_prob=0.65, price_cents=50, bankroll_cents=1000, cash_cents=440)
    assert decision.action == "buy"
    assert decision.edge_fraction < risk.HIGH_EDGE_OVERRIDE_FRACTION
    assert decision.stake_cents == 40  # 440 - reserve(400)


def test_reserve_squeeze_below_the_minimum_stake_is_a_pass():
    # Same reserve squeeze as above, but tighter: only $0.20 would be left
    # to spend (420 - reserve(400)), which is below MIN_STAKE_CENTS. A
    # technically-positive stake this small isn't worth a real order.
    decision = risk.evaluate(model_prob=0.65, price_cents=50, bankroll_cents=1000, cash_cents=420)
    assert decision.action == "pass"
    assert decision.stake_cents == 0
    assert "minimum viable size" in decision.reason


def test_min_stake_floor_rejects_a_negligible_liquidity_capped_stake():
    # A moderate price with thin book depth can pass every other gate
    # (edge, Kelly, cash reserve) and still size down to a couple of cents
    # once the liquidity cap bites.
    decision = risk.evaluate(
        model_prob=0.9,
        price_cents=50,
        bankroll_cents=100_000,
        cash_cents=100_000,
        max_contracts=0.02,
    )
    assert decision.action == "pass"
    assert decision.stake_cents == 0
    assert "minimum viable size" in decision.reason


def test_cheap_longshot_price_is_rejected_even_with_a_huge_edge():
    # 5c is well outside the 30-70c moderate-probability range -- rejected
    # before edge/Kelly sizing even runs, regardless of how large the
    # apparent edge is.
    decision = risk.evaluate(model_prob=0.9, price_cents=5, bankroll_cents=100_000, cash_cents=100_000)
    assert decision.action == "pass"
    assert "moderate-probability range" in decision.reason


def test_expensive_longshot_price_is_rejected_even_with_a_huge_edge():
    # 95c (betting NO-side-equivalent certainty) is symmetric to the cheap
    # case -- same gate, same reason.
    decision = risk.evaluate(model_prob=0.02, price_cents=95, bankroll_cents=100_000, cash_cents=100_000)
    assert decision.action == "pass"
    assert "moderate-probability range" in decision.reason


def test_moderate_price_at_the_range_boundary_is_still_tradeable():
    decision = risk.evaluate(
        model_prob=0.5, price_cents=risk.MIN_TRADEABLE_PRICE_CENTS, bankroll_cents=100_000, cash_cents=100_000
    )
    assert decision.action == "buy"


def test_high_edge_overrides_the_cash_reserve_floor():
    decision = risk.evaluate(model_prob=0.75, price_cents=50, bankroll_cents=1000, cash_cents=200)
    assert decision.action == "buy"
    assert decision.edge_fraction >= risk.HIGH_EDGE_OVERRIDE_FRACTION
    # Cash itself ($2.00) still covers the capped $0.50 stake -- the reserve
    # gate is skipped, not a license to overspend actual cash.
    assert decision.stake_cents == round(risk.MAX_POSITION_FRACTION * 1000)


def test_liquidity_caps_the_position_size():
    decision = risk.evaluate(
        model_prob=0.65,
        price_cents=50,
        bankroll_cents=1000,
        cash_cents=1000,
        max_contracts=0.5,
    )
    assert decision.action == "buy"
    assert decision.count == pytest.approx(0.5)
    assert decision.stake_cents == round(0.5 * 50)


def test_zero_depth_is_a_pass():
    decision = risk.evaluate(
        model_prob=0.9,
        price_cents=50,
        bankroll_cents=1000,
        cash_cents=1000,
        max_contracts=0,
    )
    assert decision.action == "pass"


def test_evaluate_never_spends_more_than_available_cash():
    decision = risk.evaluate(model_prob=0.95, price_cents=50, bankroll_cents=100_000, cash_cents=30)
    assert decision.stake_cents <= 30


# ── memory.py: bankroll/trade/lesson ledger ──────────────────────────────────


def test_record_open_and_settlement_round_trip(tmp_path):
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)
    trade_id = memory.record_open(
        market_slug="s",
        side="yes",
        count=1.0,
        price_cents=40,
        cost_cents=40,
        model_prob=0.6,
        market_implied_prob=0.4,
        edge_pct=0.1,
        kelly_fraction_used=0.05,
    )
    assert memory.cash_cents == 960
    assert memory.committed_cents == 40
    assert memory.bankroll_cents == 1000

    pnl = memory.record_settlement(trade_id, outcome_yes=True, payout_cents=100)
    assert pnl == 60
    assert memory.committed_cents == 0
    assert memory.cash_cents == 1060
    assert memory.bankroll_cents == 1060
    assert memory.open_trades() == []


def test_termination_when_bankroll_hits_the_floor(tmp_path):
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)
    trade_id = memory.record_open(
        market_slug="s",
        side="yes",
        count=1.0,
        price_cents=95,
        cost_cents=950,
        model_prob=0.5,
        market_implied_prob=0.5,
        edge_pct=0.0,
        kelly_fraction_used=0.05,
    )
    memory.record_settlement(trade_id, outcome_yes=False, payout_cents=0)
    assert memory.bankroll_cents <= TERMINATION_FLOOR_CENTS
    assert memory.is_terminated()
    assert "floor" in memory.terminated_reason


def test_daily_drawdown_halts_and_clears_next_day(tmp_path, monkeypatch):
    import src.trading.polymarket_us.apex.memory as memory_module

    monkeypatch.setattr(memory_module, "_today_utc", lambda: "2026-08-12")
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)
    trade_id = memory.record_open(
        market_slug="s",
        side="yes",
        count=1.0,
        price_cents=20,
        cost_cents=200,
        model_prob=0.6,
        market_implied_prob=0.2,
        edge_pct=0.3,
        kelly_fraction_used=0.05,
    )
    memory.record_settlement(trade_id, outcome_yes=False, payout_cents=0)
    assert memory.daily_drawdown_fraction() > DAILY_DRAWDOWN_HALT_FRACTION
    assert memory.is_halted_today()

    monkeypatch.setattr(memory_module, "_today_utc", lambda: "2026-08-13")
    memory.roll_daily_window()
    assert not memory.is_halted_today()
    assert memory.daily_start_bankroll_cents == memory.bankroll_cents


def test_memory_persists_across_instances(tmp_path):
    path = tmp_path / "m.json"
    first = ApexMemory(path=path, starting_bankroll_cents=1000)
    first.record_open(
        market_slug="s",
        side="yes",
        count=1.0,
        price_cents=40,
        cost_cents=40,
        model_prob=0.6,
        market_implied_prob=0.4,
        edge_pct=0.1,
        kelly_fraction_used=0.05,
    )
    first.add_lesson("test lesson")

    second = ApexMemory(path=path, starting_bankroll_cents=1000)
    assert second.cash_cents == 960
    assert len(second.open_trades()) == 1
    assert second.recent_lessons() == ["test lesson"]


# ── weather_model.py: grounded probability + calibration gate ──────────────


def test_bucket_probability_matches_the_normal_cdf_with_continuity_correction():
    from scipy.stats import norm

    expected = norm.cdf(76.5, loc=75, scale=3) - norm.cdf(73.5, loc=75, scale=3)
    assert bucket_probability(75, 3, 74, 76) == pytest.approx(expected)


def test_bucket_probability_unbounded_both_sides_is_certainty():
    assert bucket_probability(75, 3, None, None) == pytest.approx(1.0)


def test_bucket_probability_none_without_sigma():
    assert bucket_probability(75, None, 74, 76) is None
    assert bucket_probability(75, 0, 74, 76) is None


def test_bucket_probability_treats_nan_bounds_as_unbounded_like_none():
    """Regression test for a live bug: the historical parquet stores
    unbounded bucket ends as float NaN (backfill_weather_history.py writes
    `float("nan")`, not None -- a DataFrame column can't hold a mixed
    None/float column), while the live path (parse_bucket_bounds) uses
    real None. A bare `x is None` check silently treated every NaN-bounded
    historical row as if it had a real, tiny bound of `nan - 0.5`, and
    `max(0.0, min(1.0, nan))` resolved that to exactly 1.0 -- every
    unbounded bucket in the calibration history read back a fabricated
    100% probability. Confirmed live: forecast mu=62, sigma=2, threshold
    ">=65" (1.25 std above the mean, true probability ~10.6%) returned 1.0
    with NaN and the correct ~10.6% with None before this fix."""
    p_nan = bucket_probability(62.0, 2.0, 65.0, float("nan"))
    p_none = bucket_probability(62.0, 2.0, 65.0, None)
    assert p_nan == pytest.approx(p_none)
    assert p_nan == pytest.approx(0.1056, abs=1e-3)  # not 1.0

    p_nan_low = bucket_probability(62.0, 2.0, float("nan"), 56.0)
    p_none_low = bucket_probability(62.0, 2.0, None, 56.0)
    assert p_nan_low == pytest.approx(p_none_low)


def test_bucket_low_pct_clamps_100_percent_to_the_top_decile():
    assert _bucket_low_pct(1.0) == 90
    assert _bucket_low_pct(0.35) == 30


def test_build_calibration_table_groups_and_scores_correctly():
    # Tiny sigma makes bucket_probability effectively 0 or 1 depending on
    # whether mu falls inside the bucket -- lets the test control which
    # decile each row lands in without solving the CDF by hand.
    rows = [
        {
            "city": "nychigh",
            "date": f"2026-08-{i + 1:02d}",
            "forecast_mu": 75.0,
            "forecast_sigma": 0.01,
            "low": 75.0,
            "high": 76.0,
            "settlement": 1 if i < 27 else 0,
        }
        for i in range(30)
    ]
    df = pd.DataFrame(rows)
    table = _build_calibration_table(df)
    assert table[("nychigh", 90)]["count"] == 30
    assert table[("nychigh", 90)]["distinct_days"] == 30
    assert table[("nychigh", 90)]["actual_rate"] == pytest.approx(90.0)
    assert table[("nychigh", 90)]["mean_predicted"] > 99.0


def test_build_calibration_table_scopes_each_city_independently():
    """Regression for the pooled-across-cities miscalibration found live
    2026-08-15: the aggregate 60-70% decile measured 12.5pp off (92 real
    samples), driven mostly by LAX and Miami -- pooling meant every city's
    60-70% bucket was judged by that, not its own history. Same rows, same
    decile, different cities: each city's stats must stay separate."""
    rows = (
        [
            {
                "city": "laxhigh",
                "date": f"2026-08-{i + 1:02d}",
                "forecast_mu": 75.0,
                "forecast_sigma": 0.01,
                "low": 75.0,
                "high": 76.0,
                "settlement": 1 if i < 15 else 0,
            }
            for i in range(30)
        ]  # LAX: 50% actual vs ~99% predicted -- bad
        + [
            {
                "city": "nychigh",
                "date": f"2026-08-{i + 1:02d}",
                "forecast_mu": 75.0,
                "forecast_sigma": 0.01,
                "low": 75.0,
                "high": 76.0,
                "settlement": 1,
            }
            for i in range(30)
        ]  # NYC: 100% actual vs ~99% predicted -- fine
    )
    table = _build_calibration_table(pd.DataFrame(rows))
    assert table[("laxhigh", 90)]["actual_rate"] == pytest.approx(50.0)
    assert table[("nychigh", 90)]["actual_rate"] == pytest.approx(100.0)


def test_trustworthiness_requires_the_sample_floor(tmp_path):
    engine = WeatherGroundingEngine(history_path=tmp_path / "missing.parquet")
    try:
        engine._calibration = {
            ("nychigh", 90): {
                "count": MIN_SAMPLE_FLOOR - 1,
                "actual_rate": 90.0,
                "mean_predicted": 91.0,
                "distinct_days": MIN_SAMPLE_FLOOR - 1,
            }
        }
        trustworthy, _note, count = engine._trustworthiness("nychigh", 0.95)
        assert not trustworthy
        assert count == MIN_SAMPLE_FLOOR - 1
    finally:
        engine.close()


def test_trustworthiness_rejects_a_measured_miscalibration(tmp_path):
    engine = WeatherGroundingEngine(history_path=tmp_path / "missing.parquet")
    try:
        engine._calibration = {
            ("nychigh", 90): {
                "count": MIN_SAMPLE_FLOOR + 10,
                "actual_rate": 60.0,
                "mean_predicted": 95.0,
                "distinct_days": MIN_SAMPLE_FLOOR + 10,
            },
        }
        trustworthy, note, _count = engine._trustworthiness("nychigh", 0.95)
        assert not trustworthy
        assert f"{MAX_CALIBRATION_ERROR_PCT:.0f}pp" in note
    finally:
        engine.close()


def test_trustworthiness_passes_a_well_calibrated_bucket(tmp_path):
    engine = WeatherGroundingEngine(history_path=tmp_path / "missing.parquet")
    try:
        engine._calibration = {
            ("nychigh", 90): {
                "count": MIN_SAMPLE_FLOOR + 10,
                "actual_rate": 92.0,
                "mean_predicted": 94.0,
                "distinct_days": MIN_SAMPLE_FLOOR + 10,
            },
        }
        trustworthy, _note, _count = engine._trustworthiness("nychigh", 0.95)
        assert trustworthy
    finally:
        engine.close()


def test_trustworthiness_is_scoped_independently_per_city(tmp_path):
    """Regression for the pooled-across-cities miscalibration found live
    2026-08-15 (see _build_calibration_table's docstring): a bucket's
    trustworthiness must never bleed from one city into another, the same
    lesson as the existing per-side (YES/NO) independence test below."""
    engine = WeatherGroundingEngine(history_path=tmp_path / "missing.parquet")
    try:
        engine._calibration = {
            ("laxhigh", 60): {
                "count": MIN_SAMPLE_FLOOR + 62,
                "actual_rate": 46.2,
                "mean_predicted": 62.6,
                "distinct_days": MIN_SAMPLE_FLOOR + 62,
            },  # bad
            ("nychigh", 60): {
                "count": MIN_SAMPLE_FLOOR + 5,
                "actual_rate": 64.0,
                "mean_predicted": 62.5,
                "distinct_days": MIN_SAMPLE_FLOOR + 5,
            },  # fine
        }
        lax_trustworthy, _note, _count = engine._trustworthiness("laxhigh", 0.65)
        nyc_trustworthy, _note, _count = engine._trustworthiness("nychigh", 0.65)
        assert not lax_trustworthy
        assert nyc_trustworthy
    finally:
        engine.close()


def test_evaluate_bucket_checks_trustworthiness_independently_per_side(tmp_path):
    """The YES side bets on model_prob; the NO side bets on 1 - model_prob --
    a different decile of the calibration table whenever model_prob isn't
    ~50%. Regression test: evaluate_bucket used to compute trustworthiness
    once (off model_prob's own decile) and hand that same flag to both
    sides, so a bucket calibrated as trustworthy at 5% would silently also
    green-light a NO bet on the complementary, possibly-unverified 95%."""
    engine = WeatherGroundingEngine(history_path=tmp_path / "missing.parquet")
    try:
        # 0-10% decile is well-calibrated; 90-100% (the NO side's real bet)
        # has no samples at all.
        engine._calibration = {
            ("nychigh", 0): {
                "count": MIN_SAMPLE_FLOOR + 10,
                "actual_rate": 5.0,
                "mean_predicted": 5.0,
                "distinct_days": MIN_SAMPLE_FLOOR + 10,
            },
        }
        date_str = engine._local_today("nychigh")
        engine._forecast_cache[("nychigh", date_str)] = {
            "forecast_mu": 60.0,
            "forecast_sigma": 1.0,
            "forecast_cycle": "c",
        }
        grounding = engine.evaluate_bucket("nychigh", 80, 81)
        assert grounding is not None
        assert grounding.model_prob < 0.10
        assert grounding.calibration_trustworthy  # YES side: 0-10% decile is calibrated
        assert not grounding.calibration_trustworthy_no  # NO side: 90-100% decile is unverified
    finally:
        engine.close()


def test_evaluate_bucket_shrinks_sigma_for_live_sizing_only(tmp_path):
    """Regression/direction check for the 2026-08-16 sigma-shrinkage fix:
    real settlements have been landing tighter around forecast_mu than
    forecast_sigma implies (checked via standardized residuals across 328
    historical outcomes), so live trade sizing shrinks sigma to correctly
    LOWER the estimated probability of an off-mean bucket -- the opposite
    of what an earlier, wrong-direction draft of this fix would have done
    (inflating sigma, which would have made these exact losing bets look
    even more attractive). model_prob must match bucket_probability computed
    with sigma*LIVE_SIGMA_SHRINKAGE, be strictly lower than the raw-sigma
    probability for this off-mean bucket, and the grounding's own
    forecast_sigma field must stay the real, un-shrunk NOAA value."""
    engine = WeatherGroundingEngine(history_path=tmp_path / "missing.parquet")
    try:
        date_str = engine._local_today("nychigh")
        engine._forecast_cache[("nychigh", date_str)] = {
            "forecast_mu": 80.0,
            "forecast_sigma": 2.0,
            "forecast_cycle": "c",
        }
        grounding = engine.evaluate_bucket("nychigh", 84.0, 85.0)  # ~2-2.5 sigma above the mean

        assert grounding is not None
        assert grounding.forecast_sigma == 2.0  # raw NOAA value, untouched

        expected = bucket_probability(80.0, 2.0 * LIVE_SIGMA_SHRINKAGE, 84.0, 85.0)
        assert grounding.model_prob == pytest.approx(expected)

        raw_sigma_prob = bucket_probability(80.0, 2.0, 84.0, 85.0)
        assert grounding.model_prob < raw_sigma_prob
    finally:
        engine.close()


def test_history_freshness_gate(tmp_path):
    engine = WeatherGroundingEngine(history_path=tmp_path / "missing.parquet")
    try:
        assert engine.history_is_stale("2026-08-10")  # no history at all

        engine._history_max_date = "2026-08-01"
        assert engine.history_is_stale("2026-08-10")  # 9 days old > STALE_HISTORY_DAYS
        assert not engine.history_is_stale(
            (pd.Timestamp("2026-08-01") + pd.Timedelta(days=STALE_HISTORY_DAYS - 1)).strftime("%Y-%m-%d")
        )
    finally:
        engine.close()


def test_reload_calibration_reads_a_real_parquet_file(tmp_path):
    history_path = tmp_path / "weather_history.parquet"
    rows = [
        {
            "city": "nychigh",
            "forecast_mu": 75.0,
            "forecast_sigma": 0.01,
            "low": 75.0,
            "high": 76.0,
            "settlement": 1,
            "date": "2026-08-01",
        }
        for _ in range(30)
    ]
    pd.DataFrame(rows).to_parquet(history_path)

    engine = WeatherGroundingEngine(history_path=history_path)
    try:
        assert ("nychigh", 90) in engine._calibration
        assert engine._calibration[("nychigh", 90)]["count"] == 30
        assert engine._history_max_date == "2026-08-01"
    finally:
        engine.close()


# ── bot.py: scan/decide/execute loop ────────────────────────────────────────


class FakeGrounding:
    def __init__(self, buckets_by_city, groundings, stale=False):
        self._buckets_by_city = buckets_by_city
        self._groundings = groundings  # (city, low, high) -> BucketGrounding
        self._stale = stale
        self.reload_calls = 0

    def history_is_stale(self, today):
        return self._stale

    def reload_calibration(self):
        self.reload_calls += 1

    def discover_open_buckets(self, markets):
        return self._buckets_by_city

    def evaluate_bucket(self, city, low, high):
        return self._groundings.get((city, low, high))


class FakeApexClient:
    def __init__(self, markets, fills=None, balance=100_000, settlements=None):
        self._markets = {m.slug: m for m in markets}
        self._fills = fills or {}
        self._balance = balance
        self._settlements = settlements or {}
        self.positions: dict[str, float] = {}
        self.orders: list[tuple] = []

    def get_balance(self):
        return self._balance

    def get_positions(self):
        return [Position(market_slug=s, position=p) for s, p in self.positions.items() if p != 0]

    def get_market(self, slug):
        return self._markets[slug]

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


def make_bot(client, grounding, memory):
    bot_module.FILL_POLL_DELAY = 0  # no real sleeping in tests
    return ApexBot(client=client, memory=memory, grounding=grounding, poll_interval=1)


SLUG = "tc-temp-nychigh-2026-08-10-gte80lt81f"


def test_scan_buys_yes_when_the_model_beats_the_ask(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = BucketGrounding(
        city="nychigh",
        date="2026-08-10",
        low=80,
        high=81,
        model_prob=0.70,
        forecast_mu=80.5,
        forecast_sigma=1.0,
        forecast_cycle="c",
        calibration_trustworthy=True,
        calibration_note="calibration OK: 30 samples, 2.0pp measured error",
        calibration_sample_count=30,
        calibration_trustworthy_no=True,
        calibration_note_no="calibration OK: 30 samples, 2.0pp measured error",
        calibration_sample_count_no=30,
    )
    fake_grounding = FakeGrounding({"nychigh": [(SLUG, 80, 81, market)]}, {("nychigh", 80, 81): grounding})
    client = FakeApexClient([market], fills={SLUG: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, fake_grounding, memory)
    bot._scan_once()

    assert len(client.orders) == 1
    slug, side, count, yes_price, ioc = client.orders[0]
    assert (slug, side, yes_price, ioc) == (SLUG, "yes", 40, True)
    assert len(memory.open_trades()) == 1
    assert memory.open_trades()[0]["side"] == "yes"
    assert memory.cash_cents == 1000 - memory.open_trades()[0]["cost_cents"]


def test_scan_buys_no_at_the_yes_equivalent_price(tmp_path, monkeypatch):
    """A NO order's yes_price parameter must be the YES-equivalent price
    (yes_bid), not the NO cost (100 - yes_bid) -- passing the wrong one here
    would silently mis-price every NO-side trade on a live account."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    market = make_market(SLUG, yes_bid=70, yes_ask=95)
    grounding = BucketGrounding(
        city="nychigh",
        date="2026-08-10",
        low=80,
        high=81,
        model_prob=0.10,
        forecast_mu=70.0,
        forecast_sigma=1.0,
        forecast_cycle="c",
        calibration_trustworthy=True,
        calibration_note="calibration OK: 30 samples, 2.0pp measured error",
        calibration_sample_count=30,
        calibration_trustworthy_no=True,
        calibration_note_no="calibration OK: 30 samples, 2.0pp measured error",
        calibration_sample_count_no=30,
    )
    fake_grounding = FakeGrounding({"nychigh": [(SLUG, 80, 81, market)]}, {("nychigh", 80, 81): grounding})
    client = FakeApexClient([market], fills={SLUG: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, fake_grounding, memory)
    bot._scan_once()

    assert len(client.orders) == 1
    slug, side, count, yes_price, ioc = client.orders[0]
    assert (slug, side, yes_price) == (SLUG, "no", 70)
    trade = memory.open_trades()[0]
    assert trade["side"] == "no"
    assert trade["price_cents"] == 30  # the actual NO cost, for bankroll/edge accounting


def test_scan_trades_the_no_side_when_only_its_decile_is_calibrated(tmp_path, monkeypatch):
    """Regression test for the per-side calibration gate: an untrustworthy
    YES-side decile must not block a NO-side trade whose own decile (1 -
    model_prob) is independently trustworthy, and vice versa -- the two
    sides bet on different probabilities and must be gated separately."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    market = make_market(SLUG, yes_bid=40, yes_ask=42)  # NO costs 100-40=60c
    grounding = BucketGrounding(
        city="nychigh",
        date="2026-08-10",
        low=80,
        high=81,
        model_prob=0.05,
        forecast_mu=60.0,
        forecast_sigma=1.0,
        forecast_cycle="c",
        calibration_trustworthy=False,
        calibration_note="only 2 historical samples",
        calibration_sample_count=2,
        calibration_trustworthy_no=True,
        calibration_note_no="calibration OK: 40 samples, 1.0pp measured error",
        calibration_sample_count_no=40,
    )
    fake_grounding = FakeGrounding({"nychigh": [(SLUG, 80, 81, market)]}, {("nychigh", 80, 81): grounding})
    client = FakeApexClient([market], fills={SLUG: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, fake_grounding, memory)
    bot._scan_once()

    assert len(client.orders) == 1
    slug, side, count, yes_price, ioc = client.orders[0]
    assert side == "no"
    assert memory.open_trades()[0]["side"] == "no"


def test_scan_passes_when_calibration_is_untrustworthy(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = BucketGrounding(
        city="nychigh",
        date="2026-08-10",
        low=80,
        high=81,
        model_prob=0.70,
        forecast_mu=80.5,
        forecast_sigma=1.0,
        forecast_cycle="c",
        calibration_trustworthy=False,
        calibration_note="only 5 historical samples",
        calibration_sample_count=5,
        calibration_trustworthy_no=False,
        calibration_note_no="only 5 historical samples",
        calibration_sample_count_no=5,
    )
    fake_grounding = FakeGrounding({"nychigh": [(SLUG, 80, 81, market)]}, {("nychigh", 80, 81): grounding})
    client = FakeApexClient([market], fills={SLUG: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, fake_grounding, memory)
    bot._scan_once()

    assert client.orders == []
    assert memory.open_trades() == []


def test_scan_never_stacks_a_second_position_in_the_same_city(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    held_slug = "tc-temp-nychigh-2026-08-09-gte70lt71f"
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = BucketGrounding(
        city="nychigh",
        date="2026-08-10",
        low=80,
        high=81,
        model_prob=0.70,
        forecast_mu=80.5,
        forecast_sigma=1.0,
        forecast_cycle="c",
        calibration_trustworthy=True,
        calibration_note="calibration OK",
        calibration_sample_count=30,
        calibration_trustworthy_no=True,
        calibration_note_no="calibration OK",
        calibration_sample_count_no=30,
    )
    fake_grounding = FakeGrounding({"nychigh": [(SLUG, 80, 81, market)]}, {("nychigh", 80, 81): grounding})
    client = FakeApexClient([market, make_market(held_slug, yes_bid=50, yes_ask=52)], fills={SLUG: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)
    memory.record_open(
        market_slug=held_slug,
        side="yes",
        count=1.0,
        price_cents=52,
        cost_cents=52,
        model_prob=0.6,
        market_implied_prob=0.52,
        edge_pct=0.08,
        kelly_fraction_used=0.05,
    )

    bot = make_bot(client, fake_grounding, memory)
    bot._scan_once()

    assert client.orders == []
    assert len(memory.open_trades()) == 1  # still just the pre-existing one


def test_terminated_bot_only_observes(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = BucketGrounding(
        city="nychigh",
        date="2026-08-10",
        low=80,
        high=81,
        model_prob=0.70,
        forecast_mu=80.5,
        forecast_sigma=1.0,
        forecast_cycle="c",
        calibration_trustworthy=True,
        calibration_note="calibration OK",
        calibration_sample_count=30,
        calibration_trustworthy_no=True,
        calibration_note_no="calibration OK",
        calibration_sample_count_no=30,
    )
    fake_grounding = FakeGrounding({"nychigh": [(SLUG, 80, 81, market)]}, {("nychigh", 80, 81): grounding})
    client = FakeApexClient([market], fills={SLUG: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)
    memory.terminated_reason = "bankroll $0.50 at or below the $1.00 floor"

    bot = make_bot(client, fake_grounding, memory)
    bot._scan_once()

    assert client.orders == []


def test_stale_calibration_history_blocks_trading(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = BucketGrounding(
        city="nychigh",
        date="2026-08-10",
        low=80,
        high=81,
        model_prob=0.70,
        forecast_mu=80.5,
        forecast_sigma=1.0,
        forecast_cycle="c",
        calibration_trustworthy=True,
        calibration_note="calibration OK",
        calibration_sample_count=30,
        calibration_trustworthy_no=True,
        calibration_note_no="calibration OK",
        calibration_sample_count_no=30,
    )
    fake_grounding = FakeGrounding(
        {"nychigh": [(SLUG, 80, 81, market)]},
        {("nychigh", 80, 81): grounding},
        stale=True,
    )
    client = FakeApexClient([market], fills={SLUG: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, fake_grounding, memory)
    bot._scan_once()

    assert client.orders == []
    # reload_calibration() must run before the staleness check, not after --
    # see test_history_file_appearing_after_startup_is_discovered_on_next_scan
    # for why checking staleness first would prevent a history file from
    # ever being discovered once it appears.
    assert fake_grounding.reload_calls == 1


def test_record_manual_close_reconciles_without_faking_a_settlement(tmp_path):
    """Regression test for the first real incident of this kind (2026-08-13):
    a human sold an open Apex position directly through the exchange, outside
    the bot's own code, which has no sell/flatten logic at all. The ledger
    must reflect the real proceeds and stop counting the trade as open --
    but must not record a fabricated outcome_yes, since the market itself
    never actually resolved."""
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)
    trade_id = memory.record_open(
        market_slug=SLUG,
        side="yes",
        count=0.3,
        price_cents=8,
        cost_cents=2,
        model_prob=0.19,
        market_implied_prob=0.08,
        edge_pct=0.11,
        kelly_fraction_used=0.03,
    )

    pnl = memory.record_manual_close(trade_id, proceeds_cents=2)

    assert pnl == 0
    assert memory.open_trades() == []
    closed = memory.trade_log[-1]
    assert closed["status"] == "closed_manually"
    assert closed["proceeds_cents"] == 2
    assert "outcome_yes" not in closed
    assert memory.cash_cents == 1000  # 2 spent, 2 recovered -- back to even
    assert memory.committed_cents == 0


def test_record_manual_close_can_still_trip_the_termination_floor(tmp_path):
    """The same circuit breaker record_settlement() checks must fire here
    too -- a manual close changes the bankroll exactly like a real
    settlement does, and a human selling at a real loss is not a safer
    event than the market resolving against the bot."""
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)
    memory.cash_cents = 105  # $1.05, just above the $1.00 floor
    trade_id = memory.record_open(
        market_slug=SLUG,
        side="yes",
        count=5.0,
        price_cents=10,
        cost_cents=50,
        model_prob=0.3,
        market_implied_prob=0.1,
        edge_pct=0.2,
        kelly_fraction_used=0.05,
    )

    memory.record_manual_close(trade_id, proceeds_cents=0)  # sold for nothing

    assert memory.is_terminated()
    assert "manual close" in memory.terminated_reason


def test_record_manual_open_reconciles_a_stray_position_without_fabricating_a_model_prob(tmp_path):
    """Regression test for the mirror-image incident (2026-08-16): a human
    bought a position directly through the exchange -- Apex's own
    one-bucket-per-city-day guard made it impossible for the bot itself to
    have placed a second Chicago bucket while already holding a third one
    for that same day, and the exchange's own manualOrderIndicator field
    confirmed it. _reconcile_live_positions() halts on any live position
    with no matching ledger entry; this is how a human resolves that
    without either leaving it untracked forever or fabricating a model_prob
    Apex never computed."""
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    trade_id = memory.record_manual_open(
        market_slug=SLUG,
        side="yes",
        count=2.25,
        price_cents=43,
        cost_cents=97,
    )

    assert memory.cash_cents == 903  # 1000 - 97
    assert memory.committed_cents == 97
    trade = memory.open_trades()[0]
    assert trade["id"] == trade_id
    assert trade["model_prob"] == 0.5  # neutral placeholder, not a fabricated Apex estimate
    assert trade["manual"] is True
    assert "not an Apex decision" in trade["rationale"]

    # Must settle through the exact same path as any other trade -- the
    # miss-lesson logic reads trade["model_prob"] unconditionally, so this
    # also proves 0.5 doesn't crash it.
    pnl = memory.record_settlement(trade_id, outcome_yes=True, payout_cents=225)
    assert pnl == 225 - 97
    assert memory.open_trades() == []


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
    client = FakeApexClient([market], settlements={SLUG: {"settlement": 0}})  # resolved NO -- a big miss

    bot = ApexBot(client=client, memory=memory, grounding=FakeGrounding({}, {}))
    bot._reconcile_settlements()

    assert memory.open_trades() == []
    settled = memory.trade_log[-1]
    assert settled["status"] == "settled"
    assert settled["pnl_cents"] == -40
    assert memory.bankroll_cents == 960
    assert len(memory.lessons) == 1
    assert SLUG in memory.lessons[0]["text"]


# ── Regression tests for the adversarial-review findings ───────────────────


def test_recorded_cost_includes_the_taker_fee(tmp_path, monkeypatch):
    """risk.py sizes off a fee-inclusive edge, but the dollar amount it
    hands back is priced at the raw contract price -- the fee has to be
    added back in when the fill is actually booked, or the tracked
    bankroll silently drifts upward relative to the real account on every
    single trade."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = BucketGrounding(
        city="nychigh",
        date="2026-08-10",
        low=80,
        high=81,
        model_prob=0.70,
        forecast_mu=80.5,
        forecast_sigma=1.0,
        forecast_cycle="c",
        calibration_trustworthy=True,
        calibration_note="calibration OK",
        calibration_sample_count=30,
        calibration_trustworthy_no=True,
        calibration_note_no="calibration OK",
        calibration_sample_count_no=30,
    )
    fake_grounding = FakeGrounding({"nychigh": [(SLUG, 80, 81, market)]}, {("nychigh", 80, 81): grounding})
    client = FakeApexClient([market], fills={SLUG: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, fake_grounding, memory)
    bot._scan_once()

    trade = memory.open_trades()[0]
    raw_cost = round(trade["count"] * trade["price_cents"])
    assert trade["cost_cents"] > raw_cost
    fee_per_contract = taker_fee_cents(trade["price_cents"], 1)
    assert trade["cost_cents"] == raw_cost + round(trade["count"] * fee_per_contract)


def test_tiny_fractional_fill_is_never_recorded_as_costing_zero(tmp_path, monkeypatch):
    """A thin book can fill only a sliver of the requested size (IOC kills
    the rest). At a cheap enough price, that sliver's cost rounds to
    literally 0c even though a real position was acquired -- a real fill
    must always cost at least 1c in the ledger."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = BucketGrounding(
        city="nychigh",
        date="2026-08-10",
        low=80,
        high=81,
        model_prob=0.70,
        forecast_mu=80.5,
        forecast_sigma=1.0,
        forecast_cycle="c",
        calibration_trustworthy=True,
        calibration_note="calibration OK",
        calibration_sample_count=30,
        calibration_trustworthy_no=True,
        calibration_note_no="calibration OK",
        calibration_sample_count_no=30,
    )
    fake_grounding = FakeGrounding({"nychigh": [(SLUG, 80, 81, market)]}, {("nychigh", 80, 81): grounding})
    # Only 0.01 contracts actually fill, far below the ~1.25 the sizing
    # math requested -- a thin book taking just a sliver of an IOC order.
    client = FakeApexClient([market], fills={SLUG: 0.01})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, fake_grounding, memory)
    bot._scan_once()

    assert len(memory.open_trades()) == 1
    trade = memory.open_trades()[0]
    assert trade["count"] == pytest.approx(0.01)
    assert trade["cost_cents"] >= 1  # never zero for a real, non-negligible fill


class DelayedFillOnErrorClient(FakeApexClient):
    """place_order raises immediately (as if the request itself timed out),
    but the fill actually lands on the exchange a couple of position reads
    later -- reproducing the exact incident this repo's docstrings warn
    about (an order error doesn't mean nothing happened server-side)."""

    def __init__(self, *a, delay_reads=2, **kw):
        super().__init__(*a, **kw)
        self._delay_reads = delay_reads
        self._reads = 0

    def place_order(self, market_slug, side, count, yes_price, immediate_or_cancel=False):
        self.orders.append((market_slug, side, count, yes_price, immediate_or_cancel))
        raise RuntimeError("simulated timeout")

    def get_positions(self):
        self._reads += 1
        if self._reads > self._delay_reads:
            for slug, count in list(self._fills.items()):
                self.positions[slug] = self.positions.get(slug, 0) + count
            self._fills = {}
        return [Position(market_slug=s, position=p) for s, p in self.positions.items() if p != 0]


def test_exception_path_polls_before_giving_up_on_a_delayed_fill(tmp_path, monkeypatch):
    """A single immediate position read after an order error would miss a
    fill that lands a moment later; the exception path must poll exactly
    like the success path does."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = BucketGrounding(
        city="nychigh",
        date="2026-08-10",
        low=80,
        high=81,
        model_prob=0.70,
        forecast_mu=80.5,
        forecast_sigma=1.0,
        forecast_cycle="c",
        calibration_trustworthy=True,
        calibration_note="calibration OK",
        calibration_sample_count=30,
        calibration_trustworthy_no=True,
        calibration_note_no="calibration OK",
        calibration_sample_count_no=30,
    )
    fake_grounding = FakeGrounding({"nychigh": [(SLUG, 80, 81, market)]}, {("nychigh", 80, 81): grounding})
    client = DelayedFillOnErrorClient([market], fills={SLUG: 2.0}, delay_reads=2)
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, fake_grounding, memory)
    bot._scan_once()

    assert len(memory.open_trades()) == 1
    assert memory.open_trades()[0]["side"] == "yes"


class WrongDirectionFillClient(FakeApexClient):
    """place_order succeeds, but the position moves opposite to the traded
    side -- simulating something else touching the slug in the same window."""

    def place_order(self, market_slug, side, count, yes_price, immediate_or_cancel=False):
        self.orders.append((market_slug, side, count, yes_price, immediate_or_cancel))
        filled = min(count, self._fills.get(market_slug, count))
        signed = -filled if side == "yes" else filled  # backwards on purpose
        self.positions[market_slug] = self.positions.get(market_slug, 0) + signed
        return Order(
            order_id="o-" + market_slug,
            market_slug=market_slug,
            side=side,
            count=count,
            price_cents=yes_price,
            status="unknown",
        )


def test_wrong_direction_fill_halts_instead_of_recording_a_fabricated_trade(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = BucketGrounding(
        city="nychigh",
        date="2026-08-10",
        low=80,
        high=81,
        model_prob=0.70,
        forecast_mu=80.5,
        forecast_sigma=1.0,
        forecast_cycle="c",
        calibration_trustworthy=True,
        calibration_note="calibration OK",
        calibration_sample_count=30,
        calibration_trustworthy_no=True,
        calibration_note_no="calibration OK",
        calibration_sample_count_no=30,
    )
    fake_grounding = FakeGrounding({"nychigh": [(SLUG, 80, 81, market)]}, {("nychigh", 80, 81): grounding})
    client = WrongDirectionFillClient([market], fills={SLUG: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, fake_grounding, memory)
    bot._scan_once()

    assert memory.open_trades() == []
    assert memory.is_halted()
    assert SLUG in memory.anomaly_halted_reason


def test_untracked_live_position_halts_before_trading(tmp_path, monkeypatch):
    """If the live account holds a weather position memory's ledger never
    recorded, the one-position-per-city guard and the fee/drawdown
    circuit-breakers are all blind to it -- the bot must stop and wait for
    a human rather than trade around a ledger it can no longer trust."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    stray_slug = "tc-temp-nychigh-2026-08-09-gte70lt71f"
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = BucketGrounding(
        city="nychigh",
        date="2026-08-10",
        low=80,
        high=81,
        model_prob=0.70,
        forecast_mu=80.5,
        forecast_sigma=1.0,
        forecast_cycle="c",
        calibration_trustworthy=True,
        calibration_note="calibration OK",
        calibration_sample_count=30,
        calibration_trustworthy_no=True,
        calibration_note_no="calibration OK",
        calibration_sample_count_no=30,
    )
    fake_grounding = FakeGrounding({"nychigh": [(SLUG, 80, 81, market)]}, {("nychigh", 80, 81): grounding})
    client = FakeApexClient([market], fills={SLUG: 99})
    client.positions[stray_slug] = 2.0  # a live position memory has never heard of

    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, fake_grounding, memory)
    bot._scan_once()

    assert client.orders == []
    assert memory.is_halted()
    assert stray_slug in memory.anomaly_halted_reason


def test_reconciliation_ignores_other_bots_live_positions(tmp_path, monkeypatch):
    """This account is shared with the unrelated ladder-arb bot -- its
    positions (non tc-temp-* slugs) must never trigger an Apex halt."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = BucketGrounding(
        city="nychigh",
        date="2026-08-10",
        low=80,
        high=81,
        model_prob=0.70,
        forecast_mu=80.5,
        forecast_sigma=1.0,
        forecast_cycle="c",
        calibration_trustworthy=True,
        calibration_note="calibration OK",
        calibration_sample_count=30,
        calibration_trustworthy_no=True,
        calibration_note_no="calibration OK",
        calibration_sample_count_no=30,
    )
    fake_grounding = FakeGrounding({"nychigh": [(SLUG, 80, 81, market)]}, {("nychigh", 80, 81): grounding})
    client = FakeApexClient([market], fills={SLUG: 99})
    client.positions["evt-gte1"] = 5.0  # ladder bot's own position, not Apex's business
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, fake_grounding, memory)
    bot._scan_once()

    assert not memory.is_halted()
    assert len(client.orders) == 1


def test_check_external_balance_terminates_when_the_tracked_ledger_also_agrees(tmp_path):
    """The tracked bankroll can drift from reality (unbooked fees, an
    untracked position); this is the independent backstop against the real
    account balance -- a permanent stop when Apex's OWN ledger also agrees
    the money is genuinely gone, not merely because the real balance
    happens to be low (see the shared-account test below for that case)."""
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)
    memory.cash_cents = 50  # Apex's own ledger also shows it's nearly broke
    assert not memory.is_terminated()

    memory.check_external_balance(50)  # real balance at $0.50, below the $1 floor

    assert memory.is_terminated()
    assert "real exchange balance" in memory.terminated_reason


def test_check_external_balance_halts_for_the_day_on_a_shared_account_dip(tmp_path):
    """This exchange account is shared with other live bots (the ladder
    arb bot -- see bot.py's _reconcile_live_positions). If the real balance
    dips to the floor from a SIBLING bot's own spending while Apex's own
    tracked cash is still healthy, permanently terminating Apex over a
    loss it never took would retire a healthy bot by mistake and require a
    human to notice and hand-edit the ledger. It halts for the day
    (auto-recoverable, re-checked fresh every scan) instead."""
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)
    assert memory.cash_cents == 1000  # healthy, untouched

    memory.check_external_balance(50)  # real balance at $0.50, below the $1 floor

    assert not memory.is_terminated()
    assert memory.is_halted_today()
    assert "shared-account dip" in memory.daily_halted_reason


def test_history_file_appearing_after_startup_is_discovered_on_next_scan(tmp_path, monkeypatch):
    """Regression test for a live incident: reload_calibration() must run
    BEFORE history_is_stale() is checked. Checking staleness first means a
    history file that doesn't exist yet when the engine is constructed --
    the normal case, since Apex starts before the one-off backfill script
    has ever completed -- can never be discovered once it does appear: the
    early return on "stale" permanently prevents reload_calibration() from
    running again. Apex sat "stale or missing" for 6+ hours after a real
    backfill finished before this fix."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    history_path = tmp_path / "weather_history.parquet"
    grounding = WeatherGroundingEngine(history_path=history_path)
    try:
        assert grounding._history_max_date is None  # file doesn't exist yet

        # The backfill "finishes" after the engine was already constructed.
        rows = [
            {
                "city": "nychigh",
                "forecast_mu": 75.0,
                "forecast_sigma": 0.01,
                "low": 75.0,
                "high": 76.0,
                "settlement": 1,
                "date": "2026-08-12",
            }
            for _ in range(30)
        ]
        pd.DataFrame(rows).to_parquet(history_path)

        client = FakeApexClient([], fills={})
        memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)
        bot = make_bot(client, grounding, memory)

        bot._scan_once()

        assert grounding._history_max_date == "2026-08-12"
    finally:
        grounding.close()


def test_halted_bot_stops_trading_and_only_reports(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = BucketGrounding(
        city="nychigh",
        date="2026-08-10",
        low=80,
        high=81,
        model_prob=0.70,
        forecast_mu=80.5,
        forecast_sigma=1.0,
        forecast_cycle="c",
        calibration_trustworthy=True,
        calibration_note="calibration OK",
        calibration_sample_count=30,
        calibration_trustworthy_no=True,
        calibration_note_no="calibration OK",
        calibration_sample_count_no=30,
    )
    fake_grounding = FakeGrounding({"nychigh": [(SLUG, 80, 81, market)]}, {("nychigh", 80, 81): grounding})
    client = FakeApexClient([market], fills={SLUG: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)
    memory.halt("pre-existing anomaly from a prior scan")

    bot = make_bot(client, fake_grounding, memory)
    bot._scan_once()

    assert client.orders == []


# ── clear_halt ───────────────────────────────────────────────────────────────
# The programmatic form of "clear anomaly_halted_reason in the ledger file by
# hand" -- added 2026-08-17 after a real incident needed exactly this
# (sports_props_bot's stray-position halt on the ladder bot's own legs,
# cleared by hand after confirming via the exchange's own trade records).


def test_clear_halt_resets_the_anomaly_reason(tmp_path):
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)
    memory.halt("some anomaly")
    assert memory.is_halted()

    memory.clear_halt("confirmed explainable, see exchange trade records")

    assert not memory.is_halted()
    assert memory.anomaly_halted_reason is None


def test_clear_halt_persists_across_a_reload(tmp_path):
    path = tmp_path / "m.json"
    first = ApexMemory(path=path, starting_bankroll_cents=1000)
    first.halt("some anomaly")
    first.clear_halt("confirmed explainable")

    second = ApexMemory(path=path, starting_bankroll_cents=1000)
    assert not second.is_halted()


def test_clear_halt_on_an_already_clear_ledger_is_a_no_op(tmp_path):
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)
    memory.clear_halt("nothing to clear")  # must not raise
    assert not memory.is_halted()

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
        slug=slug, id=slug, title=slug, category="sports", tags=[],
        active=not closed, closed=closed, close_time=None,
        yes_bid=yes_bid, yes_ask=yes_ask,
        no_bid=(100 - yes_ask) if yes_ask is not None else None,
        no_ask=(100 - yes_bid) if yes_bid is not None else None,
        volume=0.0, volume_24h=0.0, bid_depth=bid_depth, ask_depth=ask_depth,
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
        price_cents=50, bankroll_cents=1000, cash_cents=1000,
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
    # bankroll=$10, only $4.20 in cash -- the 40% reserve floor is $4.00, so
    # only $0.20 is actually spendable, well under what quarter/hard-cap
    # sizing would otherwise take.
    decision = risk.evaluate(model_prob=0.65, price_cents=50, bankroll_cents=1000, cash_cents=420)
    assert decision.action == "buy"
    assert decision.edge_fraction < risk.HIGH_EDGE_OVERRIDE_FRACTION
    assert decision.stake_cents == 20  # 420 - reserve(400)


def test_high_edge_overrides_the_cash_reserve_floor():
    decision = risk.evaluate(model_prob=0.75, price_cents=50, bankroll_cents=1000, cash_cents=200)
    assert decision.action == "buy"
    assert decision.edge_fraction >= risk.HIGH_EDGE_OVERRIDE_FRACTION
    # Cash itself ($2.00) still covers the capped $0.50 stake -- the reserve
    # gate is skipped, not a license to overspend actual cash.
    assert decision.stake_cents == round(risk.MAX_POSITION_FRACTION * 1000)


def test_liquidity_caps_the_position_size():
    decision = risk.evaluate(
        model_prob=0.65, price_cents=50, bankroll_cents=1000, cash_cents=1000, max_contracts=0.5,
    )
    assert decision.action == "buy"
    assert decision.count == pytest.approx(0.5)
    assert decision.stake_cents == round(0.5 * 50)


def test_zero_depth_is_a_pass():
    decision = risk.evaluate(
        model_prob=0.9, price_cents=50, bankroll_cents=1000, cash_cents=1000, max_contracts=0,
    )
    assert decision.action == "pass"


def test_evaluate_never_spends_more_than_available_cash():
    decision = risk.evaluate(model_prob=0.95, price_cents=50, bankroll_cents=100_000, cash_cents=30)
    assert decision.stake_cents <= 30


# ── memory.py: bankroll/trade/lesson ledger ──────────────────────────────────

def test_record_open_and_settlement_round_trip(tmp_path):
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)
    trade_id = memory.record_open(
        market_slug="s", side="yes", count=1.0, price_cents=40, cost_cents=40,
        model_prob=0.6, market_implied_prob=0.4, edge_pct=0.1, kelly_fraction_used=0.05,
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
        market_slug="s", side="yes", count=1.0, price_cents=95, cost_cents=950,
        model_prob=0.5, market_implied_prob=0.5, edge_pct=0.0, kelly_fraction_used=0.05,
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
        market_slug="s", side="yes", count=1.0, price_cents=20, cost_cents=200,
        model_prob=0.6, market_implied_prob=0.2, edge_pct=0.3, kelly_fraction_used=0.05,
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
        market_slug="s", side="yes", count=1.0, price_cents=40, cost_cents=40,
        model_prob=0.6, market_implied_prob=0.4, edge_pct=0.1, kelly_fraction_used=0.05,
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


def test_bucket_low_pct_clamps_100_percent_to_the_top_decile():
    assert _bucket_low_pct(1.0) == 90
    assert _bucket_low_pct(0.35) == 30


def test_build_calibration_table_groups_and_scores_correctly():
    # Tiny sigma makes bucket_probability effectively 0 or 1 depending on
    # whether mu falls inside the bucket -- lets the test control which
    # decile each row lands in without solving the CDF by hand.
    rows = [{"forecast_mu": 75.0, "forecast_sigma": 0.01, "low": 75.0, "high": 76.0,
              "settlement": 1 if i < 27 else 0} for i in range(30)]
    df = pd.DataFrame(rows)
    table = _build_calibration_table(df)
    assert table[90]["count"] == 30
    assert table[90]["actual_rate"] == pytest.approx(90.0)
    assert table[90]["mean_predicted"] > 99.0


def test_trustworthiness_requires_the_sample_floor(tmp_path):
    engine = WeatherGroundingEngine(history_path=tmp_path / "missing.parquet")
    try:
        engine._calibration = {90: {"count": MIN_SAMPLE_FLOOR - 1, "actual_rate": 90.0, "mean_predicted": 91.0}}
        trustworthy, _note, count = engine._trustworthiness(0.95)
        assert not trustworthy
        assert count == MIN_SAMPLE_FLOOR - 1
    finally:
        engine.close()


def test_trustworthiness_rejects_a_measured_miscalibration(tmp_path):
    engine = WeatherGroundingEngine(history_path=tmp_path / "missing.parquet")
    try:
        engine._calibration = {
            90: {"count": MIN_SAMPLE_FLOOR + 10, "actual_rate": 60.0, "mean_predicted": 95.0},
        }
        trustworthy, note, _count = engine._trustworthiness(0.95)
        assert not trustworthy
        assert f"{MAX_CALIBRATION_ERROR_PCT:.0f}pp" in note
    finally:
        engine.close()


def test_trustworthiness_passes_a_well_calibrated_bucket(tmp_path):
    engine = WeatherGroundingEngine(history_path=tmp_path / "missing.parquet")
    try:
        engine._calibration = {
            90: {"count": MIN_SAMPLE_FLOOR + 10, "actual_rate": 92.0, "mean_predicted": 94.0},
        }
        trustworthy, _note, _count = engine._trustworthiness(0.95)
        assert trustworthy
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
    rows = [{"forecast_mu": 75.0, "forecast_sigma": 0.01, "low": 75.0, "high": 76.0,
              "settlement": 1, "date": "2026-08-01"} for _ in range(30)]
    pd.DataFrame(rows).to_parquet(history_path)

    engine = WeatherGroundingEngine(history_path=history_path)
    try:
        assert 90 in engine._calibration
        assert engine._calibration[90]["count"] == 30
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
        return Order(order_id="o-" + market_slug, market_slug=market_slug, side=side,
                     count=count, price_cents=yes_price, status="unknown")


def make_bot(client, grounding, memory):
    bot_module.FILL_POLL_DELAY = 0  # no real sleeping in tests
    return ApexBot(client=client, memory=memory, grounding=grounding, poll_interval=1)


SLUG = "tc-temp-nychigh-2026-08-10-gte80lt81f"


def test_scan_buys_yes_when_the_model_beats_the_ask(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = BucketGrounding(
        city="nychigh", date="2026-08-10", low=80, high=81, model_prob=0.70,
        forecast_mu=80.5, forecast_sigma=1.0, forecast_cycle="c",
        calibration_trustworthy=True, calibration_note="calibration OK: 30 samples, 2.0pp measured error",
        calibration_sample_count=30,
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
        city="nychigh", date="2026-08-10", low=80, high=81, model_prob=0.10,
        forecast_mu=70.0, forecast_sigma=1.0, forecast_cycle="c",
        calibration_trustworthy=True, calibration_note="calibration OK: 30 samples, 2.0pp measured error",
        calibration_sample_count=30,
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


def test_scan_passes_when_calibration_is_untrustworthy(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = BucketGrounding(
        city="nychigh", date="2026-08-10", low=80, high=81, model_prob=0.70,
        forecast_mu=80.5, forecast_sigma=1.0, forecast_cycle="c",
        calibration_trustworthy=False, calibration_note="only 5 historical samples",
        calibration_sample_count=5,
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
        city="nychigh", date="2026-08-10", low=80, high=81, model_prob=0.70,
        forecast_mu=80.5, forecast_sigma=1.0, forecast_cycle="c",
        calibration_trustworthy=True, calibration_note="calibration OK", calibration_sample_count=30,
    )
    fake_grounding = FakeGrounding({"nychigh": [(SLUG, 80, 81, market)]}, {("nychigh", 80, 81): grounding})
    client = FakeApexClient([market, make_market(held_slug, yes_bid=50, yes_ask=52)], fills={SLUG: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)
    memory.record_open(
        market_slug=held_slug, side="yes", count=1.0, price_cents=52, cost_cents=52,
        model_prob=0.6, market_implied_prob=0.52, edge_pct=0.08, kelly_fraction_used=0.05,
    )

    bot = make_bot(client, fake_grounding, memory)
    bot._scan_once()

    assert client.orders == []
    assert len(memory.open_trades()) == 1  # still just the pre-existing one


def test_terminated_bot_only_observes(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = BucketGrounding(
        city="nychigh", date="2026-08-10", low=80, high=81, model_prob=0.70,
        forecast_mu=80.5, forecast_sigma=1.0, forecast_cycle="c",
        calibration_trustworthy=True, calibration_note="calibration OK", calibration_sample_count=30,
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
        city="nychigh", date="2026-08-10", low=80, high=81, model_prob=0.70,
        forecast_mu=80.5, forecast_sigma=1.0, forecast_cycle="c",
        calibration_trustworthy=True, calibration_note="calibration OK", calibration_sample_count=30,
    )
    fake_grounding = FakeGrounding(
        {"nychigh": [(SLUG, 80, 81, market)]}, {("nychigh", 80, 81): grounding}, stale=True,
    )
    client = FakeApexClient([market], fills={SLUG: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, fake_grounding, memory)
    bot._scan_once()

    assert client.orders == []
    assert fake_grounding.reload_calls == 0  # never got past the staleness gate


def test_settlement_reconciliation_pays_out_and_logs_a_miss_lesson(tmp_path):
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)
    memory.record_open(
        market_slug=SLUG, side="yes", count=1.0, price_cents=40, cost_cents=40,
        model_prob=0.9, market_implied_prob=0.4, edge_pct=0.4, kelly_fraction_used=0.05,
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
        city="nychigh", date="2026-08-10", low=80, high=81, model_prob=0.70,
        forecast_mu=80.5, forecast_sigma=1.0, forecast_cycle="c",
        calibration_trustworthy=True, calibration_note="calibration OK", calibration_sample_count=30,
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
        city="nychigh", date="2026-08-10", low=80, high=81, model_prob=0.70,
        forecast_mu=80.5, forecast_sigma=1.0, forecast_cycle="c",
        calibration_trustworthy=True, calibration_note="calibration OK", calibration_sample_count=30,
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
        return Order(order_id="o-" + market_slug, market_slug=market_slug, side=side,
                     count=count, price_cents=yes_price, status="unknown")


def test_wrong_direction_fill_halts_instead_of_recording_a_fabricated_trade(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = BucketGrounding(
        city="nychigh", date="2026-08-10", low=80, high=81, model_prob=0.70,
        forecast_mu=80.5, forecast_sigma=1.0, forecast_cycle="c",
        calibration_trustworthy=True, calibration_note="calibration OK", calibration_sample_count=30,
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
        city="nychigh", date="2026-08-10", low=80, high=81, model_prob=0.70,
        forecast_mu=80.5, forecast_sigma=1.0, forecast_cycle="c",
        calibration_trustworthy=True, calibration_note="calibration OK", calibration_sample_count=30,
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
        city="nychigh", date="2026-08-10", low=80, high=81, model_prob=0.70,
        forecast_mu=80.5, forecast_sigma=1.0, forecast_cycle="c",
        calibration_trustworthy=True, calibration_note="calibration OK", calibration_sample_count=30,
    )
    fake_grounding = FakeGrounding({"nychigh": [(SLUG, 80, 81, market)]}, {("nychigh", 80, 81): grounding})
    client = FakeApexClient([market], fills={SLUG: 99})
    client.positions["evt-gte1"] = 5.0  # ladder bot's own position, not Apex's business
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)

    bot = make_bot(client, fake_grounding, memory)
    bot._scan_once()

    assert not memory.is_halted()
    assert len(client.orders) == 1


def test_check_external_balance_terminates_independent_of_the_tracked_ledger(tmp_path):
    """The tracked bankroll can drift from reality (unbooked fees, an
    untracked position); this is the independent backstop against the real
    account balance."""
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)
    assert not memory.is_terminated()

    memory.check_external_balance(50)  # real balance at $0.50, below the $1 floor

    assert memory.is_terminated()
    assert "real exchange balance" in memory.terminated_reason


def test_halted_bot_stops_trading_and_only_reports(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    market = make_market(SLUG, yes_bid=38, yes_ask=40)
    grounding = BucketGrounding(
        city="nychigh", date="2026-08-10", low=80, high=81, model_prob=0.70,
        forecast_mu=80.5, forecast_sigma=1.0, forecast_cycle="c",
        calibration_trustworthy=True, calibration_note="calibration OK", calibration_sample_count=30,
    )
    fake_grounding = FakeGrounding({"nychigh": [(SLUG, 80, 81, market)]}, {("nychigh", 80, 81): grounding})
    client = FakeApexClient([market], fills={SLUG: 99})
    memory = ApexMemory(path=tmp_path / "m.json", starting_bankroll_cents=1000)
    memory.halt("pre-existing anomaly from a prior scan")

    bot = make_bot(client, fake_grounding, memory)
    bot._scan_once()

    assert client.orders == []

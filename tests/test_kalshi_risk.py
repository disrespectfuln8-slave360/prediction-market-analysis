"""Tests for the Kalshi bots' fee model and deterministic risk gates.

These two modules are the executable form of
`docs/kalshi-bot-operating-manual.md` §5, so the tests are written against
the manual's rules rather than against the implementation: if a rule in the
manual changes, one of these should fail.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from src.trading.kalshi import risk
from src.trading.kalshi.fees import (
    THETA_TAKER,
    breakeven_win_rate,
    effective_cost_cents,
    maker_fee_cents,
    taker_fee_cents,
)
from src.trading.kalshi.risk import RiskLedger

# ── Fees ──────────────────────────────────────────────────────────────────


def test_reproduces_kalshi_documented_fee_example():
    """Kalshi's published worked example: 100 contracts at $0.50 costs $1.75."""
    assert taker_fee_cents(50, 100) == 175


def test_fee_uses_kalshis_theta_not_polymarket_us_theta():
    """Regression guard for the specific mistake this module exists to avoid.

    polymarket.us charges 0.06 with banker's rounding; Kalshi charges 0.07
    rounded up. Importing the sibling platform's model here would understate
    every fee, so the coefficient is asserted directly.
    """
    assert THETA_TAKER == Decimal("0.07")
    # 0.06 would give 150 for this trade; 0.07 gives 175.
    assert taker_fee_cents(50, 100) != 150


def test_fee_always_rounds_up():
    """Kalshi rounds fees up, and rounding up is the safe direction to be
    wrong in — it can never make a losing trade look profitable."""
    # 0.07 * 1 * 50 * 50 / 100 = 1.75 -> 2, not 1.
    assert taker_fee_cents(50, 1) == 2
    # 0.07 * 1 * 30 * 70 / 100 = 1.47 -> 2, not 1.
    assert taker_fee_cents(30, 1) == 2


def test_fee_is_zero_outside_tradeable_prices_and_for_no_size():
    assert taker_fee_cents(0, 100) == 0
    assert taker_fee_cents(100, 100) == 0
    assert taker_fee_cents(50, 0) == 0


def test_maker_fee_is_zero_on_the_general_schedule():
    assert maker_fee_cents(50, 100) == 0


def test_fee_is_largest_at_the_midpoint():
    """p*(1-p) peaks at 50c, so a coin-flip price is the most expensive to
    trade — which is exactly the band the bot is restricted to."""
    mid = taker_fee_cents(50, 100)
    assert taker_fee_cents(30, 100) < mid
    assert taker_fee_cents(70, 100) < mid


def test_breakeven_exceeds_the_quoted_price():
    """The hurdle is price + fee, never price alone. Trading against the
    quoted price is what makes a nominal edge evaporate."""
    assert breakeven_win_rate(50) > 50


def test_breakeven_amortizes_down_with_size_but_default_is_conservative():
    """Kalshi rounds per order, so a 1-contract order carries the worst
    per-contract fee. The default must be that conservative case."""
    assert breakeven_win_rate(50, 100) < breakeven_win_rate(50, 1)
    assert breakeven_win_rate(50) == breakeven_win_rate(50, 1)


def test_effective_cost_includes_the_fee():
    assert effective_cost_cents(50, 100) == 50 * 100 + 175


# ── Per-position gates ────────────────────────────────────────────────────


def test_rejects_longshots_below_the_band():
    """Manual §3: standing operator decision, no longshot bets. A huge
    nominal edge on a cheap contract must still be refused."""
    d = risk.evaluate(win_rate_pct=90.0, price_cents=10, bankroll_cents=100_000, cash_cents=100_000)
    assert d.action == "pass"
    assert "band" in d.reason


def test_rejects_prices_above_the_band():
    d = risk.evaluate(win_rate_pct=99.0, price_cents=85, bankroll_cents=100_000, cash_cents=100_000)
    assert d.action == "pass"
    assert "band" in d.reason


def test_accepts_a_genuine_edge_inside_the_band():
    d = risk.evaluate(win_rate_pct=60.0, price_cents=50, bankroll_cents=100_000, cash_cents=100_000)
    assert d.action == "buy"
    assert d.count > 0
    assert d.stake_cents > 0


def test_edge_floor_is_measured_after_fees():
    """A win rate that clears the raw price but not the fee-inclusive
    breakeven must be rejected. 52% at 50c is +2pp raw but only ~0pp after
    the fee — well under the 4pp floor."""
    d = risk.evaluate(win_rate_pct=52.0, price_cents=50, bankroll_cents=100_000, cash_cents=100_000)
    assert d.action == "pass"
    assert "edge" in d.reason


def test_edge_is_reported_net_of_fees():
    """60% at 50c is +10pp against the raw price but less once the fee is
    charged — the reported edge must be the honest one."""
    d = risk.evaluate(win_rate_pct=60.0, price_cents=50, bankroll_cents=100_000, cash_cents=100_000)
    assert d.edge_pp < 10.0


def test_never_stakes_more_than_the_position_cap():
    """Manual §5: no single market may exceed 5% of bankroll, whatever
    Kelly or the confidence multiplier suggest."""
    bankroll = 100_000
    d = risk.evaluate(
        win_rate_pct=69.0,
        price_cents=30,
        bankroll_cents=bankroll,
        cash_cents=bankroll,
        size_multiplier=1.2,
    )
    assert d.action == "buy"
    cap = risk.MAX_POSITION_FRACTION * bankroll
    # Allow the fee on top of the capped notional, but not the notional itself
    # exceeding the cap.
    assert d.count * 30 <= cap


def test_confidence_multiplier_cannot_breach_the_position_cap():
    """The learning tracker's multiplier is a nudge, never a licence to
    exceed a hard cap."""
    bankroll = 100_000
    huge = risk.evaluate(
        win_rate_pct=69.0,
        price_cents=30,
        bankroll_cents=bankroll,
        cash_cents=bankroll,
        size_multiplier=100.0,
    )
    capped = risk.evaluate(
        win_rate_pct=69.0,
        price_cents=30,
        bankroll_cents=bankroll,
        cash_cents=bankroll,
        size_multiplier=1.0,
    )
    assert huge.count <= max(capped.count, int(risk.MAX_POSITION_FRACTION * bankroll / 30))


def test_cash_reserve_is_respected():
    """Manual §5: at least 40% of bankroll stays in cash. With cash already
    at the reserve line there is nothing left to deploy."""
    bankroll = 100_000
    d = risk.evaluate(
        win_rate_pct=65.0,
        price_cents=50,
        bankroll_cents=bankroll,
        cash_cents=int(bankroll * risk.MIN_CASH_RESERVE_FRACTION),
    )
    assert d.action == "pass"
    assert "reserve" in d.reason


def test_rejects_a_stake_below_the_minimum_viable_size():
    """A technically-positive but negligible position isn't worth the fee
    and ledger footprint."""
    d = risk.evaluate(win_rate_pct=65.0, price_cents=50, bankroll_cents=200, cash_cents=200)
    assert d.action == "pass"


def test_liquidity_cap_limits_count():
    d = risk.evaluate(
        win_rate_pct=65.0,
        price_cents=50,
        bankroll_cents=100_000,
        cash_cents=100_000,
        max_contracts=3,
    )
    assert d.count <= 3


def test_never_spends_more_cash_than_available():
    d = risk.evaluate(win_rate_pct=70.0, price_cents=50, bankroll_cents=100_000, cash_cents=600)
    assert d.stake_cents <= 600


def test_zero_bankroll_is_a_pass_not_a_crash():
    d = risk.evaluate(win_rate_pct=70.0, price_cents=50, bankroll_cents=0, cash_cents=0)
    assert d.action == "pass"


# ── Portfolio circuit breakers ────────────────────────────────────────────


def test_ledger_starts_unhalted():
    ledger = RiskLedger(starting_bankroll_cents=100_000)
    assert not ledger.is_halted()
    assert ledger.halt_reason() is None


def test_daily_loss_limit_halts_trading():
    ledger = RiskLedger(starting_bankroll_cents=100_000)
    ledger.update_bankroll(89_000)  # -11%, past the 10% limit
    assert ledger.is_halted()
    assert "today" in ledger.halt_reason()


def test_daily_loss_limit_is_not_a_hard_halt():
    """A bad day pauses until tomorrow; it does not require a manual re-arm."""
    ledger = RiskLedger(starting_bankroll_cents=100_000)
    ledger.update_bankroll(89_000)
    assert ledger.is_halted()
    assert not ledger.is_hard_halted()


def test_drawdown_kill_switch_is_a_hard_halt():
    """Manual §5: 25% from peak is a hard stop requiring manual re-arm."""
    ledger = RiskLedger(starting_bankroll_cents=100_000)
    ledger.update_bankroll(70_000)  # -30% from peak
    assert ledger.is_hard_halted()
    assert "HARD STOP" in ledger.halt_reason()


def test_drawdown_measured_from_peak_not_from_start():
    """An account that ran up then fell back is drawn down, even if it is
    still above where it started."""
    ledger = RiskLedger(starting_bankroll_cents=100_000)
    ledger.update_bankroll(200_000)  # new peak
    ledger.update_bankroll(140_000)  # -30% from peak, still +40% from start
    assert ledger.is_hard_halted()


def test_hard_halt_does_not_lift_on_a_new_day():
    """The whole point of the kill switch: sleeping on a broken thesis does
    not fix it."""
    ledger = RiskLedger(starting_bankroll_cents=100_000)
    ledger.update_bankroll(70_000)
    assert ledger.is_hard_halted()

    # Simulate the day rolling over.
    ledger._day = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    ledger.update_bankroll(70_000)
    assert ledger.is_hard_halted()
    assert ledger.is_halted()


def test_hard_halt_does_not_lift_on_recovery():
    ledger = RiskLedger(starting_bankroll_cents=100_000)
    ledger.update_bankroll(70_000)
    ledger.update_bankroll(100_000)
    assert ledger.is_hard_halted()


def test_re_arm_clears_a_hard_halt_and_rebases_the_peak():
    ledger = RiskLedger(starting_bankroll_cents=100_000)
    ledger.update_bankroll(70_000)
    assert ledger.is_hard_halted()

    ledger.re_arm()
    assert not ledger.is_hard_halted()
    assert not ledger.is_halted()
    # Peak rebased to current, so the old peak can't immediately re-trip it.
    assert ledger.peak_bankroll_cents == 70_000
    ledger.update_bankroll(70_000)
    assert not ledger.is_halted()


def test_daily_limit_resets_on_a_new_day():
    ledger = RiskLedger(starting_bankroll_cents=100_000)
    ledger.update_bankroll(89_000)
    assert ledger.is_halted()

    ledger._day = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    ledger.update_bankroll(89_000)
    assert not ledger.is_halted()

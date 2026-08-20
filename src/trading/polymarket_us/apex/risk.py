"""Deterministic sizing and risk gates — the part of "Apex" that is never
left to an LLM.

Everything here is plain arithmetic so it can be unit-tested and audited.
The system prompt this bot implements asks for an LLM that "thinks in
expected-value and probability terms," but an LLM computing its own Kelly
fraction and bet size is exactly the failure mode this repo has already
seen once (see strategy.py's note on the removed crypto/macro mispricing
bots): a model can narrate correct-sounding math and still be wrong, with
nothing to catch it before an order goes out. The probability estimate
comes from weather_model.py (grounded in NOAA forecast data, gated by
measured historical calibration); this module takes that estimate as a
given input and turns it into a position size using the exact numeric
rules the system prompt specifies, enforced in code:

  - Edge must be >= MIN_EDGE_FRACTION (4%) after fees to act at all.
  - Kelly fraction capped at MAX_KELLY_FRACTION (quarter-Kelly) for safety.
  - Never more than MAX_POSITION_FRACTION (5%) of bankroll on one market.
  - Keep at least MIN_CASH_RESERVE_FRACTION (40%) of bankroll in cash,
    unless edge >= HIGH_EDGE_OVERRIDE_FRACTION (15%).
  - Never place a stake below MIN_STAKE_CENTS: a technically-positive but
    negligible size (thin depth, a near-worthless longshot price, a cash
    reserve squeeze) isn't worth a real order's fee and ledger footprint.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.trading.polymarket_us.strategy import taker_fee_cents

MIN_EDGE_FRACTION = 0.04
MAX_KELLY_FRACTION = 0.25
MAX_POSITION_FRACTION = 0.05
MIN_CASH_RESERVE_FRACTION = 0.40
HIGH_EDGE_OVERRIDE_FRACTION = 0.15
MIN_STAKE_CENTS = 25

# Added 2026-08-19 after real evidence that every longshot bucket Apex and
# sports-props had actually traded on (price under ~20c) was also among the
# thinnest-sampled: laxhigh 30-40% had only 8 historical rows, nychigh
# 20-30% had 19, miahigh 30-40% had 29 -- right at or below the sample
# floor. A probability near the edges of [0,1] has little room before a
# couple of extra observed outcomes swing the measured rate by double
# digits in relative terms, so "edge" on a cheap longshot is the easiest
# kind of edge to manufacture from noise. Restricting to prices where the
# market itself is genuinely uncertain (30-70c) trades on the calibration's
# most stable ground and gives up the asymmetric "rare big win" payoff
# shape entirely, by design -- explicit user preference 2026-08-19: "i
# dont want long shot strat, thats just setting myself up to fail."
MIN_TRADEABLE_PRICE_CENTS = 30
MAX_TRADEABLE_PRICE_CENTS = 70


@dataclass(frozen=True)
class ApexDecision:
    action: str  # "buy" or "pass"
    edge_fraction: float  # after-fee edge, e.g. 0.06 == 6%
    kelly_fraction_raw: float
    kelly_fraction_used: float
    stake_cents: int
    count: float
    reason: str


def effective_cost_fraction(price_cents: int) -> float:
    """Cost per $1 of payout, as a fraction, including the taker fee — the
    real breakeven price, not just the quoted one."""
    fee = taker_fee_cents(price_cents, 1)
    return (price_cents + fee) / 100.0


def edge_fraction(model_prob: float, price_cents: int) -> float:
    """Estimated probability minus the fee-inclusive market-implied probability."""
    return model_prob - effective_cost_fraction(price_cents)


def raw_kelly_fraction(model_prob: float, price_frac: float) -> float:
    """Full (un-capped) Kelly fraction for a binary bet costing `price_frac`
    per $1 of payout, at true win probability `model_prob`.

    b = net odds = (1 - price_frac) / price_frac
    f* = p - (1 - p) / b

    e.g. p=0.6 at a 50c (price_frac=0.5) coin-flip price gives b=1.0 and
    f*=0.2, matching the textbook even-money Kelly formula 2p-1.
    """
    if not (0.0 < price_frac < 1.0):
        return 0.0
    b = (1.0 - price_frac) / price_frac
    f = model_prob - (1.0 - model_prob) / b
    return max(0.0, f)


def evaluate(
    model_prob: float,
    price_cents: int,
    bankroll_cents: int,
    cash_cents: int,
    max_contracts: float | None = None,
) -> ApexDecision:
    """Full pipeline: edge check -> Kelly sizing -> caps -> cash-reserve gate
    -> liquidity cap. Returns a PASS decision with a reason at the first
    gate that fails; a BUY decision only survives every gate."""
    if not (1 <= price_cents <= 99):
        return ApexDecision("pass", 0.0, 0.0, 0.0, 0, 0.0, f"price {price_cents}c out of tradeable range")

    if not (MIN_TRADEABLE_PRICE_CENTS <= price_cents <= MAX_TRADEABLE_PRICE_CENTS):
        return ApexDecision(
            "pass",
            0.0,
            0.0,
            0.0,
            0,
            0.0,
            f"price {price_cents}c outside the {MIN_TRADEABLE_PRICE_CENTS}-{MAX_TRADEABLE_PRICE_CENTS}c "
            "moderate-probability range -- no longshot bets",
        )

    edge = edge_fraction(model_prob, price_cents)
    if edge < MIN_EDGE_FRACTION:
        return ApexDecision(
            "pass",
            edge,
            0.0,
            0.0,
            0,
            0.0,
            f"edge {edge:+.1%} below the {MIN_EDGE_FRACTION:.0%} floor after fees",
        )

    price_frac = effective_cost_fraction(price_cents)
    kelly_raw = raw_kelly_fraction(model_prob, price_frac)
    kelly_used = min(kelly_raw * MAX_KELLY_FRACTION, MAX_POSITION_FRACTION)
    if kelly_used <= 0 or bankroll_cents <= 0:
        return ApexDecision(
            "pass",
            edge,
            kelly_raw,
            0.0,
            0,
            0.0,
            "no positive sizeable Kelly fraction despite a passing edge",
        )

    stake_cents = round(kelly_used * bankroll_cents)

    reserve_needed_cents = round(MIN_CASH_RESERVE_FRACTION * bankroll_cents)
    cash_after = cash_cents - stake_cents
    if cash_after < reserve_needed_cents and edge < HIGH_EDGE_OVERRIDE_FRACTION:
        stake_cents = max(0, cash_cents - reserve_needed_cents)

    # Never spend more than is actually liquid, override or not.
    stake_cents = max(0, min(stake_cents, cash_cents))

    if stake_cents <= 0:
        return ApexDecision(
            "pass",
            edge,
            kelly_raw,
            kelly_used,
            0,
            0.0,
            "sized to zero after the cash-reserve constraint",
        )

    count = stake_cents / price_cents
    if max_contracts is not None and count > max_contracts:
        count = max(0.0, max_contracts)
        stake_cents = round(count * price_cents)

    if count <= 0 or stake_cents <= 0:
        return ApexDecision(
            "pass",
            edge,
            kelly_raw,
            kelly_used,
            0,
            0.0,
            "no book depth available to fill any size",
        )

    if stake_cents < MIN_STAKE_CENTS:
        return ApexDecision(
            "pass",
            edge,
            kelly_raw,
            kelly_used,
            0,
            0.0,
            f"stake {stake_cents}c below the {MIN_STAKE_CENTS}c minimum viable size",
        )

    return ApexDecision("buy", edge, kelly_raw, kelly_used, stake_cents, count, "edge and sizing checks passed")

"""Kalshi's own fee schedule.

Deliberately a separate module from polymarket.us's fee model rather than
an import of it. The two venues charge different amounts and round
differently, and quietly reusing the sibling platform's constant would
understate cost on every single trade:

  polymarket.us   theta = 0.06, banker's rounding (ties to even)
  Kalshi          theta = 0.07, rounded UP

Kalshi's published general formula is

    fee = roundup(0.07 * C * P * (1 - P))

with P in dollars and C the contract count. Working in cents (p = 1..99):

    fee_cents = ceil(0.07 * count * p * (100 - p) / 100)

Worked example from Kalshi's own documentation: 100 contracts at $0.50 is
0.07 * 100 * 0.5 * 0.5 = $1.75, which this reproduces as 175 cents.

IMPORTANT — this is the published formula, not a measured one. The
sibling platform's theta was only pinned down by reconciling against real
fills from the account's own activity feed, and the pre-verification value
there turned out to be wrong. Per `docs/kalshi-bot-operating-manual.md`
§6.3, reconcile this against real Kalshi settlements before going live.
Rounding UP is the safe direction to be wrong in: it never understates
cost, so a trade that clears the edge floor here still clears it if the
real charge is lower.

Maker fees are modelled as zero, which matches Kalshi's general schedule
for most series. Series that do charge makers are not modelled — another
reason §6.3 requires reconciliation against real fills before live use.
"""

from __future__ import annotations

from decimal import ROUND_CEILING, Decimal

# Kalshi's general trading fee coefficient.
THETA_TAKER = Decimal("0.07")


def taker_fee_cents(price_cents: int, count: float) -> int:
    """Fee charged for taking `count` contracts at `price_cents`.

    Rounded up to the next whole cent, per Kalshi's published rule.
    """
    if count <= 0 or not (0 < price_cents < 100):
        return 0
    p = Decimal(price_cents)
    # count may arrive as a float from Kelly sizing (stake / price). str()
    # first because Decimal * float is a TypeError, and going through str
    # avoids silently truncating a fractional contract count.
    exact = THETA_TAKER * p * (100 - p) * Decimal(str(count)) / 100
    return int(exact.quantize(Decimal("1"), rounding=ROUND_CEILING))


def maker_fee_cents(price_cents: int, count: float) -> int:
    """Fee charged to the resting side. Zero on Kalshi's general schedule.

    Exists so callers can price a maker fill explicitly rather than
    assuming free, and so a future series-specific schedule has one place
    to land.
    """
    return 0


def effective_cost_cents(price_cents: int, count: float) -> int:
    """Total cash out the door to buy `count` contracts at `price_cents`,
    fee included — the number that actually leaves the balance."""
    return round(price_cents * count) + taker_fee_cents(price_cents, count)


def breakeven_win_rate(price_cents: int, count: float = 1) -> float:
    """Win rate (as a percentage, 0-100) a position must beat just to break
    even after the taker fee.

    A contract bought at p pays 100 on a win and 0 on a loss, so with an
    amortized per-contract fee f the breakeven probability is (p + f) / 100.
    This — not p itself — is what the calibration table has to clear; using
    p alone is what lets a nominal 5pp "edge" evaporate into a loss.

    `count` matters because Kalshi rounds the fee up per order, so the
    per-contract fee is largest for a 1-contract order and amortizes down
    as size grows. The default of 1 is therefore the conservative case:
    it never understates the hurdle. Pass the real intended count to price
    an actual order.
    """
    if not (0 < price_cents < 100) or count <= 0:
        return 100.0
    fee_per_contract = taker_fee_cents(price_cents, count) / count
    return min(100.0, price_cents + fee_per_contract)

"""Ladder-arbitrage strategy for polymarket.us.

polymarket.us has no crypto/short-duration markets (confirmed by scanning
9,500+ live active markets — it's almost entirely sports player-props and
futures), so the convergence-bot approach used elsewhere in this repo
doesn't apply here. Instead this targets a real structural feature: markets
that come in families where one outcome logically implies another, so their
prices are constrained by arithmetic rather than by anyone's forecast.

The general form
----------------
Whenever event X implies event Y, coherent pricing requires P(X) <= P(Y).
Buy YES on Y and NO on X and the payoff is >= $1/contract in every state:

  - X happens (so Y happens): YES_Y pays $1, NO_X pays $0  -> $1
  - Y without X:              YES_Y pays $1, NO_X pays $1  -> $2
  - neither:                  YES_Y pays $0, NO_X pays $1  -> $1

Cost is ask(Y) + (100 - bid(X)), so the trade is free money exactly when
bid(X) > ask(Y). This is a mathematical constraint, not a probability
estimate, so — unlike the crypto/macro mispricing bots that were removed
elsewhere in this repo for relying on a miscalibrated model with no
supporting data — trading a violation of it needs no historical calibration.

Which families qualify
----------------------
The implication direction is a property of what the market *means*, and it
is not always the direction the slug numbering suggests:

  `-gteN`        "N+ goals". gte(n) implies gte(n-1), so the subset is the
                 HIGHER-numbered rung and prices fall as N rises.
  `-before-rN`   "Fight ends before Round N begins". before(n) implies
                 before(n+1), so the subset is the LOWER-numbered rung and
                 prices RISE as N rises — the opposite direction.

Getting that backwards turns a hedge into a doubled directional bet, so
each family declares its direction explicitly rather than inferring one.

Not every numbered family is a ladder at all. `-rov-f1-rN` ("Gamrot in
Round 1", "...Round 2") are mutually exclusive outcomes — winning in round
1 does not imply winning in round 2 — and pricing them as nested would
manufacture losses while logging them as guaranteed profit. Families are
therefore an explicit allowlist of verified patterns, never a general
"slug ends in a number" rule. Read the market titles before adding one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Callable

from src.trading.polymarket_us.models import Market

# Verified against docs.polymarket.us/fees.md (2026-08-03), effective
# exchange-wide from 2026-07-01: fee = theta * count * p * (1-p), rounded to
# the nearest cent with banker's rounding (ties to even). theta_taker=0.06
# reproduces all 9 real fills from this account (/v1/portfolio/activities,
# 2026-08-01) exactly — see test_fee_estimate_never_understates_observed_commissions.
# The prior 0.07/ceil formula was a pre-verification safety pad, not measured
# truth. Multi-fill sweeps (one aggressive order crossing several resting
# prices) aren't modeled here — every caller today prices a leg as a single
# fill at a single price.
THETA_TAKER = Decimal("0.06")
THETA_MAKER = Decimal("-0.0125")


def _fee_cents_exact(theta: Decimal, price_cents: int, count: float) -> Decimal:
    p = Decimal(price_cents)
    # count arrives as a float from any caller sizing via Kelly (stake_cents /
    # price_cents, deliberately never int()'d -- see apex/bot.py's fill-count
    # paranoia note). Decimal * float raises TypeError; str() first avoids
    # that without silently truncating a fractional contract count to int.
    return theta * p * (100 - p) * Decimal(str(count)) / 100


def taker_fee_cents(price_cents: int, count: float) -> int:
    """Commission charged to the aggressor for taking `count` contracts at
    `price_cents`. See THETA_TAKER above for the source and rounding rule."""
    exact = _fee_cents_exact(THETA_TAKER, price_cents, count)
    return int(exact.quantize(Decimal("1"), rounding=ROUND_HALF_EVEN))


def maker_rebate_cents(price_cents: int, count: float) -> int:
    """Rebate credited to the resting side when `count` contracts fill at
    `price_cents`. Positive = credit. See THETA_MAKER above."""
    exact = -_fee_cents_exact(THETA_MAKER, price_cents, count)
    return int(exact.quantize(Decimal("1"), rounding=ROUND_HALF_EVEN))


def _int_index(token: str) -> int | None:
    try:
        return int(token)
    except ValueError:
        return None


def _date_index(token: str) -> int | None:
    """Normalise a slug date to a sortable YYYYMMDD integer.

    Both orderings occur live (`prdc-gtavi-12-31-2026` and
    `ipcc-openai-2027-06-30`), and they are not ambiguous: a YYYY-MM-DD
    string can't parse as MM-DD-YYYY (month 2026) and vice versa (month 31).
    Ordering these wrong would invert the nesting direction and turn the
    hedge into a doubled directional bet, so an unparseable date returns
    None and the rung is dropped rather than guessed at.
    """
    for fmt in ("%Y-%m-%d", "%m-%d-%Y"):
        try:
            return int(datetime.strptime(token, fmt).strftime("%Y%m%d"))
        except ValueError:
            continue
    return None


# Slug prefixes whose settlement text has been read and confirmed to be
# cumulative "by <date>" (or "at any point before <date>") wording, so that
# an earlier date genuinely implies a later one.
#
# Deliberately excluded: `aec` — 29 of the 36 date-suffixed families, and not
# ladders at all. They are per-game winner markets ("Who will win St. Louis
# vs Toronto"), where the trailing date is just the fixture date. Pairing two
# of those would "arbitrage" two unrelated baseball games.
#
# Do not add a prefix here from its question title. `cpc` reads "When will
# Bitcoin hit $150k?", which sounds like exclusive buckets, but its
# description is "above $149,999.99 at any point before <date>" — cumulative.
# The title misleads in both directions; only the description settles it.
DATE_LADDER_PREFIXES = (
    "aimc",  # "releases GPT-6 by <date>"
    "apdc",  # "departure officially announced by <date>"
    "cpc",  # "above $X at any point before <date>"
    "ipcc",  # "confirms an IPO by <date>"
    "lawec",  # "enacted into U.S. federal law by <date>"
    "mowc",  # "formally initiates withdrawal by <date>"
    "opdc",  # "resigns by <date>"
    "pandc",  # "declares any disease a pandemic by <date>"
    "pintc",  # "meets in person between issuance and <date>"
    "prdc",  # "releases GTA VI by <date>"
    "stsc",  # "for any day between <start> and <date>"
)

MIN_EDGE_CENTS = 1  # smallest gross edge per contract worth even evaluating
MAX_PLAUSIBLE_EDGE_CENTS = 20  # bigger than this is far more likely stale/bad data than real free money

# Smallest profit per contract AFTER estimated fees. The live GTA VI lock
# (2026-07-31) cost 99c/contract and pays $1.00 at settlement: +1c gross,
# which entry commissions of ~1-2c turn into breakeven-or-worse. "Guaranteed"
# arithmetic that ignores fees books losers as winners.
MIN_NET_PROFIT_CENTS = 1

# Extra net edge demanded per 30 days the capital stays locked, per contract.
# A lock costs up to ~$1/contract and pays out only at settlement, so 2c per
# 30 days is roughly a 24%/yr hurdle rate. Same-day sports ladders pass on
# 1c net; a 1c-net lock settling four months out (the GTA VI trade: ~3%/yr,
# worse than a treasury bill) is correctly rejected.
LOCKUP_CENTS_PER_30_DAYS = 2


def net_edge_cents(buy_price: int, sell_price: int) -> int:
    """Per-contract profit after estimated entry fees on both legs.

    The NO leg is bought at (100 - sell_price), and fees apply to what each
    leg actually costs. Per-contract ceil overestimates slightly on
    multi-contract orders, which is the safe direction for a floor.
    """
    gross = sell_price - buy_price
    fees = taker_fee_cents(buy_price, 1) + taker_fee_cents(100 - sell_price, 1)
    return gross - fees


def required_net_edge_cents(close_time: datetime | None, now: datetime | None = None) -> int:
    """Minimum acceptable net edge per contract for a lock settling at `close_time`.

    A missing close_time is treated as imminent (sports markets settle same
    day); the fee floor still applies, only the lock-up premium is waived.
    """
    if close_time is None:
        return MIN_NET_PROFIT_CENTS
    now = now or datetime.now(timezone.utc)
    ct = close_time if close_time.tzinfo else close_time.replace(tzinfo=timezone.utc)
    days = max(0.0, (ct - now).total_seconds() / 86400.0)
    return MIN_NET_PROFIT_CENTS + int(days) * LOCKUP_CENTS_PER_30_DAYS // 30


DEFAULT_LOCK_BUDGET_CENTS = 500  # $5, matching the flat bet-size convention used across this repo
MAX_CONTRACTS = 10  # cap size regardless of budget — book depth isn't visible from the bulk markets scan

# Reject a rung whose own book is this wide. Measured across 2,198 live
# two-sided ladder rungs: p50=5c, p75=13c, p90=32c. Past ~20c the quotes are
# a market maker's placeholder rather than a price anyone will trade at, so
# an "edge" computed against them is arithmetic on noise. This is hygiene,
# not the main defence — the binding check is the fresh-BBO re-quote the bot
# does immediately before committing (see LadderArbBot._place_lock); a lock
# found here is a candidate, not a decision.
MAX_SPREAD_CENTS = 20

# NOTE: do not add a volume floor here. Every one of those 2,198 rungs
# reported volume=0 and volume24hr=0, including markets with active
# two-sided quotes that demonstrably trade — this API just doesn't populate
# those fields on the markets endpoints. Filtering on them rejects the
# entire catalogue and silently disables the strategy.


@dataclass(frozen=True)
class LadderFamily:
    """A set of markets whose outcomes nest, and which way the nesting runs.

    `subset_is_higher_index` says which rung implies the other: True when the
    higher-indexed rung is the stricter event (gte3 implies gte2), False when
    the lower-indexed one is (before-r2 implies before-r3; released-by-March
    implies released-by-June).

    `prefix_allowlist`, when set, restricts the family to slugs whose first
    hyphen segment is listed. Pattern shape alone is not evidence that a set
    of markets nests — see DATE_LADDER_PREFIXES.
    """

    name: str
    regex: re.Pattern
    subset_is_higher_index: bool
    index_parser: Callable[[str], int | None] = _int_index
    prefix_allowlist: tuple | None = None


LADDER_FAMILIES = (
    # "N+ goals / hits / strikeouts" player-prop threshold ladders.
    LadderFamily("gte", re.compile(r"^(.*-gte)(\d+)$"), subset_is_higher_index=True),
    # "Fight ends before Round N begins". Deliberately anchored on
    # `-before-r` and not `-r`, so it cannot pick up the `-rov-f1-rN`
    # win-in-round-N markets, which are mutually exclusive rather than nested.
    LadderFamily("before_round", re.compile(r"^(.*-before-r)(\d+)$"), subset_is_higher_index=False),
    # "<thing> happens by <date>". An earlier deadline implies a later one,
    # so the subset is the lower (earlier) index, same direction as rounds.
    LadderFamily(
        "by_date",
        re.compile(r"^(.*?)-(\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4})$"),
        subset_is_higher_index=False,
        index_parser=_date_index,
        prefix_allowlist=DATE_LADDER_PREFIXES,
    ),
)


@dataclass
class LadderLock:
    buy_slug: str  # the implied (superset) market — buy YES here
    buy_price: int  # its yes_ask, cents
    sell_slug: str  # the implying (subset) market — buy NO here
    sell_price: int  # its yes_bid, cents (NO costs 100 - sell_price)
    count: int
    edge_cents: int  # guaranteed profit per contract
    total_cost_cents: int  # both legs together — a live caller must gate on this, see below
    family: str = "gte"  # which LadderFamily produced this, for logging


def _has_tradeable_book(m: Market) -> bool:
    """Both sides quoted and the spread tight enough for the quotes to mean
    something. A one-sided book gives no way to tell a real price from a
    stale placeholder, so it is treated the same as a wide one."""
    if m.yes_bid is None or m.yes_ask is None:
        return False
    return (m.yes_ask - m.yes_bid) <= MAX_SPREAD_CENTS


def _group_ladders(markets: list[Market]) -> dict[tuple, list[tuple[int, Market]]]:
    """Group rungs by (family, shared prefix). Keyed on the family too so two
    families that happen to share a prefix can never be paired together."""
    groups: dict[tuple, list[tuple[int, Market]]] = {}
    for m in markets:
        if not _has_tradeable_book(m):
            continue
        for family in LADDER_FAMILIES:
            match = family.regex.match(m.slug)
            if not match:
                continue
            if family.prefix_allowlist is not None:
                if m.slug.split("-", 1)[0] not in family.prefix_allowlist:
                    break  # shape matches but this family isn't verified to nest
            index = family.index_parser(match.group(2))
            if index is None:
                break
            groups.setdefault((family.name, match.group(1)), []).append((index, m))
            break
    return groups


def _size_lock(buy_price: int, sell_price: int, budget_cents: int) -> int:
    """Contracts affordable within `budget_cents`, or 0 if not even one is.

    Returning 0 rather than rounding up to 1 is what keeps a live account
    from stranding a naked leg: both legs are paid for out of the same
    balance, so a lock the budget can't cover in full must not be started
    at all. An earlier `max(1, ...)` here forced a minimum of one contract
    regardless of budget — harmless in paper mode (the paper ledger just
    raises insufficient_balance before anything moves) but on a real
    account it fills the YES leg and then gets the NO leg rejected, which
    is an unhedged position no cancel can undo.
    """
    cost_per_contract = buy_price + (100 - sell_price)
    if cost_per_contract <= 0:
        return 0
    count = budget_cents // cost_per_contract
    return min(count, MAX_CONTRACTS)


def find_ladder_locks(
    markets: list[Market],
    held: set | None = None,
    budget_cents: int = DEFAULT_LOCK_BUDGET_CENTS,
) -> list[LadderLock]:
    """Monotonicity violations sized to fit `budget_cents` per lock.

    Callers trading a real account must pass their actual buying power
    (see LadderArbBot._scan_once) — the default is only a sensible size
    for paper mode, where the ledger starts at $1,000.
    """
    held = held or set()
    locks: list[LadderLock] = []
    by_name = {f.name: f for f in LADDER_FAMILIES}
    now = datetime.now(timezone.utc)

    for (family_name, _prefix), rungs in _group_ladders(markets).items():
        family = by_name[family_name]
        rungs.sort(key=lambda t: t[0])
        best: tuple[int, Market, Market] | None = None  # (net_edge, superset, subset)

        for i in range(len(rungs)):
            for j in range(i + 1, len(rungs)):
                lower, higher = rungs[i][1], rungs[j][1]
                # The subset is the stricter event; it must not price above
                # the event it implies.
                subset, superset = (higher, lower) if family.subset_is_higher_index else (lower, higher)
                if subset.slug in held or superset.slug in held:
                    continue
                if subset.yes_bid is None or superset.yes_ask is None:
                    continue
                edge = subset.yes_bid - superset.yes_ask
                if edge < MIN_EDGE_CENTS or edge > MAX_PLAUSIBLE_EDGE_CENTS:
                    continue
                # Fees first, then the lock-up hurdle: capital is tied until
                # the LATER of the two legs settles.
                net = net_edge_cents(superset.yes_ask, subset.yes_bid)
                closes = [m.close_time for m in (subset, superset) if m.close_time]
                if net < required_net_edge_cents(max(closes) if closes else None, now):
                    continue
                if best is None or net > best[0]:
                    best = (net, superset, subset)

        if best is None:
            continue
        _net, superset, subset = best
        edge = subset.yes_bid - superset.yes_ask  # gross, for logging/limits
        count = _size_lock(superset.yes_ask, subset.yes_bid, budget_cents)
        if count <= 0:
            continue
        locks.append(
            LadderLock(
                buy_slug=superset.slug,
                buy_price=superset.yes_ask,
                sell_slug=subset.slug,
                sell_price=subset.yes_bid,
                count=count,
                edge_cents=edge,
                total_cost_cents=(superset.yes_ask + (100 - subset.yes_bid)) * count,
                family=family.name,
            )
        )

    return locks

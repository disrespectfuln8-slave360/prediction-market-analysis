"""Ladder-arbitrage strategy for polymarket.us.

polymarket.us has no crypto/short-duration markets (confirmed by scanning
9,500+ live active markets — it's almost entirely sports player-props and
futures), so the convergence-bot approach used elsewhere in this repo
doesn't apply here. Instead this targets a real structural feature: many
player-stat markets come in threshold "ladders" — e.g.
`...-fwcmikoya-gte1`, `...-gte2`, `...-gte3` for "1+ goals", "2+ goals",
"3+ goals" on the same player/game. Since gte(n) implies gte(n-1), the
probability (and therefore price) of gte(n) must be non-increasing in n.
This is a mathematical constraint on any coherent pricing, not a
probability estimate, so — unlike the crypto/macro mispricing bots that
were removed elsewhere in this repo for relying on a miscalibrated model
with no supporting data — trading a violation of it needs no historical
calibration.

The arbitrage: buy YES on the lower rung (m1) and buy NO on the higher
rung (m2). Payoff is >= $1/contract in every possible outcome (worked out
below), so whenever cost < $1/contract (i.e. m1.yes_ask < m2.yes_bid) it's
a guaranteed profit:
  - stat >= n2 (implies >= n1): YES1 pays $1, NO2 pays $0  -> $1
  - n1 <= stat < n2:            YES1 pays $1, NO2 pays $1  -> $2
  - stat < n1 (implies < n2):   YES1 pays $0, NO2 pays $1  -> $1
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from src.trading.polymarket_us.models import Market

LADDER_RE = re.compile(r"^(.*-gte)(\d+)$")

MIN_EDGE_CENTS = 1  # smallest guaranteed profit per contract worth acting on
MAX_PLAUSIBLE_EDGE_CENTS = 20  # bigger than this is far more likely stale/bad data than real free money
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


@dataclass
class LadderLock:
    buy_slug: str  # lower threshold market — buy YES here
    buy_price: int  # yes_ask, cents
    sell_slug: str  # higher threshold market — buy NO here
    sell_price: int  # yes_bid, cents (NO costs 100 - sell_price)
    count: int
    edge_cents: int  # guaranteed profit per contract
    total_cost_cents: int  # both legs together — a live caller must gate on this, see below


def _has_tradeable_book(m: Market) -> bool:
    """Both sides quoted and the spread tight enough for the quotes to mean
    something. A one-sided book gives no way to tell a real price from a
    stale placeholder, so it is treated the same as a wide one."""
    if m.yes_bid is None or m.yes_ask is None:
        return False
    return (m.yes_ask - m.yes_bid) <= MAX_SPREAD_CENTS


def _group_ladders(markets: list[Market]) -> dict[str, list[tuple[int, Market]]]:
    groups: dict[str, list[tuple[int, Market]]] = {}
    for m in markets:
        match = LADDER_RE.match(m.slug)
        if not match:
            continue
        if not _has_tradeable_book(m):
            continue
        groups.setdefault(match.group(1), []).append((int(match.group(2)), m))
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
    held: Optional[set] = None,
    budget_cents: int = DEFAULT_LOCK_BUDGET_CENTS,
) -> list[LadderLock]:
    """Ladder violations sized to fit `budget_cents` per lock.

    Callers trading a real account must pass their actual buying power
    (see LadderArbBot._scan_once) — the default is only a sensible size
    for paper mode, where the ledger starts at $1,000.
    """
    held = held or set()
    locks: list[LadderLock] = []

    for rungs in _group_ladders(markets).values():
        rungs.sort(key=lambda t: t[0])
        best: Optional[tuple[int, Market, Market]] = None  # (edge, m1, m2)
        for i in range(len(rungs)):
            _, m1 = rungs[i]
            if m1.yes_ask is None or m1.slug in held:
                continue
            for j in range(i + 1, len(rungs)):
                _, m2 = rungs[j]
                if m2.yes_bid is None or m2.slug in held:
                    continue
                edge = m2.yes_bid - m1.yes_ask
                if edge < MIN_EDGE_CENTS or edge > MAX_PLAUSIBLE_EDGE_CENTS:
                    continue
                if best is None or edge > best[0]:
                    best = (edge, m1, m2)
        if best is None:
            continue
        edge, m1, m2 = best
        count = _size_lock(m1.yes_ask, m2.yes_bid, budget_cents)
        if count <= 0:
            continue
        locks.append(LadderLock(
            buy_slug=m1.slug, buy_price=m1.yes_ask,
            sell_slug=m2.slug, sell_price=m2.yes_bid,
            count=count, edge_cents=edge,
            total_cost_cents=(m1.yes_ask + (100 - m2.yes_bid)) * count,
        ))

    return locks

"""Ladder-arbitrage execution loop for polymarket.us.

Execution model, and why it looks paranoid
------------------------------------------
On 2026-07-30 this bot placed two YES legs, was told by the API that neither
had filled, cancelled both, logged "skipping lock", and moved on. Both had in
fact filled. The account was left holding two unhedged long legs on opposite
rungs of the same ladder — the exact directional risk this strategy exists to
avoid — in a market that does not resolve until November.

The root cause is not the cancel logic. It is that the order-creation
response on this API does not reliably report fills: it returned an empty
executions array for orders that had filled completely. This was reproduced
afterwards on the two flattening sells, which also reported zero fills while
actually filling in full. Any fill/no-fill decision made from that response
is a coin flip.

So the rules here are:
  1. Never infer fill state from the order response. Read it back from
     get_positions(), which is the only source that has been observed to
     tell the truth.
  2. Both legs go in immediate-or-cancel. IOC leaves no resting remainder,
     which removes the fill-versus-cancel race entirely rather than trying
     to win it.
  3. Hedge the fill that actually happened, not the fill that was intended.
     A partially filled first leg is hedged at its real size.
  4. Re-quote both legs immediately before committing. A scan of ~13,000
     markets takes about a minute, and orders priced off the top of that
     scan are being sent against quotes that have already moved.
  5. If an unhedged remainder cannot be flattened, stop trading and say so.
     Carrying naked directional risk is worse than placing no trades.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional

from src.trading.polymarket_us.strategy import (
    DEFAULT_LOCK_BUDGET_CENTS,
    MAX_PLAUSIBLE_EDGE_CENTS,
    MIN_EDGE_CENTS,
    LadderLock,
    find_ladder_locks,
    net_edge_cents,
    required_net_edge_cents,
)

logger = logging.getLogger(__name__)

# Fraction of reported buying power the bot is allowed to commit per scan.
# Both legs are paid out of one balance, so sizing to the exact figure leaves
# no room for a fee charged at fill or a cent of rounding on the second leg —
# and coming up short there is not a skipped trade, it's a naked position.
BALANCE_SAFETY_FRACTION = 0.95

# How long to let the exchange's position view catch up with an order we just
# sent. With IOC the order itself is resolved at placement, so this is only
# covering propagation delay, not waiting for a fill to happen.
FILL_POLL_ATTEMPTS = 5
FILL_POLL_DELAY = 1.0

# After a lock fails to hedge, stop retrying that pair for a while. Flattening
# an unhedged leg costs the spread, so a pair that fails repeatedly is not a
# missed opportunity, it is a recurring bill. Live on 2026-07-31 the same
# three pairs were re-attempted every scan for 35 minutes and burned 79% of
# the account without ever completing a lock.
FAILED_PAIR_COOLDOWN_SECS = 3600.0

# Consecutive failed hedges before the bot stops trading altogether. Cooldowns
# handle one bad pair; this catches the case where conditions are broadly
# wrong (an API degradation, a systemic mispricing of the short side) and
# every pair is failing.
MAX_CONSECUTIVE_HEDGE_FAILURES = 3

# Contracts to risk when the book depth is unknown. get_market() supplies
# depth from /bbo, so this only applies if that field ever goes missing —
# in which case one contract is the most that can be lost to a failed hedge.
UNKNOWN_DEPTH_CONTRACTS = 1


class LadderArbBot:
    """Scans polymarket.us for ladder-monotonicity violations and places
    guaranteed-profit lock trades (see strategy.py for why this needs no
    historical calibration — it's a math constraint, not a probability
    model).
    """

    def __init__(
        self,
        client,
        poll_interval: int = 60,
        max_open_legs: int = 20,
        max_lock_budget_cents: int = DEFAULT_LOCK_BUDGET_CENTS,
    ):
        self.client = client
        self.poll_interval = poll_interval
        self.max_open_legs = max_open_legs
        self.max_lock_budget_cents = max_lock_budget_cents
        # Set when an unhedged remainder could not be flattened, or when too
        # many hedges fail in a row. Placing more locks while carrying unknown
        # directional risk compounds the problem, so the bot keeps scanning
        # and reporting but stops trading.
        self._halted_reason: Optional[str] = None
        # (buy_slug, sell_slug) -> monotonic time the pair may be retried.
        self._cooldown: dict[tuple, float] = {}
        self._consecutive_failures = 0

    def _held_slugs(self, positions, open_orders) -> set:
        return {p.market_slug for p in positions} | {o.market_slug for o in open_orders}

    # ── Position readback ────────────────────────────────────────────────────

    def _net_position(self, slug: str) -> float:
        for p in self.client.get_positions():
            if p.market_slug == slug:
                return float(p.position)
        return 0.0

    def _position_after_order(self, slug: str, before: float) -> float:
        """Net position once it reflects the order just sent.

        Returns as soon as it moves off `before`; if it never moves, returns
        the last reading, which is the honest answer for "nothing filled".
        """
        latest = before
        for _ in range(FILL_POLL_ATTEMPTS):
            time.sleep(FILL_POLL_DELAY)
            latest = self._net_position(slug)
            if latest != before:
                return latest
        return latest

    # ── Fresh quotes ─────────────────────────────────────────────────────────

    def _requote(self, lock: LadderLock) -> Optional[tuple[int, int, int]]:
        """Re-price both legs against a fresh per-market BBO.

        Returns (buy_price, sell_price, max_contracts) if the edge still
        holds, else None. get_market() fetches the market's own BBO endpoint,
        which is both fresher than the bulk scan snapshot and the only source
        of book depth.

        max_contracts is what the two books can actually absorb. Buying YES
        lifts the superset's ask side; buying NO is economically selling YES,
        so it hits the subset's bid side. Sizing past either is what turns a
        lock into a half-filled position that has to be unwound at a loss.
        """
        try:
            m1 = self.client.get_market(lock.buy_slug)
            m2 = self.client.get_market(lock.sell_slug)
        except Exception as exc:
            logger.warning("Re-quote failed for %s / %s: %s",
                           lock.buy_slug, lock.sell_slug, exc)
            return None

        if m1.yes_ask is None or m2.yes_bid is None:
            logger.debug("Re-quote for %s lost a side — skipping", lock.buy_slug)
            return None

        edge = m2.yes_bid - m1.yes_ask
        if edge < MIN_EDGE_CENTS or edge > MAX_PLAUSIBLE_EDGE_CENTS:
            logger.info(
                "Edge on %s gone at re-quote (scan +%dc -> live %+dc) — skipping",
                lock.buy_slug, lock.edge_cents, edge,
            )
            return None

        # Same fee + lock-up standard the scan applied, but against the
        # fresh prices actually about to be traded.
        net = net_edge_cents(m1.yes_ask, m2.yes_bid)
        closes = [m.close_time for m in (m1, m2) if m.close_time]
        required = required_net_edge_cents(max(closes) if closes else None)
        if net < required:
            logger.info(
                "Net edge on %s too thin at re-quote (%+dc after fees, need %+dc) — skipping",
                lock.buy_slug, net, required,
            )
            return None

        depths = [d for d in (m1.ask_depth, m2.bid_depth) if d is not None]
        if len(depths) < 2:
            max_contracts = UNKNOWN_DEPTH_CONTRACTS
            logger.debug("Depth unavailable for %s — capping at %d contract(s)",
                         lock.buy_slug, max_contracts)
        else:
            max_contracts = min(depths)
        if max_contracts <= 0:
            logger.info("No book depth on %s / %s — skipping",
                        lock.buy_slug, lock.sell_slug)
            return None
        return m1.yes_ask, m2.yes_bid, max_contracts

    # ── Flattening ───────────────────────────────────────────────────────────

    def _flatten_long(self, slug: str, count: int) -> bool:
        """Sell `count` contracts of an unhedged long. True if it's gone."""
        try:
            market = self.client.get_market(slug)
            bid = market.yes_bid
            if not bid or bid < 1:
                logger.error("Cannot flatten %s — no bid to sell into", slug)
                return False
            self.client.close_long(market_slug=slug, count=count, price_cents=bid)
        except Exception as exc:
            logger.error("Flatten order failed on %s: %s", slug, exc)
            return False

        # Same rule as everywhere else here: believe the position, not the
        # order response. The flattening sells during the incident post-mortem
        # reported zero fills while fully filling.
        remaining = self._position_after_order(slug, float(count))
        if abs(remaining) < 1e-9:
            logger.info("Flattened %dx %s", count, slug)
            return True
        logger.error("Flatten of %s incomplete — %s still held", slug, remaining)
        return False

    # ── Lock placement ───────────────────────────────────────────────────────

    def _place_lock(self, lock: LadderLock, budget_cents: int) -> int:
        """Place both legs of a lock; returns the cents actually committed.

        The return value counts money that left the account even when the
        lock failed, so the caller's remaining budget stays honest.
        """
        quotes = self._requote(lock)
        if quotes is None:
            return 0
        buy_price, sell_price, max_contracts = quotes

        cost_per_contract = buy_price + (100 - sell_price)
        count = min(budget_cents // cost_per_contract, lock.count, max_contracts)
        if count <= 0:
            logger.debug("Re-quoted %s no longer fits the budget/book — skipping",
                         lock.buy_slug)
            return 0

        before_buy = self._net_position(lock.buy_slug)
        try:
            self.client.place_order(
                market_slug=lock.buy_slug, side="yes",
                count=count, yes_price=buy_price, immediate_or_cancel=True,
            )
        except Exception as exc:
            logger.warning("YES leg order rejected on %s: %s", lock.buy_slug, exc)
            # It was rejected outright, but the response is not trustworthy
            # enough to assume that means no position — check.
            stray = self._net_position(lock.buy_slug) - before_buy
            if stray > 0:
                logger.error(
                    "CRITICAL: YES leg on %s reported an error but left %s contracts",
                    lock.buy_slug, stray,
                )
                if not self._flatten_long(lock.buy_slug, int(stray)):
                    self._halt(f"unflattened {stray}x long on {lock.buy_slug}")
                return int(stray) * buy_price
            return 0

        yes_filled = int(self._position_after_order(lock.buy_slug, before_buy) - before_buy)
        if yes_filled <= 0:
            logger.info("YES leg on %s did not fill (IOC) — no position taken", lock.buy_slug)
            return 0
        if yes_filled < count:
            logger.info("YES leg on %s partially filled %d/%d — hedging the real fill",
                        lock.buy_slug, yes_filled, count)

        spent = yes_filled * buy_price

        before_sell = self._net_position(lock.sell_slug)
        try:
            self.client.place_order(
                market_slug=lock.sell_slug, side="no",
                count=yes_filled, yes_price=sell_price, immediate_or_cancel=True,
            )
        except Exception as exc:
            logger.error("NO leg order rejected on %s: %s", lock.sell_slug, exc)

        # Buying NO is a short, so the net position moves negative.
        after_sell = self._position_after_order(lock.sell_slug, before_sell)
        no_filled = int(before_sell - after_sell)
        spent += max(0, no_filled) * (100 - sell_price)

        unhedged = yes_filled - max(0, no_filled)
        if unhedged <= 0:
            logger.info(
                "LOCK: %dx YES %s@%dc + %dx NO %s@%dc  edge=+%dc/ct gross, ~%+dc/ct net of fees  cost=$%.2f",
                yes_filled, lock.buy_slug, buy_price,
                no_filled, lock.sell_slug, 100 - sell_price,
                sell_price - buy_price, net_edge_cents(buy_price, sell_price),
                spent / 100,
            )
            self._consecutive_failures = 0
            return spent

        logger.error(
            "NO leg on %s covered %d of %d — flattening %dx unhedged long on %s",
            lock.sell_slug, max(0, no_filled), yes_filled, unhedged, lock.buy_slug,
        )
        # Unwinding costs the spread whether or not it succeeds, so this pair
        # has to stop being retried either way.
        self._mark_failed(lock)
        if self._flatten_long(lock.buy_slug, unhedged):
            spent -= unhedged * buy_price  # recovered, roughly — exact P&L is in the balance
        else:
            self._halt(
                f"unhedged {unhedged}x long on {lock.buy_slug} could not be flattened"
            )
        return max(0, spent)

    def _mark_failed(self, lock: LadderLock) -> None:
        """Cool a pair off, and halt if hedging keeps failing across pairs."""
        self._cooldown[(lock.buy_slug, lock.sell_slug)] = (
            time.monotonic() + FAILED_PAIR_COOLDOWN_SECS
        )
        self._consecutive_failures += 1
        logger.info(
            "Cooling off %s / %s for %.0f min (%d consecutive hedge failures)",
            lock.buy_slug, lock.sell_slug, FAILED_PAIR_COOLDOWN_SECS / 60,
            self._consecutive_failures,
        )
        if self._consecutive_failures >= MAX_CONSECUTIVE_HEDGE_FAILURES:
            self._halt(
                f"{self._consecutive_failures} consecutive hedge failures — the "
                "short side is not filling at the quoted price"
            )

    def _in_cooldown(self, lock: LadderLock) -> bool:
        until = self._cooldown.get((lock.buy_slug, lock.sell_slug))
        return until is not None and time.monotonic() < until

    def _halt(self, reason: str) -> None:
        self._halted_reason = reason
        logger.error(
            "CRITICAL: trading halted — %s. The account is carrying directional "
            "risk this strategy is not supposed to take. Flatten it by hand, then "
            "restart the bot.", reason,
        )

    # ── Scan loop ────────────────────────────────────────────────────────────

    def _scan_once(self):
        positions = self.client.get_positions()
        open_orders = self.client.get_open_orders()
        held = self._held_slugs(positions, open_orders)

        open_legs = len(positions) + len(open_orders)
        slots = max(0, (self.max_open_legs - open_legs) // 2)

        # Size every lock against what the account can actually pay for right
        # now. Both legs come out of this one balance, so a lock sized past it
        # doesn't fail cleanly — the YES leg fills and the NO leg gets rejected
        # for insufficient funds, leaving an unhedged position. Re-read each
        # scan because settlements and fills move it between passes.
        #
        # The flat per-lock cap still applies on top: it's the repo-wide bet
        # size, and without it a well-funded account (the $1,000 paper ledger,
        # say) would silently size up to MAX_CONTRACTS on every lock.
        balance = self.client.get_balance()
        budget = min(self.max_lock_budget_cents, int(balance * BALANCE_SAFETY_FRACTION))

        markets = self.client.list_all_markets(active=True, closed=False)
        locks = find_ladder_locks(markets, held, budget_cents=budget)

        logger.info(
            "[%s] markets=%d  ladder_locks_found=%d  open_legs=%d  slots=%d  "
            "balance=$%.2f  budget=$%.2f%s",
            datetime.now().strftime("%H:%M:%S"), len(markets), len(locks),
            open_legs, slots, balance / 100, budget / 100,
            "  HALTED" if self._halted_reason else "",
        )

        if self._halted_reason:
            return

        # Each lock was sized against the full budget independently, so the
        # running total is what stops a second lock from spending money the
        # first one already committed.
        remaining = budget
        for lock in locks[:slots]:
            if self._in_cooldown(lock):
                logger.debug("Skipping %s / %s — cooling off after a failed hedge",
                             lock.buy_slug, lock.sell_slug)
                continue
            if lock.total_cost_cents > remaining:
                logger.debug(
                    "Skipping lock on %s — cost $%.2f exceeds remaining budget $%.2f",
                    lock.buy_slug, lock.total_cost_cents / 100, remaining / 100,
                )
                continue
            remaining -= self._place_lock(lock, remaining)
            if self._halted_reason:
                break

    def run_forever(self, stop_event=None):
        while True:
            try:
                self._scan_once()
            except Exception as exc:
                logger.warning("Ladder bot scan error: %s", exc)
            if stop_event is not None:
                if stop_event.wait(timeout=self.poll_interval):
                    break
            else:
                time.sleep(self.poll_interval)

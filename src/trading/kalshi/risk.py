"""Deterministic sizing and risk gates for the Kalshi bots.

This module is the executable form of `docs/kalshi-bot-operating-manual.md`
§5. Every number here has a rationale in that document, and every gate is
plain arithmetic so it can be unit-tested and audited — nothing here is
left to a model, a heuristic, or an operator's judgement at 2am.

Two layers:

  `evaluate()`   per-position gates. Price band -> after-fee edge floor ->
                 Kelly sizing -> position cap -> cash reserve -> liquidity
                 -> minimum viable stake. A BUY only survives every gate.

  `RiskLedger`   portfolio-level circuit breakers that outlive any single
                 position: a same-day loss limit that pauses until
                 tomorrow, and a peak-to-trough drawdown kill switch that
                 is a HARD stop — it does not lift on its own and does not
                 lift at midnight. Re-arming is a deliberate human act,
                 because every incident in the manual's §4 ledger was
                 something that kept running while nobody was looking.

The price band deserves its own note. The manual (§3, "known false")
records a standing operator decision from 2026-08-19 — "i dont want long
shot strat, thats just setting myself up to fail" — reinforced by measured
evidence on the sibling platform that every longshot bucket actually
traded was also among the thinnest-sampled (8, 19, 29 rows). Cheap prices
are precisely where a handful of extra outcomes moves the measured win
rate by double digits, which is how noise gets mistaken for edge. The band
is a standing preference, not a tunable to widen after a losing stretch.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from src.trading.kalshi.fees import effective_cost_cents, taker_fee_cents

logger = logging.getLogger(__name__)

# ── Per-position gates (manual §5, "Position level") ──────────────────────
MIN_TRADEABLE_PRICE_CENTS = 30
MAX_TRADEABLE_PRICE_CENTS = 70
MIN_EDGE_PP = 4.0  # after-fee edge, in percentage points
MAX_KELLY_FRACTION = 0.25  # quarter-Kelly
MAX_POSITION_FRACTION = 0.05  # of bankroll, per market
MIN_STAKE_CENTS = 25

# ── Portfolio gates (manual §5, "Portfolio level") ────────────────────────
MIN_CASH_RESERVE_FRACTION = 0.40
DAILY_LOSS_LIMIT_FRACTION = 0.10  # of the day's starting bankroll
MAX_DRAWDOWN_FRACTION = 0.25  # from peak bankroll — hard stop


@dataclass(frozen=True)
class SizingDecision:
    action: str  # "buy" or "pass"
    edge_pp: float
    kelly_fraction_raw: float
    kelly_fraction_used: float
    stake_cents: int
    count: int
    reason: str


def after_fee_edge_pp(win_rate_pct: float, price_cents: int, count: float = 1) -> float:
    """Edge in percentage points over the fee-inclusive breakeven price.

    `win_rate_pct` is the calibration table's estimate (0-100) that this
    side resolves in our favour. The hurdle is not `price_cents` but the
    price plus the amortized fee — see fees.breakeven_win_rate.
    """
    if not (0 < price_cents < 100) or count <= 0:
        return 0.0
    fee_per_contract = taker_fee_cents(price_cents, count) / count
    return win_rate_pct - (price_cents + fee_per_contract)


def raw_kelly_fraction(win_prob: float, price_frac: float) -> float:
    """Full (un-capped) Kelly fraction for a binary contract costing
    `price_frac` per $1 of payout at true win probability `win_prob`.

    b = net odds = (1 - price_frac) / price_frac
    f* = p - (1 - p) / b
    """
    if not (0.0 < price_frac < 1.0):
        return 0.0
    b = (1.0 - price_frac) / price_frac
    f = win_prob - (1.0 - win_prob) / b
    return max(0.0, f)


def evaluate(
    win_rate_pct: float,
    price_cents: int,
    bankroll_cents: int,
    cash_cents: int,
    max_contracts: int | None = None,
    size_multiplier: float = 1.0,
) -> SizingDecision:
    """Run every per-position gate in order, returning at the first failure.

    `size_multiplier` is the learning tracker's confidence weighting (a
    realized-track-record nudge, never a gate) and is applied to the stake
    after the Kelly caps, so it can only ever shrink a position below the
    caps or nudge it modestly up within them — it can never lift a stake
    past MAX_POSITION_FRACTION.
    """
    if not (1 <= price_cents <= 99):
        return SizingDecision("pass", 0.0, 0.0, 0.0, 0, 0, f"price {price_cents}c outside 1-99c")

    if not (MIN_TRADEABLE_PRICE_CENTS <= price_cents <= MAX_TRADEABLE_PRICE_CENTS):
        return SizingDecision(
            "pass",
            0.0,
            0.0,
            0.0,
            0,
            0,
            f"price {price_cents}c outside the {MIN_TRADEABLE_PRICE_CENTS}-"
            f"{MAX_TRADEABLE_PRICE_CENTS}c band — no longshot bets",
        )

    edge = after_fee_edge_pp(win_rate_pct, price_cents)
    if edge < MIN_EDGE_PP:
        return SizingDecision(
            "pass",
            edge,
            0.0,
            0.0,
            0,
            0,
            f"after-fee edge {edge:+.1f}pp below the {MIN_EDGE_PP:.0f}pp floor",
        )

    if bankroll_cents <= 0 or cash_cents <= 0:
        return SizingDecision("pass", edge, 0.0, 0.0, 0, 0, "no bankroll or no cash available")

    # Kelly is computed against the fee-inclusive cost, so the odds used to
    # size are the odds actually being taken, not the quoted ones.
    price_frac = (price_cents + taker_fee_cents(price_cents, 1)) / 100.0
    kelly_raw = raw_kelly_fraction(win_rate_pct / 100.0, price_frac)
    kelly_used = min(kelly_raw * MAX_KELLY_FRACTION, MAX_POSITION_FRACTION)
    if kelly_used <= 0:
        return SizingDecision(
            "pass",
            edge,
            kelly_raw,
            0.0,
            0,
            0,
            "no positive Kelly fraction despite a passing edge",
        )

    stake_cents = round(kelly_used * bankroll_cents * max(0.0, size_multiplier))
    # The multiplier must never breach the hard position cap.
    stake_cents = min(stake_cents, round(MAX_POSITION_FRACTION * bankroll_cents))

    # Cash reserve: never deploy so much that the account can't ride out a
    # correlated cluster. Unlike the sibling platform's Apex bot there is no
    # high-edge override here — a "huge edge" on a calibration estimate is
    # exactly the case the manual says to distrust (§3), not to lever into.
    reserve_needed = round(MIN_CASH_RESERVE_FRACTION * bankroll_cents)
    if cash_cents - stake_cents < reserve_needed:
        stake_cents = max(0, cash_cents - reserve_needed)

    stake_cents = max(0, min(stake_cents, cash_cents))
    if stake_cents <= 0:
        return SizingDecision(
            "pass",
            edge,
            kelly_raw,
            kelly_used,
            0,
            0,
            "sized to zero by the cash-reserve constraint",
        )

    count = int(stake_cents // price_cents)
    if max_contracts is not None:
        count = min(count, max(0, max_contracts))
    if count <= 0:
        return SizingDecision(
            "pass",
            edge,
            kelly_raw,
            kelly_used,
            0,
            0,
            "no whole contract affordable at this size",
        )

    # Re-price off the whole-contract count, fee included, so the number the
    # caller spends is the number the caller was authorized to spend.
    stake_cents = effective_cost_cents(price_cents, count)
    if stake_cents > cash_cents:
        count -= 1
        if count <= 0:
            return SizingDecision(
                "pass",
                edge,
                kelly_raw,
                kelly_used,
                0,
                0,
                "fee pushes even one contract past available cash",
            )
        stake_cents = effective_cost_cents(price_cents, count)

    if stake_cents < MIN_STAKE_CENTS:
        return SizingDecision(
            "pass",
            edge,
            kelly_raw,
            kelly_used,
            0,
            0,
            f"stake {stake_cents}c below the {MIN_STAKE_CENTS}c minimum viable size",
        )

    return SizingDecision(
        "buy",
        edge,
        kelly_raw,
        kelly_used,
        stake_cents,
        count,
        "edge, sizing, reserve and liquidity checks passed",
    )


class RiskLedger:
    """Portfolio-level circuit breakers: same-day loss limit and a
    peak-to-trough drawdown kill switch.

    Deliberately tracks bankroll rather than realized P&L: an account whose
    open positions have gone underwater is drawn down whether or not
    anything has settled yet, and a settlement-only measure would let the
    kill switch sleep through exactly the stretch it exists to catch.

    Thread-safe — the bots run several scan threads against one account.
    """

    def __init__(
        self,
        starting_bankroll_cents: int,
        daily_loss_limit_fraction: float = DAILY_LOSS_LIMIT_FRACTION,
        max_drawdown_fraction: float = MAX_DRAWDOWN_FRACTION,
    ):
        self._lock = threading.Lock()
        self.daily_loss_limit_fraction = daily_loss_limit_fraction
        self.max_drawdown_fraction = max_drawdown_fraction
        self.peak_bankroll_cents = max(0, starting_bankroll_cents)
        self.day_start_bankroll_cents = max(0, starting_bankroll_cents)
        self.current_bankroll_cents = max(0, starting_bankroll_cents)
        self._day = self._today()
        self._halt_reason: str | None = None
        self._hard_halt = False

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def update_bankroll(self, bankroll_cents: int) -> None:
        """Record the account's current total value and re-evaluate breakers.

        Call once per scan with balance + mark-to-market position value (or
        just balance if positions are not marked — the manual's weekly
        review is what catches the difference).
        """
        with self._lock:
            today = self._today()
            if today != self._day:
                # A new day resets the daily loss limit but never the
                # drawdown kill switch — a hard halt is not a bad day, it
                # is a broken thesis, and sleeping on it doesn't fix it.
                self._day = today
                self.day_start_bankroll_cents = bankroll_cents
                if not self._hard_halt:
                    self._halt_reason = None

            self.current_bankroll_cents = bankroll_cents
            self.peak_bankroll_cents = max(self.peak_bankroll_cents, bankroll_cents)

            if self.peak_bankroll_cents > 0:
                drawdown = 1.0 - (bankroll_cents / self.peak_bankroll_cents)
                if drawdown >= self.max_drawdown_fraction:
                    self._hard_halt = True
                    self._halt_reason = (
                        f"HARD STOP: drawdown {drawdown:.1%} from peak "
                        f"${self.peak_bankroll_cents / 100:,.2f} exceeds the "
                        f"{self.max_drawdown_fraction:.0%} limit — manual re-arm required"
                    )
                    return

            if self.day_start_bankroll_cents > 0:
                day_loss = 1.0 - (bankroll_cents / self.day_start_bankroll_cents)
                if day_loss >= self.daily_loss_limit_fraction:
                    self._halt_reason = (
                        f"halted for today: down {day_loss:.1%} from the day's "
                        f"opening ${self.day_start_bankroll_cents / 100:,.2f} "
                        f"(limit {self.daily_loss_limit_fraction:.0%})"
                    )

    def halt_reason(self) -> str | None:
        """Why trading is halted, or None if clear to trade."""
        with self._lock:
            return self._halt_reason

    def is_halted(self) -> bool:
        return self.halt_reason() is not None

    def is_hard_halted(self) -> bool:
        """True once the drawdown kill switch has tripped. Only `re_arm()`
        clears this — not a new day, not a recovery in bankroll."""
        with self._lock:
            return self._hard_halt

    def re_arm(self) -> None:
        """Clear a hard halt. Deliberately explicit: the manual (§7) requires
        root-causing before re-arming, and nothing in the bot calls this."""
        with self._lock:
            logger.warning("RiskLedger re-armed by explicit request (was: %s)", self._halt_reason)
            self._hard_halt = False
            self._halt_reason = None
            self.peak_bankroll_cents = self.current_bankroll_cents
            self.day_start_bankroll_cents = self.current_bankroll_cents

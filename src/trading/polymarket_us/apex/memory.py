"""Persistent bankroll/trade/lesson journal for the Apex bot.

Same persistence pattern as paper_client.py: JSON on disk, atomic
tmp-then-replace writes, loaded once at startup and rewritten on every
state change. Two numbers make up the tracked bankroll:

  cash_cents      — settled, uncommitted money.
  committed_cents — cost basis of currently open (unsettled) positions.

bankroll_cents = cash_cents + committed_cents is deliberately marked at cost,
not at the model's current view of fair value. Marking open positions at
model_prob * 100 would let an optimistic probability estimate inflate the
bankroll used to size the *next* bet, compounding a bad estimate into a
bigger bet before the first one has even resolved.

Three distinct "stop" states, matching the system prompt's rules plus one
this repo's own incident history (see the ladder bot's module docstring)
says is necessary for anything trading this exchange unattended:
  terminated       — bankroll hit the $1 floor, by the tracked ledger or by
                      an independent read of the real exchange balance
                      (check_external_balance). Permanent; a human has to
                      re-fund and reset the ledger by hand.
  halted (anomaly)  — something the bot can't safely reason about happened
                      (an untracked live position, a fill in the wrong
                      direction). Permanent until a human clears it by
                      hand; carrying unknown risk is worse than trading
                      blind.
  halted (daily)    — the day's drawdown hit the circuit-breaker. Clears
                      automatically at the next UTC day boundary.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_PATH = Path("data/polymarket_us/apex_memory.json")
DEFAULT_STARTING_BANKROLL_CENTS = 1000  # $10, matching the system prompt's starting stake
TERMINATION_FLOOR_CENTS = 100  # $1 — bankroll at or below this is a permanent stop
DAILY_DRAWDOWN_HALT_FRACTION = 0.15  # stop trading for the rest of the UTC day past this loss

MAX_TRADE_LOG = 1000
MAX_LESSONS = 300


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class ApexMemory:
    def __init__(
        self,
        path: Path = DEFAULT_MEMORY_PATH,
        starting_bankroll_cents: int = DEFAULT_STARTING_BANKROLL_CENTS,
        label: str = "Apex",
    ):
        self.path = path
        self._lock = threading.Lock()
        self.label = label  # this class is shared with sports_props_bot.py -- keeps log/rationale text honest about which bot it is
        self.starting_bankroll_cents = starting_bankroll_cents
        self.cash_cents = starting_bankroll_cents
        self.committed_cents = 0
        self.trade_log: list[dict] = []
        self.lessons: list[dict] = []
        self.terminated_reason: Optional[str] = None
        self.anomaly_halted_reason: Optional[str] = None
        self.daily_halted_reason: Optional[str] = None
        self.daily_start_bankroll_cents = starting_bankroll_cents
        self.daily_date = _today_utc()
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("[%s] Failed to load memory at %s, starting fresh: %s", self.label.lower(), self.path, exc)
            return
        self.starting_bankroll_cents = raw.get("starting_bankroll_cents", self.starting_bankroll_cents)
        self.cash_cents = raw.get("cash_cents", self.cash_cents)
        self.committed_cents = raw.get("committed_cents", self.committed_cents)
        self.trade_log = raw.get("trade_log", [])
        self.lessons = raw.get("lessons", [])
        self.terminated_reason = raw.get("terminated_reason")
        self.anomaly_halted_reason = raw.get("anomaly_halted_reason")
        self.daily_halted_reason = raw.get("daily_halted_reason")
        self.daily_start_bankroll_cents = raw.get("daily_start_bankroll_cents", self.bankroll_cents)
        self.daily_date = raw.get("daily_date", _today_utc())
        logger.info(
            "[%s] Loaded memory: bankroll=$%.2f  open_trades=%d  lessons=%d%s",
            self.label.lower(),
            self.bankroll_cents / 100,
            len(self.open_trades()),
            len(self.lessons),
            "  TERMINATED" if self.terminated_reason else "",
        )

    def _save_locked(self) -> None:
        """Caller must hold self._lock."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "starting_bankroll_cents": self.starting_bankroll_cents,
                    "cash_cents": self.cash_cents,
                    "committed_cents": self.committed_cents,
                    "trade_log": self.trade_log[-MAX_TRADE_LOG:],
                    "lessons": self.lessons[-MAX_LESSONS:],
                    "terminated_reason": self.terminated_reason,
                    "anomaly_halted_reason": self.anomaly_halted_reason,
                    "daily_halted_reason": self.daily_halted_reason,
                    "daily_start_bankroll_cents": self.daily_start_bankroll_cents,
                    "daily_date": self.daily_date,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)

    # ── Bankroll ─────────────────────────────────────────────────────────────

    @property
    def bankroll_cents(self) -> int:
        return self.cash_cents + self.committed_cents

    def roll_daily_window(self) -> None:
        """Reset the daily drawdown baseline if the UTC day has rolled over.
        Call at the top of every scan, before checking is_halted_today()."""
        today = _today_utc()
        with self._lock:
            if today != self.daily_date:
                self.daily_date = today
                self.daily_start_bankroll_cents = self.bankroll_cents
                self.daily_halted_reason = None
                self._save_locked()

    def daily_drawdown_fraction(self) -> float:
        """Fraction lost so far today, relative to the day's starting bankroll.
        0.0 (or negative, i.e. up on the day) is fine; only used against the
        15% circuit breaker."""
        start = self.daily_start_bankroll_cents
        if start <= 0:
            return 0.0
        return (start - self.bankroll_cents) / start

    def is_terminated(self) -> bool:
        return self.terminated_reason is not None

    def is_halted_today(self) -> bool:
        return self.daily_halted_reason is not None

    def is_halted(self) -> bool:
        return self.anomaly_halted_reason is not None

    def halt(self, reason: str) -> None:
        """Stop trading indefinitely over something the bot can't safely
        reason about on its own (an untracked live position, a fill that
        moved the wrong direction). Unlike the daily drawdown halt, this
        never clears itself -- a human has to look at the account and
        clear anomaly_halted_reason in the ledger file by hand."""
        with self._lock:
            if self.anomaly_halted_reason is None:
                self.anomaly_halted_reason = reason
                logger.error("[%s] CRITICAL: HALTED — %s", self.label.lower(), reason)
                self._save_locked()

    def clear_halt(self, note: str) -> None:
        """The programmatic form of "clear anomaly_halted_reason in the
        ledger file by hand" -- for when a human (or an agent acting on a
        human's explicit confirmation) has actually verified what triggered
        halt() is explainable, e.g. real exchange evidence showing an
        "untracked" position belongs to a sibling live bot's own legitimate
        trade rather than a genuine anomaly. Never called automatically by
        this class itself -- see halt()'s own docstring on why this never
        clears on its own."""
        with self._lock:
            if self.anomaly_halted_reason is not None:
                logger.warning(
                    "[%s] Halt cleared: %s (was: %s)",
                    self.label.lower(), note, self.anomaly_halted_reason,
                )
                self.anomaly_halted_reason = None
                self._save_locked()

    def check_external_balance(self, balance_cents: int) -> None:
        """Terminate if the REAL exchange balance is at/below the floor,
        even if the tracked ledger hasn't independently caught it. The
        tracked bankroll can drift from reality (an unbooked fee, a
        position the ledger doesn't know about) in ways that only ever
        understate risk, so this is a second, independent check against
        the one number that can't be wrong: the account's own balance.

        This account is shared with other live bots on this exchange (see
        bot.py's _reconcile_live_positions) — a low real balance does not
        always mean Apex itself lost money. A sibling bot's own spending
        can push the shared balance to the floor while Apex's own
        cash_cents is still healthy. Terminating permanently over another
        bot's dip would retire a perfectly healthy bot for a loss it never
        took, and require a human to notice and hand-edit the ledger to
        undo it. Only a floor breach that Apex's OWN tracked cash agrees
        with is treated as real and permanent; otherwise this halts for
        the day, which this check re-evaluates fresh on every scan, so it
        self-corrects the moment the shared balance recovers."""
        with self._lock:
            if balance_cents > TERMINATION_FLOOR_CENTS or self.terminated_reason is not None:
                return
            if self.cash_cents <= TERMINATION_FLOOR_CENTS:
                self.terminated_reason = (
                    f"real exchange balance ${balance_cents / 100:.2f} at or below the "
                    f"${TERMINATION_FLOOR_CENTS / 100:.2f} floor, and {self.label}'s own tracked cash "
                    f"${self.cash_cents / 100:.2f} agrees it's genuinely spent "
                    "(independent check against the live account)"
                )
                logger.error("[%s] CRITICAL: TERMINATED — %s", self.label.lower(), self.terminated_reason)
                self._save_locked()
            elif self.daily_halted_reason is None:
                self.daily_halted_reason = (
                    f"real exchange balance ${balance_cents / 100:.2f} at or below the "
                    f"${TERMINATION_FLOOR_CENTS / 100:.2f} floor, but {self.label}'s own tracked cash "
                    f"${self.cash_cents / 100:.2f} is still healthy — looks like a shared-account "
                    "dip from another live bot, not a real loss. Halting for today rather than "
                    "terminating; re-checked fresh every scan."
                )
                logger.warning("[%s] Trading halted for the rest of the day — %s", self.label.lower(), self.daily_halted_reason)
                self._save_locked()

    # ── Trade lifecycle ──────────────────────────────────────────────────────

    def record_open(
        self,
        market_slug: str,
        side: str,
        count: float,
        price_cents: int,
        cost_cents: int,
        model_prob: float,
        market_implied_prob: float,
        edge_pct: float,
        kelly_fraction_used: float,
        rationale: str = "",
    ) -> str:
        trade_id = uuid.uuid4().hex[:12]
        with self._lock:
            self.cash_cents -= cost_cents
            self.committed_cents += cost_cents
            self.trade_log.append(
                {
                    "id": trade_id,
                    "opened_ts": datetime.now(timezone.utc).isoformat(),
                    "market_slug": market_slug,
                    "side": side,
                    "count": count,
                    "price_cents": price_cents,
                    "cost_cents": cost_cents,
                    "model_prob": model_prob,
                    "market_implied_prob": market_implied_prob,
                    "edge_pct": edge_pct,
                    "kelly_fraction_used": kelly_fraction_used,
                    "rationale": rationale,
                    "status": "open",
                }
            )
            self._save_locked()
        return trade_id

    def record_manual_open(
        self,
        market_slug: str,
        side: str,
        count: float,
        price_cents: int,
        cost_cents: int,
        note: str = "",
    ) -> str:
        """A position opened OUTSIDE the bot — a human bought directly
        through the exchange's own interface. Mirror image of
        record_manual_close(): _reconcile_live_positions() finds a live
        position with no matching ledger entry and halts until a human
        confirms what it was and calls this to reconcile it (see that
        method's own real incident for why this can't just be silently
        auto-tracked). model_prob is recorded as 0.5 (neutral) rather than
        left out entirely — record_settlement()'s miss-lesson logic reads
        trade["model_prob"] unconditionally, and 0.5 means "no real
        estimate exists" without crashing or fabricating a number Apex
        never computed. First real incident: 2026-08-16, two Chicago
        temperature buckets bought manually (confirmed via the exchange's
        own manualOrderIndicator field) while Apex already held a third
        bucket for the same city-day — something only a human could do,
        since Apex's own one-bucket-per-city-day guard would never allow it."""
        trade_id = uuid.uuid4().hex[:12]
        with self._lock:
            self.cash_cents -= cost_cents
            self.committed_cents += cost_cents
            self.trade_log.append(
                {
                    "id": trade_id,
                    "opened_ts": datetime.now(timezone.utc).isoformat(),
                    "market_slug": market_slug,
                    "side": side,
                    "count": count,
                    "price_cents": price_cents,
                    "cost_cents": cost_cents,
                    "model_prob": 0.5,
                    "market_implied_prob": price_cents / 100.0,
                    "edge_pct": 0.0,
                    "kelly_fraction_used": 0.0,
                    "rationale": f"Manually placed by the user, not an {self.label} decision.{(' ' + note) if note else ''}",
                    "status": "open",
                    "manual": True,
                }
            )
            self._save_locked()
        return trade_id

    def open_trades(self) -> list[dict]:
        return [t for t in self.trade_log if t["status"] == "open"]

    def record_settlement(self, trade_id: str, outcome_yes: bool, payout_cents: int) -> Optional[int]:
        """Close out an open trade against its real resolution. Returns the
        realized P&L in cents, or None if trade_id wasn't found/already closed.

        Checks the termination floor and daily drawdown circuit breaker as
        part of the same locked update — both read the post-settlement
        bankroll, so they can never observe a stale mid-update value.
        """
        with self._lock:
            trade = next((t for t in self.trade_log if t["id"] == trade_id and t["status"] == "open"), None)
            if trade is None:
                return None
            cost_cents = trade["cost_cents"]
            self.committed_cents -= cost_cents
            self.cash_cents += payout_cents
            pnl_cents = payout_cents - cost_cents

            trade["status"] = "settled"
            trade["settled_ts"] = datetime.now(timezone.utc).isoformat()
            trade["outcome_yes"] = outcome_yes
            trade["payout_cents"] = payout_cents
            trade["pnl_cents"] = pnl_cents

            self._check_floor_and_drawdown_locked(f"settlement of {trade['market_slug']}")
            self._save_locked()
        return pnl_cents

    def record_manual_close(self, trade_id: str, proceeds_cents: int) -> Optional[int]:
        """Close out an open trade that was closed OUTSIDE the bot — a human
        sold the position directly through the exchange's own interface.
        Apex has no sell/flatten logic of its own; _reconcile_live_positions()
        only ever notices this after the fact (memory still shows the trade
        open, but the live account no longer holds it). Deliberately distinct
        from record_settlement(): the market hasn't actually resolved, so
        there is no real outcome_yes to record — forcing one in here would
        misrepresent a manual liquidation as a market settlement in any
        future win-rate read of this log. First real incident: 2026-08-13,
        the Miami temperature bucket bought automatically at 8c, sold
        manually at 7c about an hour later."""
        with self._lock:
            trade = next((t for t in self.trade_log if t["id"] == trade_id and t["status"] == "open"), None)
            if trade is None:
                return None
            cost_cents = trade["cost_cents"]
            self.committed_cents -= cost_cents
            self.cash_cents += proceeds_cents
            pnl_cents = proceeds_cents - cost_cents

            trade["status"] = "closed_manually"
            trade["closed_ts"] = datetime.now(timezone.utc).isoformat()
            trade["proceeds_cents"] = proceeds_cents
            trade["pnl_cents"] = pnl_cents

            self._check_floor_and_drawdown_locked(f"manual close of {trade['market_slug']}")
            self._save_locked()
        return pnl_cents

    def _check_floor_and_drawdown_locked(self, trigger_desc: str) -> None:
        """Caller must hold self._lock. Shared by record_settlement() and
        record_manual_close() — both change the bankroll and both need the
        same two circuit breakers re-checked against the fresh total."""
        bankroll = self.bankroll_cents
        if bankroll <= TERMINATION_FLOOR_CENTS and self.terminated_reason is None:
            self.terminated_reason = (
                f"bankroll ${bankroll / 100:.2f} at or below the "
                f"${TERMINATION_FLOOR_CENTS / 100:.2f} floor as of {trigger_desc}"
            )
            logger.error("[%s] CRITICAL: TERMINATED — %s", self.label.lower(), self.terminated_reason)

        drawdown = self.daily_drawdown_fraction()
        if drawdown > DAILY_DRAWDOWN_HALT_FRACTION and self.daily_halted_reason is None:
            self.daily_halted_reason = (
                f"daily drawdown {drawdown * 100:.1f}% exceeds the "
                f"{DAILY_DRAWDOWN_HALT_FRACTION * 100:.0f}% circuit breaker"
            )
            logger.warning("[%s] Trading halted for the rest of the day — %s", self.label.lower(), self.daily_halted_reason)

    # ── Lessons / session recall ────────────────────────────────────────────

    def add_lesson(self, text: str) -> None:
        with self._lock:
            self.lessons.append({"ts": datetime.now(timezone.utc).isoformat(), "text": text})
            self._save_locked()

    def recent_lessons(self, n: int = 5) -> list[str]:
        return [entry["text"] for entry in self.lessons[-n:]]

    def session_banner(self) -> str:
        """"Begin every new session by stating current bankroll and summarizing
        the previous day's key lessons" — rendered from real ledger state, not
        an LLM's recollection of one."""
        open_ct = len(self.open_trades())
        settled = [t for t in self.trade_log if t["status"] == "settled"]
        realized_pnl = sum(t.get("pnl_cents", 0) for t in settled)
        lines = [
            f"{self.label} session start — bankroll ${self.bankroll_cents / 100:.2f} "
            f"(started at ${self.starting_bankroll_cents / 100:.2f}, "
            f"realized P&L ${realized_pnl / 100:+.2f} over {len(settled)} settled trade(s), "
            f"{open_ct} open).",
        ]
        if self.terminated_reason:
            lines.append(f"TERMINATED: {self.terminated_reason}")
        if self.is_halted():
            lines.append(f"HALTED (needs manual review): {self.anomaly_halted_reason}")
        if self.is_halted_today():
            lines.append(f"Halted for today: {self.daily_halted_reason}")
        lessons = self.recent_lessons()
        if lessons:
            lines.append("Recent lessons:")
            lines.extend(f"  - {text}" for text in lessons)
        else:
            lines.append("No lessons logged yet.")
        return "\n".join(lines)

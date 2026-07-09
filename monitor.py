"""Continuous health monitor for the Kalshi trading bots.

Watches three things on a 60-second cycle:

  1. Win-rate model  — loaded? enough price points? consistently zero opps?
  2. Scan / position — positions approaching cap? slots exhausted? balance drop?
  3. Learning loop   — pending bets, buckets building vs adjusting, P&L trend

Logs to monitor.log and stdout. Run alongside run_all.py.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("monitor.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("monitor")

# ── Config ────────────────────────────────────────────────────────────────────

BOT_LOG       = Path("bot_err.log")
LEARNING_FILE = Path("data/kalshi/bot_learning.json")

CHECK_INTERVAL     = 60    # seconds between log-tail passes
LEARNING_INTERVAL  = 600   # seconds between learning-file checks (10 min)
WIN_RATE_MIN_PTS   = 20    # warn if fewer price points loaded than this
ZERO_OPPS_WARN     = 5     # warn after this many consecutive zero-opp scans
BALANCE_DROP_PCT   = 5.0   # warn if balance drops more than this % in one scan
MAX_ERRORS_PER_HR  = 5     # suppress error spam beyond this
MIN_BETS_BUCKET    = 10    # matches learning.py MIN_BETS_FOR_ADJUSTMENT

# ── Patterns ──────────────────────────────────────────────────────────────────

_PAT_WIN_RATES = re.compile(r"Win rates loaded: (\d+) price")
_PAT_SCAN      = re.compile(
    r"\[(\d{2}:\d{2}:\d{2})\] balance=\$([\d.]+) \(budget=\$([\d.]+)\)"
    r"  markets=(\d+)  opps=(\d+)  positions=(\d+)  slots=(\d+)"
)
_PAT_BET       = re.compile(
    r"Placed (YES|NO) on (\S+): (\d+)x @ (\d+)¢.*edge=([\d.]+)pp"
)
_PAT_LOCK      = re.compile(r"LOCK on (\S+):.*profit=\+(\d+)¢ guaranteed")
_PAT_SETTLE    = re.compile(
    r"Learning: (\S+) (YES|NO)@(\d+)¢ → (WIN|LOSS)  P&L=([+-]\d+)¢"
)
_PAT_ERROR     = re.compile(r"\bERROR\b")
# Convergence bot
_PAT_CONV_BET  = re.compile(
    r"\[convergence\] BET (YES|NO) on (\S+): (\d+)x@(\d+)¢\s+"
    r"prob=([\d.]+)%\s+edge=\+([\d.]+)pp\s+(\w+)=\$([\d,.]+)"
)
_PAT_CONV_NOSIG = re.compile(r"\[convergence\] No signals")
_PAT_CONV_SIG   = re.compile(r"\[convergence\] (\d+) signal")

# ── Mutable state ─────────────────────────────────────────────────────────────

CONV_NO_PRICE_WARN = 10   # warn after this many consecutive no-signal scans

class _S:
    log_pos:               int   = 0
    win_rates_loaded:      bool  = False
    win_rate_points:       int   = 0
    consecutive_zero_opps: int   = 0
    last_balance:          float | None = None
    errors_this_hour:      int   = 0
    error_hour_start:      float = time.monotonic()
    last_learning_check:   float = 0.0
    session_pnl_cents:     int   = 0
    # Convergence bot
    conv_bets:             int   = 0
    conv_no_signal_streak: int   = 0

S = _S()

# ── Log processing ────────────────────────────────────────────────────────────

def _process_lines(lines: list[str]) -> None:
    for line in lines:

        # 1 — Win-rate model loaded
        m = _PAT_WIN_RATES.search(line)
        if m:
            n = int(m.group(1))
            S.win_rates_loaded = True
            S.win_rate_points  = n
            S.consecutive_zero_opps = 0
            if n < WIN_RATE_MIN_PTS:
                log.warning(
                    "[WIN-RATE] Loaded only %d price points — too sparse, bots may not trade", n
                )
            else:
                log.info("[WIN-RATE] Loaded %d price points — bots are ready to trade", n)
            continue

        # 2 — Scan line (balance / opps / positions / slots)
        m = _PAT_SCAN.search(line)
        if m:
            ts, bal, budget, markets, opps, positions, slots = m.groups()
            bal, budget = float(bal), float(budget)
            opps, positions, slots = int(opps), int(positions), int(slots)

            # Balance drop alert
            if S.last_balance is not None:
                drop_pct = 100.0 * (S.last_balance - bal) / S.last_balance
                if drop_pct >= BALANCE_DROP_PCT:
                    log.warning(
                        "[BALANCE] Drop of %.1f%%: $%.2f → $%.2f",
                        drop_pct, S.last_balance, bal,
                    )
            S.last_balance = bal

            # Position cap alert
            if slots == 0 and opps > 0:
                log.warning(
                    "[POSITIONS] Cap hit — %d missed opportunities (positions full, slots=0)",
                    opps,
                )
            elif slots == 0:
                log.info("[POSITIONS] Cap hit — slots=0, no new bets until positions settle")

            # Zero-opps streak (only meaningful after win rates loaded)
            if S.win_rates_loaded:
                if opps == 0:
                    S.consecutive_zero_opps += 1
                    if S.consecutive_zero_opps == ZERO_OPPS_WARN:
                        log.warning(
                            "[OPPS] %d consecutive scans with zero opportunities "
                            "(win-rate pts=%d, markets=%d) — "
                            "edge threshold may be too tight or price range too narrow",
                            S.consecutive_zero_opps, S.win_rate_points, markets,
                        )
                else:
                    if S.consecutive_zero_opps >= ZERO_OPPS_WARN:
                        log.info(
                            "[OPPS] Opportunities found again after %d-scan drought",
                            S.consecutive_zero_opps,
                        )
                    S.consecutive_zero_opps = 0
            continue

        # 3 — Bet placed
        m = _PAT_BET.search(line)
        if m:
            side, ticker, qty, price, edge = m.groups()
            log.info("[BET] %s  %s  %sx@%s¢  edge=+%spp", side, ticker, qty, price, edge)
            continue

        # 4 — Lock placed
        m = _PAT_LOCK.search(line)
        if m:
            ticker, profit = m.groups()
            log.info("[LOCK] %s  +%s¢ guaranteed profit", ticker, profit)
            continue

        # 5 — Bet settled
        m = _PAT_SETTLE.search(line)
        if m:
            ticker, side, price, result, pnl = m.groups()
            S.session_pnl_cents += int(pnl)
            log.info(
                "[SETTLED] %s %s@%s¢ → %s  P&L=%s¢  (session total: %+d¢)",
                ticker, side, price, result, pnl, S.session_pnl_cents,
            )
            continue

        # 6 — Convergence bot bet
        m = _PAT_CONV_BET.search(line)
        if m:
            side, ticker, qty, price, prob, edge, coin, spot = m.groups()
            S.conv_bets += 1
            S.conv_no_signal_streak = 0
            log.info(
                "[CONVERGENCE] BET %s on %s: %sx@%s¢  prob=%s%%  edge=+%spp  "
                "%s=$%s  (session bets: %d)",
                side, ticker, qty, price, prob, edge, coin, spot, S.conv_bets,
            )
            continue

        # 6a — Convergence no-signal scan
        if _PAT_CONV_NOSIG.search(line):
            S.conv_no_signal_streak += 1
            if S.conv_no_signal_streak == CONV_NO_PRICE_WARN:
                log.warning(
                    "[CONVERGENCE] %d consecutive scans with no signal — "
                    "check live price feed or widen window parameters",
                    S.conv_no_signal_streak,
                )
            continue

        # 6b — Convergence found signals (reset streak)
        if _PAT_CONV_SIG.search(line):
            S.conv_no_signal_streak = 0
            continue

        # 7 — Error lines (rate-limited)
        if _PAT_ERROR.search(line):
            now = time.monotonic()
            if now - S.error_hour_start > 3600:
                S.errors_this_hour = 0
                S.error_hour_start = now
            S.errors_this_hour += 1
            if S.errors_this_hour <= MAX_ERRORS_PER_HR:
                log.error("[BOT ERROR] %s", line[20:160].strip())
            elif S.errors_this_hour == MAX_ERRORS_PER_HR + 1:
                log.error("[BOT ERROR] Suppressing further errors this hour (>%d)", MAX_ERRORS_PER_HR)


# ── Learning checker ──────────────────────────────────────────────────────────

def _check_learning() -> None:
    if not LEARNING_FILE.exists():
        log.debug("[LEARNING] File not found yet")
        return
    try:
        data = json.loads(LEARNING_FILE.read_text())
    except Exception as exc:
        log.warning("[LEARNING] Could not read file: %s", exc)
        return

    if not isinstance(data, dict):
        log.warning("[LEARNING] Unexpected content in %s (expected a JSON object)", LEARNING_FILE)
        return

    bets    = data.get("bets", [])
    pending = [b for b in bets if not b.get("result")]
    settled = [b for b in bets if b.get("result")]

    if not bets:
        log.info("[LEARNING] No bets recorded yet")
        return

    # Per-bucket aggregation
    buckets: dict[tuple, dict] = {}
    for b in settled:
        try:
            key = (b["series"], b["side"], (b["entry_price"] // 10) * 10)
        except (KeyError, TypeError):
            continue  # malformed/partial bet record — skip rather than crash the check
        buckets.setdefault(key, {"wins": 0, "total": 0, "preds": []})
        buckets[key]["total"] += 1
        if b["result"] == "win":
            buckets[key]["wins"] += 1
        if b.get("predicted_win_rate") is not None:
            buckets[key]["preds"].append(b["predicted_win_rate"])

    adjusting = {k: v for k, v in buckets.items() if v["total"] >= MIN_BETS_BUCKET}
    building  = {k: v for k, v in buckets.items() if v["total"] <  MIN_BETS_BUCKET}

    # Overall P&L
    total_pnl = sum(b.get("profit_cents") or 0 for b in settled)
    wins = sum(1 for b in settled if b.get("result") == "win")
    wr   = 100 * wins / len(settled) if settled else 0.0

    lines = [
        f"[LEARNING] {len(bets)} total  {len(pending)} pending  {len(settled)} settled"
        f"  win-rate={wr:.0f}%  P&L={total_pnl:+d}¢ (${total_pnl/100:+.2f})"
        f"  |  {len(adjusting)} buckets adjusting  {len(building)} buckets building"
    ]

    # Adjusting buckets — show their actual adjustment
    for (series, side, bucket), v in sorted(adjusting.items()):
        actual_wr  = 100 * v["wins"] / v["total"]
        avg_pred   = sum(v["preds"]) / len(v["preds"]) if v["preds"] else 0.0
        adj        = round(max(0.0, min(6.0, (avg_pred - actual_wr) * 0.5)), 2)
        status     = f"adj=+{adj:.1f}pp" if adj > 0 else "adj=none (on-target)"
        lines.append(
            f"  ADJUSTING [{series} {side.upper()} {bucket}-{bucket+9}¢]"
            f"  n={v['total']}  actual={actual_wr:.0f}%  pred={avg_pred:.0f}%  {status}"
        )

    # Building buckets — top 5 closest to threshold
    for (series, side, bucket), v in sorted(building.items(), key=lambda x: -x[1]["total"])[:5]:
        lines.append(
            f"  BUILDING  [{series} {side.upper()} {bucket}-{bucket+9}¢]"
            f"  n={v['total']}/{MIN_BETS_BUCKET}"
        )

    # Warn if many bets are stuck as pending for too long
    if len(pending) > 20:
        log.warning(
            "[LEARNING] %d bets still pending — check if settlement check is working",
            len(pending),
        )

    log.info("\n".join(lines))


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 70)
    log.info("Monitor started")
    log.info("  Watching : %s", BOT_LOG)
    log.info("  Learning : %s", LEARNING_FILE)
    log.info("  Alerts   : win-rate load, zero-opps streak (%d), "
             "balance drop (%.0f%%), position cap, errors", ZERO_OPPS_WARN, BALANCE_DROP_PCT)
    log.info("  Cycle    : log-tail every %ds, learning check every %ds",
             CHECK_INTERVAL, LEARNING_INTERVAL)
    log.info("=" * 70)

    while True:
        try:
            # ── Tail bot log ──────────────────────────────────────────────────
            if BOT_LOG.exists():
                size = BOT_LOG.stat().st_size
                if size != S.log_pos:
                    if size < S.log_pos:
                        S.log_pos = 0          # log was rotated
                    with open(BOT_LOG, encoding="utf-8", errors="ignore") as f:
                        f.seek(S.log_pos)
                        new_text = f.read()
                    S.log_pos = BOT_LOG.stat().st_size
                    lines = new_text.splitlines()
                    if lines:
                        _process_lines(lines)
            else:
                if not S.win_rates_loaded:
                    log.debug("[MONITOR] bot_err.log not found yet — bots may not have started")

            # ── Learning check (every 10 min) ─────────────────────────────────
            now = time.monotonic()
            if now - S.last_learning_check >= LEARNING_INTERVAL:
                _check_learning()
                S.last_learning_check = now

        except Exception as exc:
            log.warning("[MONITOR] Loop error: %s", exc)

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()

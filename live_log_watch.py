"""Lightweight live log watcher for Kalshi and Polymarket bots.

Watches the live bot log files and writes a compact health stream to
live_watch.log. It is intentionally dependency-free so it can keep running even
when uv/the virtualenv is fussy.

Usage:
    python live_log_watch.py
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

WATCHED_LOGS = {
    "kalshi_err": Path("bot_err.log"),
    "kalshi_out": Path("bot_out.log"),
    "poly_err": Path("poly_err.log"),
    "poly_out": Path("poly_out.log"),
}

STATE_FILE = Path("data/.live_log_watch_state.json")
WATCH_LOG = Path("live_watch.log")
CHECK_INTERVAL_SECONDS = 15
QUIET_WARN_SECONDS = 300


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(WATCH_LOG, encoding="utf-8"),
    ],
)
log = logging.getLogger("live_log_watch")


@dataclass
class LogCursor:
    path: str
    offset: int = 0
    inode_hint: float = 0.0


@dataclass
class WatchStats:
    started_at: str
    last_line_at: dict[str, str]
    alerts: int = 0
    errors: int = 0
    warnings: int = 0
    orders: int = 0
    scans: int = 0
    no_data_warnings: int = 0
    auth_errors: int = 0


PATTERNS = {
    "order": re.compile(r"\b(Placed|BET|LOCK|ARB: Kalshi leg placed)\b", re.I),
    "scan": re.compile(r"\bbalance=\$|Arb scan:|Reloading win rates|Started\b", re.I),
    "error": re.compile(r"\b(ERROR|Traceback|Exception in thread|CRITICAL|failed)\b", re.I),
    "warning": re.compile(r"\b(WARNING|WARN)\b", re.I),
    "auth": re.compile(r"\b(auth|api key|private key|signature|unauthorized|forbidden|401|403)\b", re.I),
    "no_data": re.compile(r"No historical data yet|run the indexer first|win rates.*not", re.I),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state() -> tuple[dict[str, LogCursor], WatchStats]:
    if STATE_FILE.exists():
        try:
            raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            cursors = {name: LogCursor(**data) for name, data in raw.get("cursors", {}).items()}
            stats = WatchStats(**raw["stats"])
            for name, path in WATCHED_LOGS.items():
                cursors.setdefault(name, LogCursor(path=str(path)))
            return cursors, stats
        except Exception:
            pass

    cursors = {name: LogCursor(path=str(path)) for name, path in WATCHED_LOGS.items()}
    stats = WatchStats(started_at=_now(), last_line_at={})
    return cursors, stats


def save_state(cursors: dict[str, LogCursor], stats: WatchStats) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cursors": {name: asdict(cursor) for name, cursor in cursors.items()},
        "stats": asdict(stats),
    }
    STATE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_new_lines(cursor: LogCursor) -> list[str]:
    path = Path(cursor.path)
    if not path.exists():
        return []

    size = path.stat().st_size
    if size < cursor.offset:
        cursor.offset = 0

    with path.open("rb") as fh:
        fh.seek(cursor.offset)
        chunk = fh.read()
        cursor.offset = fh.tell()

    if not chunk:
        return []

    text = chunk.decode("utf-8", errors="ignore")
    return [line.strip("\x00\r\n") for line in text.splitlines() if line.strip("\x00\r\n")]


def classify_line(source: str, line: str, stats: WatchStats) -> None:
    if PATTERNS["order"].search(line):
        stats.orders += 1
        log.info("[%s] ORDER/SIGNAL: %s", source, line[-240:])
        return

    if PATTERNS["auth"].search(line) and PATTERNS["error"].search(line):
        stats.auth_errors += 1
        stats.alerts += 1
        log.error("[%s] AUTH ALERT: %s", source, line[-240:])
        return

    if PATTERNS["no_data"].search(line):
        stats.no_data_warnings += 1
        stats.warnings += 1
        log.warning("[%s] DATA WARNING: %s", source, line[-240:])
        return

    if PATTERNS["error"].search(line):
        stats.errors += 1
        stats.alerts += 1
        log.error("[%s] ERROR: %s", source, line[-240:])
        return

    if PATTERNS["warning"].search(line):
        stats.warnings += 1
        log.warning("[%s] WARN: %s", source, line[-240:])
        return

    if PATTERNS["scan"].search(line):
        stats.scans += 1
        log.info("[%s] LIVE: %s", source, line[-220:])


def quiet_check(stats: WatchStats) -> None:
    now = datetime.now(timezone.utc)
    for name in WATCHED_LOGS:
        stamp = stats.last_line_at.get(name)
        if not stamp:
            continue
        try:
            last = datetime.fromisoformat(stamp)
        except ValueError:
            continue
        quiet_for = (now - last).total_seconds()
        if quiet_for >= QUIET_WARN_SECONDS:
            stats.alerts += 1
            log.warning("[%s] QUIET: no new log lines for %.0f seconds", name, quiet_for)


def heartbeat(stats: WatchStats) -> None:
    log.info(
        "HEARTBEAT scans=%d orders=%d warnings=%d errors=%d auth_errors=%d no_data=%d alerts=%d",
        stats.scans,
        stats.orders,
        stats.warnings,
        stats.errors,
        stats.auth_errors,
        stats.no_data_warnings,
        stats.alerts,
    )


def main() -> None:
    cursors, stats = load_state()
    log.info("Live log watcher started. Watching: %s", ", ".join(WATCHED_LOGS))
    next_heartbeat = time.monotonic()
    next_quiet_check = time.monotonic() + QUIET_WARN_SECONDS

    while True:
        try:
            for name, cursor in cursors.items():
                lines = read_new_lines(cursor)
                if lines:
                    stats.last_line_at[name] = _now()
                for line in lines:
                    classify_line(name, line, stats)

            now = time.monotonic()
            if now >= next_heartbeat:
                heartbeat(stats)
                next_heartbeat = now + 60
            if now >= next_quiet_check:
                quiet_check(stats)
                next_quiet_check = now + 60

            save_state(cursors, stats)
        except Exception as exc:
            log.warning("[LIVE_LOG_WATCH] Loop error: %s", exc)

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()

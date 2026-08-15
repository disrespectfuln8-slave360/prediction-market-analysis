"""Read-only quote+outcome logger: polymarket.us MLB/NFL player-prop markets.

Weather has NOAA as an independent forecast to calibrate against; nothing
equivalent exists for "will this batter get 3+ total bases tonight" --
building a real projection model from player/matchup stats is a much
bigger, separate undertaking (see the 2026-08-14 discussion that led to
this file). The only calibration this platform can support right now is
price-level: "when this market has historically priced a bucket at X,
how often did it resolve YES?" -- the same shape as Kalshi's win-rate
table, not Apex's NOAA-grounded one. That needs both halves of history —
what the market quoted, and what actually happened — logged here since
neither exists anywhere else. This can only start accumulating going
forward, the same way weather_quote_logger.py did before Apex could be
calibrated.

Bucket parsing
--------------
astatc-{sport}-{matchup}-{YYYY-MM-DD}-{stat}-{player}-gte{N}
e.g. astatc-mlb-col-sf-2026-08-14-outs-kylfre-gte16
     -> sport=mlb, stat=outs, player=kylfre, threshold=16

Deliberately keyed on (sport, stat), never pooled across stat types or
sports. Total bases, strikeouts, home runs, touchdowns are different
distributions with different base rates; mixing them into one calibration
table would reproduce the exact mismatched-model problem found in
Kalshi's tennis-falls-back-to-global-rates bug (see MEMORY / that
session's postmortem) -- just one level finer-grained.

Two output streams, meant to be joined later by slug for calibration:
    data/polymarket_us/sports_prop_quotes.jsonl      -- one row per bucket per poll
    data/polymarket_us/sports_prop_settlements.jsonl -- one row per resolved bucket

READ-ONLY: only list_all_markets/get_bbo/get_settlement are ever called.
Never places, cancels, or otherwise touches an order -- shares a client
and account with the live ladder bot, which trades a completely separate
strategy on a completely separate slug family (astatc-* buckets the
ladder bot itself locks are fine to also see logged here; this module
takes no position of its own regardless).
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.trading.polymarket_us.paper_client import _settlement_to_cents

logger = logging.getLogger(__name__)

SPORTS = ("mlb", "nfl")
POLL_INTERVAL_SECS = 300
QUOTES_PATH = Path("data/polymarket_us/sports_prop_quotes.jsonl")
SETTLEMENTS_PATH = Path("data/polymarket_us/sports_prop_settlements.jsonl")

# A bucket that closes but never resolves within this many cycles is more
# likely a dead/cancelled market than a slow settlement -- drop it rather
# than retrying forever and growing _pending_settlement without bound.
MAX_PENDING_CYCLES = 4 * 24 * 14  # ~14 days at the default 5-minute poll

_SLUG_RE = re.compile(
    r"^astatc-(?P<sport>[a-z]+)-(?P<matchup>.+?)-(?P<date>\d{4}-\d{2}-\d{2})-"
    r"(?P<stat>[a-z]+)-(?P<player>[a-z0-9]+)-gte(?P<threshold>\d+)$"
)


def parse_prop_slug(slug: str) -> Optional[dict]:
    """(sport, matchup, date, stat, player, threshold) or None if `slug`
    isn't a player-prop threshold ladder in the family this logger tracks
    (e.g. astatc-fwc-* World Cup markets don't have this shape at all and
    are silently skipped, not an error)."""
    m = _SLUG_RE.match(slug)
    if not m:
        return None
    if m.group("sport") not in SPORTS:
        return None
    return {
        "sport": m.group("sport"),
        "matchup": m.group("matchup"),
        "date": m.group("date"),
        "stat": m.group("stat"),
        "player": m.group("player"),
        "threshold": int(m.group("threshold")),
    }


class SportsPropQuoteLogger:
    def __init__(self, client):
        self.client = client  # PolymarketUSTradingClient (market data only)
        self._active: dict[str, dict] = {}  # slug -> parsed metadata
        self._pending_settlement: dict[str, tuple[dict, int]] = {}  # slug -> (metadata, cycles waited)

    # ── Discovery ────────────────────────────────────────────────────────────

    def _discover(self) -> None:
        markets = self.client.list_all_markets(active=True, closed=False)
        fresh: dict[str, dict] = {}
        for m in markets:
            if not m.slug.startswith("astatc-"):
                continue
            meta = parse_prop_slug(m.slug)
            if meta is not None:
                fresh[m.slug] = meta

        # Anything tracked last cycle but missing now has closed (or was
        # delisted) since the last scan -- move it to the settlement queue
        # rather than just dropping it, so its eventual outcome still gets
        # captured.
        closed_slugs = set(self._active) - set(fresh)
        for slug in closed_slugs:
            self._pending_settlement[slug] = (self._active[slug], 0)

        newly_seen = set(fresh) - set(self._active)
        self._active = fresh
        logger.info(
            "Discovery: %d active buckets (%d new, %d closed since last scan), %d pending settlement",
            len(fresh), len(newly_seen), len(closed_slugs), len(self._pending_settlement),
        )

    # ── Output ───────────────────────────────────────────────────────────────

    def _append(self, path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    # ── Measurement cycle ────────────────────────────────────────────────────

    def _log_quotes(self) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        logged = 0
        for slug, meta in self._active.items():
            try:
                md = self.client.get_bbo(slug).get("marketData", {})
            except Exception as exc:
                logger.debug("BBO failed for %s: %s", slug, exc)
                continue

            def cents(key, data=md):
                v = (data.get(key) or {}).get("value")
                return None if v is None else round(float(v) * 100)

            self._append(QUOTES_PATH, {
                "ts": now_iso, "slug": slug, **meta,
                "bid": cents("bestBid"), "ask": cents("bestAsk"),
                "bid_depth": md.get("bidDepth"), "ask_depth": md.get("askDepth"),
            })
            logged += 1
        logger.info("Logged quotes for %d/%d active buckets", logged, len(self._active))

    def _check_settlements(self) -> None:
        if not self._pending_settlement:
            return
        resolved, still_pending, dropped = 0, {}, 0
        for slug, (meta, cycles_waited) in self._pending_settlement.items():
            try:
                settlement = self.client.get_settlement(slug)
            except Exception:
                if cycles_waited + 1 >= MAX_PENDING_CYCLES:
                    logger.warning(
                        "Giving up on settlement for %s after %d cycles unresolved -- "
                        "likely a dead/cancelled market, not a slow settlement",
                        slug, cycles_waited + 1,
                    )
                    dropped += 1
                    continue
                still_pending[slug] = (meta, cycles_waited + 1)
                continue

            settlement_cents = _settlement_to_cents(settlement)
            if settlement_cents is None:
                still_pending[slug] = (meta, cycles_waited + 1)
                continue

            self._append(SETTLEMENTS_PATH, {
                "resolved_ts": datetime.now(timezone.utc).isoformat(),
                "slug": slug, **meta,
                "outcome_yes": settlement_cents >= 50,
            })
            resolved += 1
        self._pending_settlement = still_pending
        if resolved or dropped:
            logger.info("Settlements: %d resolved, %d dropped (unresolved too long), %d still pending",
                        resolved, dropped, len(still_pending))

    def run_cycle(self) -> None:
        self._discover()
        self._log_quotes()
        self._check_settlements()

    def run_forever(self, stop_event=None) -> None:
        logger.info(
            "Sports-prop quote logger started — READ-ONLY, no orders will be placed. "
            "Sports: %s  poll=%ds", SPORTS, POLL_INTERVAL_SECS,
        )
        while True:
            try:
                self.run_cycle()
            except Exception as exc:
                logger.error("Cycle error: %s", exc, exc_info=True)
            if stop_event is not None:
                if stop_event.wait(timeout=POLL_INTERVAL_SECS):
                    break
            else:
                time.sleep(POLL_INTERVAL_SECS)


def main():
    from logging.handlers import RotatingFileHandler

    from dotenv import load_dotenv
    load_dotenv()

    log_dir = Path(__file__).resolve().parents[3] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_dir / "sports_prop_quote_logger.log", maxBytes=5 * 1024 * 1024, backupCount=2,
        encoding="utf-8",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(name)-20s]  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[handler, logging.StreamHandler()],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    from src.common.single_instance import acquire_single_instance_lock
    _lock = acquire_single_instance_lock("sports_prop_quote_logger")  # noqa: F841

    from src.trading.polymarket_us.trading_client import PolymarketUSTradingClient
    client = PolymarketUSTradingClient()
    try:
        SportsPropQuoteLogger(client).run_forever()
    finally:
        client.close()


if __name__ == "__main__":
    main()

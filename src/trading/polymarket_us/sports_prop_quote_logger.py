"""Read-only quote+outcome logger: polymarket.us MLB player-prop markets.

NFL split out into its own dedicated nfl_prop_quote_logger.py on 2026-08-17,
per explicit user request -- see that module's docstring for the real
market-shape survey behind the split (NFL's astatc-nfl- family turned out to
span three genuinely different bet shapes, only one of which is tracked at
all). SPORTS below stays a tuple (not a bare "mlb" str) so a future sport
sharing MLB's exact per-game-matchup-ladder shape can still be added here
directly, the way NFL originally was.

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

Settlement verification (added 2026-08-18)
--------------------------------------------
Every settlement is cross-checked against the real MLB Stats API before
being written -- see mlb_stats_verification.py's own docstring for why:
polymarket.us's own settlement determination was confirmed live to have a
real, ~14% one-directional error rate (players who did not play, or played
and genuinely fell short, still settled YES). A disagreement uses the
verified value and logs a loud warning; an inconclusive check (game/player
not resolvable) falls back to trusting the exchange, same as before this
existed.

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

from src.trading.polymarket_us import mlb_stats_verification
from src.trading.polymarket_us.paper_client import _settlement_to_cents

logger = logging.getLogger(__name__)

SPORTS = ("mlb",)
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
    def __init__(self, client, verify_settlement_fn=mlb_stats_verification.verify_settlement):
        self.client = client  # PolymarketUSTradingClient (market data only)
        # Injectable so tests can substitute a fake without real network
        # calls; defaults to the real MLB Stats API cross-check (see that
        # module's docstring for why this exists -- polymarket.us's own
        # settlements were confirmed to have a real, ~14% one-directional
        # error rate live 2026-08-18).
        self._verify_settlement_fn = verify_settlement_fn
        self._schedule_cache: dict = {}
        self._boxscore_cache: dict = {}
        self._active: dict[str, dict] = {}  # slug -> parsed metadata
        self._pending_settlement: dict[str, tuple[dict, int]] = {}  # slug -> (metadata, cycles waited)
        # sports_prop_calibration.py's _build_calibration_table only ever
        # reads each slug's FIRST recorded ask and FIRST recorded bid --
        # every quote after that is logged but never used for anything.
        # Left unbounded, this grew to 60MB/228k lines in one day (confirmed
        # live 2026-08-15) and re-parsing that on every calibration reload
        # (every ~5 min) was the single largest memory consumer on the whole
        # machine. Stop fetching/logging a slug at all once both are known.
        self._have_ask: set[str] = set()
        self._have_bid: set[str] = set()
        self._load_captured_state()

    def _load_captured_state(self) -> None:
        """Seed _have_ask/_have_bid from whatever's already on disk, so a
        restart (crash, redeploy) doesn't forget which slugs are already
        fully captured and start re-logging them -- the exact regrowth this
        whole change exists to prevent."""
        if not QUOTES_PATH.exists():
            return
        with QUOTES_PATH.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                slug = row.get("slug")
                if slug is None:
                    continue
                if row.get("ask") is not None:
                    self._have_ask.add(slug)
                if row.get("bid") is not None:
                    self._have_bid.add(slug)
        logger.info(
            "Seeded capture state from disk: %d slugs with a known ask, %d with a known bid",
            len(self._have_ask), len(self._have_bid),
        )

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
        pending = {
            slug: meta for slug, meta in self._active.items()
            if slug not in self._have_ask or slug not in self._have_bid
        }
        for slug, meta in pending.items():
            try:
                md = self.client.get_bbo(slug).get("marketData", {})
            except Exception as exc:
                logger.debug("BBO failed for %s: %s", slug, exc)
                continue

            def cents(key, data=md):
                v = (data.get(key) or {}).get("value")
                return None if v is None else round(float(v) * 100)

            bid, ask = cents("bestBid"), cents("bestAsk")
            self._append(QUOTES_PATH, {
                "ts": now_iso, "slug": slug, **meta,
                "bid": bid, "ask": ask,
                "bid_depth": md.get("bidDepth"), "ask_depth": md.get("askDepth"),
            })
            logged += 1
            if ask is not None:
                self._have_ask.add(slug)
            if bid is not None:
                self._have_bid.add(slug)
        logger.info(
            "Logged quotes for %d/%d active buckets needing them (%d already fully captured)",
            logged, len(pending), len(self._active) - len(pending),
        )

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

            outcome_yes = settlement_cents >= 50
            try:
                verified = self._verify_settlement_fn(
                    self.client, slug, meta, self._schedule_cache, self._boxscore_cache,
                )
            except Exception as exc:
                logger.debug("Settlement verification failed for %s (trusting the exchange): %s", slug, exc)
                verified = None
            if verified is not None and verified != outcome_yes:
                logger.warning(
                    "Settlement CORRECTED for %s: exchange said outcome_yes=%s, verified against "
                    "the real MLB box score says %s -- using the verified value",
                    slug, outcome_yes, verified,
                )
                outcome_yes = verified

            self._append(SETTLEMENTS_PATH, {
                "resolved_ts": datetime.now(timezone.utc).isoformat(),
                "slug": slug, **meta,
                "outcome_yes": outcome_yes,
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

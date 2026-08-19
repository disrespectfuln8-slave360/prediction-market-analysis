"""Read-only quote+outcome logger: polymarket.us NFL per-game player-prop
markets.

Split out of sports_prop_quote_logger.py on 2026-08-17, per explicit user
request, into its own module with its own data files -- not because MLB and
NFL were ever statistically pooled (sports_prop_calibration.py has always
scoped independently per (sport, stat), never per source file), but because
NFL genuinely warrants its own physical dataset the same way the tennis
quote logger got one.

Scope: this covers ONLY the per-game matchup shape, identical to MLB's --
astatc-{sport}-{matchup}-{YYYY-MM-DD}-{stat}-{player}-gte{N}, e.g.
astatc-nfl-den-atl-2026-08-14-ptd-jarsti-gte1. Reuses sports_prop_quote_
logger.py's _SLUG_RE directly (same shape, single source of truth) rather
than a second copy of the regex.

Real market survey (2026-08-17, live account): the astatc-nfl- family
actually spans THREE genuinely different bet shapes sharing one prefix,
only one of which is this module's concern:

    (this module)      Per-game matchup props, e.g. astatc-nfl-den-atl-
                        2026-08-14-ptd-jarsti-gte1. Same shape as MLB, proven
                        -- 116 real settlements already collected from the
                        2026-08-14/15 preseason window (migrated in from the
                        old shared files, not starting from zero). Zero
                        active as of 2026-08-17 (preseason just ended), but
                        expected back once the regular season starts
                        mid-September -- this logger runs and waits.
    NOT tracked here    Season-long stat totals, e.g. astatc-nfl-passtds-ou-
                        2027-01-10-tylsho ("Tyler Shough 20.5+") -- 943
                        active right now. Settles once per player per
                        SEASON; the threshold isn't even in the slug, only
                        the title. Deliberately excluded per explicit user
                        choice 2026-08-17: real calibratability is unproven
                        (one settlement per player per year is a much
                        thinner, slower-accumulating signal than a nightly
                        game prop) and would need a title-parsing regex
                        rather than a slug-parsing one -- a separate,
                        riskier undertaking than this pass.
    NOT tracked here    Game margin buckets, e.g. astatc-nfl-lv-hou-2026-08-
                        20-margin-long-25-plus ("Las Vegas by 25+") -- ~240
                        active. Categorical mutually-exclusive buckets, not
                        a threshold ladder -- the same structural shape
                        problem as the tennis exact-score markets
                        (astatc-atp/wta-*-es-*) already excluded from
                        tennis_prop_quote_logger.py, and for the same
                        reason: this module's calibration approach (price
                        decile -> historical win rate) assumes a threshold
                        ladder, not a fixed set of mutually-exclusive
                        outcomes.

Two output streams, meant to be joined later by slug for calibration:
    data/polymarket_us/nfl_prop_quotes.jsonl      -- one row per bucket per poll
    data/polymarket_us/nfl_prop_settlements.jsonl -- one row per resolved bucket

Logging stops for a slug once both its first ask and first bid are known --
same reasoning as sports_prop_quote_logger.py (which grew to 60MB/228k lines
in one day before that was fixed) and every quote logger built since.

Settlement verification (added 2026-08-18): every settlement is cross-
checked against ESPN's real box score before being written -- see
nfl_stats_verification.py's docstring. Mirrors sports_prop_quote_logger.py's
identical MLB-side wiring, added after polymarket.us's own settlement
determination was confirmed to have a real, ~14% one-directional error rate
on the MLB side; NFL's own spot-check came back clean, but the check is
wired in regardless rather than assumed safe.

READ-ONLY: only list_all_markets/get_bbo/get_settlement are ever called.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.trading.polymarket_us import nfl_stats_verification
from src.trading.polymarket_us.paper_client import _settlement_to_cents
from src.trading.polymarket_us.sports_prop_quote_logger import _SLUG_RE

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECS = 300
QUOTES_PATH = Path("data/polymarket_us/nfl_prop_quotes.jsonl")
SETTLEMENTS_PATH = Path("data/polymarket_us/nfl_prop_settlements.jsonl")

# ~14 days at the default 5-minute poll, matching sports_prop_quote_logger.py
# -- per-game props settle same-day/next-day, not weeks out.
MAX_PENDING_CYCLES = 4 * 24 * 14


def parse_nfl_prop_slug(slug: str) -> Optional[dict]:
    """(sport, matchup, date, stat, player, threshold) or None if `slug`
    isn't an NFL per-game matchup prop in the shape this logger tracks (see
    module docstring for the two other real NFL shapes this deliberately
    excludes)."""
    m = _SLUG_RE.match(slug)
    if not m or m.group("sport") != "nfl":
        return None
    return {
        "sport": m.group("sport"),
        "matchup": m.group("matchup"),
        "date": m.group("date"),
        "stat": m.group("stat"),
        "player": m.group("player"),
        "threshold": int(m.group("threshold")),
    }


class NflPropQuoteLogger:
    def __init__(self, client, verify_settlement_fn=nfl_stats_verification.verify_settlement):
        self.client = client  # PolymarketUSTradingClient (market data only)
        # Injectable so tests can substitute a fake without real network
        # calls; defaults to the real ESPN cross-check -- see
        # nfl_stats_verification.py's docstring and sports_prop_quote_
        # logger.py's identical MLB-side wiring for why this exists.
        self._verify_settlement_fn = verify_settlement_fn
        self._scoreboard_cache: dict = {}
        self._active: dict[str, dict] = {}  # slug -> parsed metadata
        self._pending_settlement: dict[str, tuple[dict, int]] = {}  # slug -> (metadata, cycles waited)
        # See module docstring -- only the first ask and first bid per slug
        # are ever used downstream, so stop fetching/logging once both are known.
        self._have_ask: set[str] = set()
        self._have_bid: set[str] = set()
        self._load_captured_state()

    def _load_captured_state(self) -> None:
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
            if not m.slug.startswith("astatc-nfl-"):
                continue
            meta = parse_nfl_prop_slug(m.slug)
            if meta is not None:
                fresh[m.slug] = meta

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
                verified = self._verify_settlement_fn(self.client, slug, meta, self._scoreboard_cache)
            except Exception as exc:
                logger.debug("Settlement verification failed for %s (trusting the exchange): %s", slug, exc)
                verified = None
            if verified is not None and verified != outcome_yes:
                logger.warning(
                    "Settlement CORRECTED for %s: exchange said outcome_yes=%s, verified against "
                    "the real ESPN box score says %s -- using the verified value",
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
            "NFL per-game prop quote logger started — READ-ONLY, no orders will be placed. "
            "poll=%ds", POLL_INTERVAL_SECS,
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
        log_dir / "nfl_prop_quote_logger.log", maxBytes=5 * 1024 * 1024, backupCount=2,
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
    _lock = acquire_single_instance_lock("nfl_prop_quote_logger")  # noqa: F841

    from src.trading.polymarket_us.trading_client import PolymarketUSTradingClient
    client = PolymarketUSTradingClient()
    try:
        NflPropQuoteLogger(client).run_forever()
    finally:
        client.close()


if __name__ == "__main__":
    main()

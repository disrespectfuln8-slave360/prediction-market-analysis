"""Read-only quote+outcome logger: polymarket.us tennis game-spread and
total-games markets.

Same motivation and architecture as macro_mlb_quote_logger.py -- polymarket.us
has no public trade-history API (see that module's docstring), so the only
viable calibration source is logging this account's own observed quotes
going forward and joining against real settlements once they resolve.

Category scoping
-----------------
Real market survey (2026-08-17, live account): tennis markets on this
exchange span several genuinely different bet shapes sharing overlapping
slug prefixes. Only two of those shapes are tracked here, deliberately:

    asc-atp / asc-wta   Point spread ladder (neg/pos X.5)   -- 181 / 214 active
    tsc-atp / tsc-wta   Total threshold ladder (X.5)        -- 125 / 131 active

These are structurally identical to asc-mlb/tsc-mlb in macro_mlb_quote_logger.py
(same neg/pos-spread and threshold-ladder shape), just a different sport --
the same quote-until-both-sides-known + decile-bucket-by-price approach
applies directly.

A second split, found 2026-08-18 while building the trading bot on top of
this data: asc-*/tsc-* each secretly cover TWO genuinely different
quantities, distinguished by a marker segment right after the date --
confirmed via real threshold-value ranges, not guessed:

    gs   game spread  (asc-*, thresholds 0.5-10.5)   -- match margin in games
    ss   set spread   (asc-*, threshold 1.5 only)     -- match margin in sets
    tg   game total    (tsc-*, thresholds 11.5-30.5)   -- total games in the match
    st   set total     (tsc-*, threshold 2.5 only)     -- total sets in the match

ss/st only ever taking one value each (1.5, 2.5) is exactly what a best-of-3
match's set-margin/set-total would produce (win 2-0 or 2-1; play 2 or 3
sets) -- structurally a coin-flip-shaped ladder, nothing like the wide,
genuinely graduated game-based ladders. Pooling these together (the
original 2026-08-17 version of this module did, before this was found) is
the exact never-pool-categories mistake already caught twice this session
elsewhere, just not noticed here until the bot's calibration needed real
per-category sample counts. category is now tour_{game,set}_{spreads,totals}
-- eight categories, not four. Explicitly NOT tracked, surveyed and rejected:

    astatc-atp/wta   Exact set score / set winner (categorical, not a
                      threshold ladder -- e.g. "Fery wins 2-0" -- a
                      fundamentally different bet shape this module's
                      calibration approach doesn't fit)
    aec-              Match moneyline, one YES/NO market per matchup, no
                      threshold at all -- same categorical-shape problem
    tec-atp-*          Tournament outright winner -- settles once per
                      tournament (weeks between resolutions)
    arankc-atp         Year-end #1 ranking -- settles once a YEAR

tec-/arankc- would never accumulate enough settlements to calibrate in any
reasonable timeframe -- the same reasoning sports_prop_quote_logger.py
already uses to silently skip astatc-fwc-* (World Cup) markets.

ATP and WTA are tracked as fully separate categories, never pooled -- ATP
(best-of-5 at majors, best-of-3 otherwise) and WTA (always best-of-3) have
different match-length and scoring dynamics that would bias a pooled
totals/spreads calibration exactly the way pooling MLB totals/spreads into
"macro" already did once this session. Per explicit user request, this
module and its data are also kept fully separate from macro_mlb_quotes.jsonl
and sports_prop_quotes.jsonl, not folded into either even though the code
pattern is copied from macro_mlb_quote_logger.py.

Two output streams, meant to be joined later by slug for calibration:
    data/polymarket_us/tennis_prop_quotes.jsonl      -- one row per bucket per poll
    data/polymarket_us/tennis_prop_settlements.jsonl -- one row per resolved bucket

Logging stops for a slug once both its first ask and first bid are known --
same reasoning as macro_mlb_quote_logger.py and (after the fact) sports_prop_
quote_logger.py -- logging every poll forever is exactly what grew
sports_prop_quotes.jsonl to 60MB/228k lines in one day before that was fixed.

READ-ONLY: only list_all_markets/get_bbo/get_settlement are ever called.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from src.trading.polymarket_us import tennis_stats_verification
from src.trading.polymarket_us.paper_client import _settlement_to_cents

logger = logging.getLogger(__name__)

TOUR_PREFIXES = {
    "asc-atp": ("atp", "spreads"),
    "asc-wta": ("wta", "spreads"),
    "tsc-atp": ("atp", "totals"),
    "tsc-wta": ("wta", "totals"),
}
# Longest-prefix-first, matching macro_mlb_quote_logger.py's defensive
# convention, even though none of these four currently collide.
_ORDERED_PREFIXES = sorted(TOUR_PREFIXES, key=len, reverse=True)

# The marker segment right after the date -- see module docstring for the
# real threshold-value evidence behind this split.
_SUBTYPE_LABELS = {
    ("spreads", "gs"): "game_spreads",
    ("spreads", "ss"): "set_spreads",
    ("totals", "tg"): "game_totals",
    ("totals", "st"): "set_totals",
}

# All eight real categories -- tour x {game,set} x {spreads,totals}.
CATEGORIES = (
    "atp_game_spreads",
    "atp_set_spreads",
    "atp_game_totals",
    "atp_set_totals",
    "wta_game_spreads",
    "wta_set_spreads",
    "wta_game_totals",
    "wta_set_totals",
)

POLL_INTERVAL_SECS = 300
QUOTES_PATH = Path("data/polymarket_us/tennis_prop_quotes.jsonl")
SETTLEMENTS_PATH = Path("data/polymarket_us/tennis_prop_settlements.jsonl")

# ~14 days at the default 5-minute poll -- matches sports_prop_quote_logger's
# figure, not macro_mlb's 30 days, since tennis matches settle same-day/next-day
# like MLB/NFL games, not weeks out like macro events.
MAX_PENDING_CYCLES = 4 * 24 * 14

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_SUBTYPE_RE = re.compile(r"-(gs|ss|tg|st)-")
_MATCHUP_RE = re.compile(r"^(?:asc|tsc)-(?:atp|wta)-([a-z0-9]+-[a-z0-9]+)-\d{4}-\d{2}-\d{2}-")
# Totals: "...-tg-17pt5" (unsigned). Spreads: "...-gs-neg-1pt5"/"...-gs-pos-8pt5"
# (signed) -- captured for completeness even though the sign convention is
# unresolved and spreads aren't used by calibration/the bot yet (see
# tennis_stats_verification.py's module docstring).
_THRESHOLD_RE = re.compile(r"-(?:(neg|pos)-)?(\d+)pt(\d+)$")


def _parse_threshold(slug: str) -> float | None:
    m = _THRESHOLD_RE.search(slug)
    if not m:
        return None
    sign, whole, frac = m.groups()
    value = float(f"{whole}.{frac}")
    return -value if sign == "neg" else value


def parse_tennis_slug(slug: str) -> dict | None:
    """(category, matchup, date, threshold) for a tracked tennis spread/
    totals slug, or None if `slug` doesn't start with any tracked prefix,
    or its sub-type marker isn't one of the four recognized ones (including
    tennis markets of an untracked shape entirely -- see module docstring).
    `date` is best-effort (regex search for the first YYYY-MM-DD
    substring), used only for staleness/display. `matchup` (player1-
    player2) is used by the trading bot's one-position-per-match guard -- a
    spread bet and a totals bet on the SAME match are correlated (both
    depend on that one match's real outcome), the same way multiple stat
    props on one MLB game are. `threshold` is unsigned for totals, signed
    (neg negative) for spreads."""
    for prefix in _ORDERED_PREFIXES:
        if not slug.startswith(prefix + "-"):
            continue
        tour, bet_type = TOUR_PREFIXES[prefix]
        m_subtype = _SUBTYPE_RE.search(slug)
        if m_subtype is None:
            return None
        label = _SUBTYPE_LABELS.get((bet_type, m_subtype.group(1)))
        if label is None:
            return None
        m_date = _DATE_RE.search(slug)
        m_matchup = _MATCHUP_RE.match(slug)
        return {
            "category": f"{tour}_{label}",
            "matchup": m_matchup.group(1) if m_matchup else None,
            "date": m_date.group(1) if m_date else None,
            "threshold": _parse_threshold(slug),
        }
    return None


class TennisPropQuoteLogger:
    def __init__(self, client, verify_settlement_fn=tennis_stats_verification.verify_settlement):
        self.client = client  # PolymarketUSTradingClient (market data only)
        # Injectable so tests can substitute a fake without real network
        # calls; defaults to the real ESPN cross-check -- see
        # tennis_stats_verification.py's docstring. Only covers totals
        # categories (returns None, i.e. "trust the exchange", for
        # spreads) -- see that module's docstring for why.
        self._verify_settlement_fn = verify_settlement_fn
        self._verification_cache: dict = {}
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
            len(self._have_ask),
            len(self._have_bid),
        )

    # ── Discovery ────────────────────────────────────────────────────────────

    def _discover(self) -> None:
        markets = self.client.list_all_markets(active=True, closed=False)
        fresh: dict[str, dict] = {}
        for m in markets:
            meta = parse_tennis_slug(m.slug)
            if meta is not None:
                fresh[m.slug] = meta

        closed_slugs = set(self._active) - set(fresh)
        for slug in closed_slugs:
            self._pending_settlement[slug] = (self._active[slug], 0)

        newly_seen = set(fresh) - set(self._active)
        self._active = fresh
        logger.info(
            "Discovery: %d active buckets (%d new, %d closed since last scan), %d pending settlement",
            len(fresh),
            len(newly_seen),
            len(closed_slugs),
            len(self._pending_settlement),
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
            slug: meta
            for slug, meta in self._active.items()
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
            self._append(
                QUOTES_PATH,
                {
                    "ts": now_iso,
                    "slug": slug,
                    **meta,
                    "bid": bid,
                    "ask": ask,
                    "bid_depth": md.get("bidDepth"),
                    "ask_depth": md.get("askDepth"),
                },
            )
            logged += 1
            if ask is not None:
                self._have_ask.add(slug)
            if bid is not None:
                self._have_bid.add(slug)
        logger.info(
            "Logged quotes for %d/%d active buckets needing them (%d already fully captured)",
            logged,
            len(pending),
            len(self._active) - len(pending),
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
                        slug,
                        cycles_waited + 1,
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
                verified = self._verify_settlement_fn(self.client, slug, meta, self._verification_cache)
            except Exception as exc:
                logger.debug("Settlement verification failed for %s (trusting the exchange): %s", slug, exc)
                verified = None
            if verified is not None and verified != outcome_yes:
                logger.warning(
                    "Settlement CORRECTED for %s: exchange said outcome_yes=%s, verified against "
                    "the real ESPN match data says %s -- using the verified value",
                    slug,
                    outcome_yes,
                    verified,
                )
                outcome_yes = verified

            self._append(
                SETTLEMENTS_PATH,
                {
                    "resolved_ts": datetime.now(timezone.utc).isoformat(),
                    "slug": slug,
                    **meta,
                    "outcome_yes": outcome_yes,
                },
            )
            resolved += 1
        self._pending_settlement = still_pending
        if resolved or dropped:
            logger.info(
                "Settlements: %d resolved, %d dropped (unresolved too long), %d still pending",
                resolved,
                dropped,
                len(still_pending),
            )

    def run_cycle(self) -> None:
        self._discover()
        self._log_quotes()
        self._check_settlements()

    def run_forever(self, stop_event=None) -> None:
        logger.info(
            "Tennis spread/totals quote logger started — READ-ONLY, no orders will be placed. Categories: %s  poll=%ds",
            CATEGORIES,
            POLL_INTERVAL_SECS,
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
        log_dir / "tennis_prop_quote_logger.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=2,
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

    _lock = acquire_single_instance_lock("tennis_prop_quote_logger")  # noqa: F841

    from src.trading.polymarket_us.trading_client import PolymarketUSTradingClient

    client = PolymarketUSTradingClient()
    try:
        TennisPropQuoteLogger(client).run_forever()
    finally:
        client.close()


if __name__ == "__main__":
    main()

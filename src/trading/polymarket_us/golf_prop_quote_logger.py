"""Read-only quote+outcome logger: polymarket.us PGA Tour round-leader markets.

Real market survey (2026-08-20, live account): the golf catalog (tec-pga-*,
tec-lpga-*, tec-liv-*) spans four real bet shapes:

    (this module)   PGA Tour round-leader, e.g. tec-pga-bmwcham-2026-08-20-
                     r1l-jspa ("J.J. Spaun leads BMW Championship after
                     Round 1"). Confirmed real, liquid, moderately-priced
                     markets for in-contention players -- e.g. r1l-jspa
                     showed a real 41.5/43.9c spread, 41 bid / 16 ask
                     depth, 22,283 shares traded. Settles once per round
                     (up to 3x per tournament), ties split $1.00 evenly.
    NOT tracked here PGA Tour outright winner (tec-pga-*-w-*). Structurally
                      a longshot category by nature -- a real check found
                      a 6.2c/6.3c market for the field's actual favorite,
                      median price across the field was 4c. Will never
                      clear this account's 30-70c no-longshot gate (see
                      apex/risk.py's MIN/MAX_TRADEABLE_PRICE_CENTS), so
                      there's no reason to spend collection effort on it.
    NOT tracked here LPGA and LIV round-leader/winner markets (tec-lpga-*,
                      tec-liv-*). Real check: multiple different players'
                      markets showing the identical placeholder 1c bid /
                      ~48-51c ask spread with 1-5 total depth and ZERO
                      shares ever traded -- these are unfilled/illiquid
                      listings, not real price discovery. Trading (or
                      calibrating) on them would mean treating a stale
                      default quote as a real market signal.

Scoped to PGA round-leader only, the one shape confirmed both liquid and
naturally landing in a moderate ("no longshot") price range for realistic
contenders -- same "narrow to the one proven, tradeable shape" discipline
already applied to NFL (one shape of three) and tennis (totals only, out
of four sub-types).

No independent settlement-verification module exists for golf yet (unlike
MLB/NFL's real box-score cross-check) -- this trusts the exchange's own
settlement outcome. A real gap, not an oversight; worth adding once real
settlement volume exists to check against.

Two output streams, meant to be joined later by slug for calibration:
    data/polymarket_us/golf_prop_quotes.jsonl      -- one row per bucket per poll
    data/polymarket_us/golf_prop_settlements.jsonl -- one row per resolved bucket

Logging stops for a slug once both its first ask and first bid are known --
same reasoning as every other quote logger in this repo (sports_prop_quote_
logger.py grew to 60MB/228k lines in one day before that was fixed).

READ-ONLY: only list_all_markets/get_bbo/get_settlement are ever called.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from src.trading.polymarket_us.paper_client import _settlement_to_cents

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECS = 300
QUOTES_PATH = Path("data/polymarket_us/golf_prop_quotes.jsonl")
SETTLEMENTS_PATH = Path("data/polymarket_us/golf_prop_settlements.jsonl")

# A round settles same-day/next-day, not weeks out -- same window as the
# other per-event quote loggers (~14 days at the default 5-minute poll).
MAX_PENDING_CYCLES = 4 * 24 * 14

_SLUG_RE = re.compile(
    r"^tec-pga-(?P<tournament>[a-z0-9]+)-(?P<date>\d{4}-\d{2}-\d{2})-"
    r"(?P<round>r[123]l)-(?P<player>[a-z]+)$"
)


def parse_golf_slug(slug: str) -> dict | None:
    """(sport, matchup, date, stat, player) or None if `slug` isn't a PGA
    round-leader market in the shape this logger tracks (see module
    docstring for the three other real golf shapes deliberately excluded).

    Deliberately reuses sports_prop_calibration.py's generic (sport, stat)
    shape rather than a golf-specific one: "sport" is always "golf",
    "stat" is which round (r1l/r2l/r3l) -- genuinely different bet types,
    same as MLB's "hits" vs "tb". "matchup" is the tournament name, not a
    2-team matchup -- the same field-of-many-entrants-share-one-outcome
    correlation sports_props_bot.py's one-position-per-game guard already
    protects against applies here too, just with "tournament" standing in
    for "game"."""
    m = _SLUG_RE.match(slug)
    if not m:
        return None
    return {
        "sport": "golf",
        "matchup": m.group("tournament"),
        "date": m.group("date"),
        "stat": m.group("round"),
        "player": m.group("player"),
    }


class GolfPropQuoteLogger:
    def __init__(self, client):
        self.client = client  # PolymarketUSTradingClient (market data only)
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
            if not m.slug.startswith("tec-pga-"):
                continue
            meta = parse_golf_slug(m.slug)
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

            # Ties split $1.00 evenly (see module docstring) -- >=50c is
            # still the right YES/NO cut since a real tie-split settlement
            # for a 2-3 way tie lands at 33-50c, and this logger only cares
            # about the binary outcome, not the exact split value.
            outcome_yes = settlement_cents >= 50
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
            "PGA round-leader quote logger started — READ-ONLY, no orders will be placed. poll=%ds",
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
        log_dir / "golf_prop_quote_logger.log",
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

    _lock = acquire_single_instance_lock("golf_prop_quote_logger")  # noqa: F841

    from src.trading.polymarket_us.trading_client import PolymarketUSTradingClient

    client = PolymarketUSTradingClient()
    try:
        GolfPropQuoteLogger(client).run_forever()
    finally:
        client.close()


if __name__ == "__main__":
    main()

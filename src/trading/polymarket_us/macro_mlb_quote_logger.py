"""Read-only quote+outcome logger: polymarket.us macro-economic and MLB
totals/spreads markets.

Same motivation as sports_prop_quote_logger.py, extended to a different
category set: Kalshi's own macro/mlb_totals bot grounds itself in a local
archive of PUBLIC historical trade prints (src/indexers/kalshi/trades.py),
but polymarket.us has no public trade-history API at all (confirmed live
2026-08-17 -- every plausible endpoint on both the gateway and api hosts
404s). The only viable calibration source here is the same one sports-props
already validated: log this account's own observed quotes going forward,
join against real settlements once they resolve, and decile-bucket by
price -- see sports_prop_calibration.py for the calibration engine this
feeds, and macro_mlb_calibration.py (not yet built) for its analog here.

Category scoping
-----------------
Real market survey (2026-08-17, 85 active macro + 439 active MLB
totals/spreads): 8 genuinely different question shapes share this module,
never pooled with each other -- the same never-pool-categories lesson
already learned twice this session (Kalshi's MLB-totals/macro mixup, and
sports-props' own per-(sport,stat) scoping):

    cbpac    Fed full-year cut/hike (2 markets, oddly-shaped slugs)
    rdc      Central bank rate decision, discrete buckets (BoC/BoI/Fed FOMC)
    gdpc     GDP growth threshold ladder
    urc      Unemployment rate threshold ladder
    nfpc     Nonfarm payrolls threshold ladder
    cpic     CPI YoY threshold ladder
    tsc-mlb  MLB game total runs threshold ladder (over/under)
    asc-mlb  MLB game point spread ladder

Slug shapes vary too much across these eight for one shared regex to be
worth it (rdc-boc-2026-09-02-hike50 vs cbpac-usfed-2026-cut vs
tsc-mlb-stl-cin-2026-08-17-dh1-f5-3pt5) -- parse_macro_mlb_slug() only
extracts what calibration actually needs (category, for scoping; date, for
staleness/display), the same minimal-parsing approach sports-props uses
(decile bucketing is driven entirely by quoted price, not by decoding the
bucket's own semantic threshold).

Two output streams, meant to be joined later by slug for calibration:
    data/polymarket_us/macro_mlb_quotes.jsonl      -- one row per bucket per poll
    data/polymarket_us/macro_mlb_settlements.jsonl -- one row per resolved bucket

Logging stops for a slug once both its first ask and first bid are known --
that's all _build_calibration_table-style logic ever reads, and logging
every poll forever is exactly what grew sports_prop_quotes.jsonl to
60MB/228k lines in one day before that was fixed (2026-08-16).

READ-ONLY: only list_all_markets/get_bbo/get_settlement are ever called.
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

CATEGORY_PREFIXES = {
    "cbpac": "fed_annual",
    "rdc": "rate_decision",
    "gdpc": "gdp",
    "urc": "unemployment",
    "nfpc": "nonfarm_payrolls",
    "cpic": "cpi",
    "tsc-mlb": "mlb_totals",
    "asc-mlb": "mlb_spreads",
}
# Longest-prefix-first so "tsc-mlb"/"asc-mlb" (containing a hyphen) are
# checked before any bare single-segment prefix could shadow them.
_ORDERED_PREFIXES = sorted(CATEGORY_PREFIXES, key=len, reverse=True)

POLL_INTERVAL_SECS = 300
QUOTES_PATH = Path("data/polymarket_us/macro_mlb_quotes.jsonl")
SETTLEMENTS_PATH = Path("data/polymarket_us/macro_mlb_settlements.jsonl")

MAX_PENDING_CYCLES = 4 * 24 * 30  # ~30 days at the default 5-minute poll -- macro events can be weeks out

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def parse_macro_mlb_slug(slug: str) -> Optional[dict]:
    """(category, date) for a macro or MLB-totals/spreads slug, or None if
    `slug` doesn't start with any tracked prefix. `date` is best-effort
    (regex search for the first YYYY-MM-DD substring) -- not every prefix
    embeds one the same way (cbpac's 2 markets are irregular), and it's
    only used for staleness/display, never for calibration bucketing."""
    for prefix in _ORDERED_PREFIXES:
        if slug.startswith(prefix + "-"):
            m = _DATE_RE.search(slug)
            return {"category": CATEGORY_PREFIXES[prefix], "date": m.group(1) if m else None}
    return None


class MacroMlbQuoteLogger:
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
            len(self._have_ask), len(self._have_bid),
        )

    # ── Discovery ────────────────────────────────────────────────────────────

    def _discover(self) -> None:
        markets = self.client.list_all_markets(active=True, closed=False)
        fresh: dict[str, dict] = {}
        for m in markets:
            meta = parse_macro_mlb_slug(m.slug)
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
            "Macro/MLB-totals quote logger started — READ-ONLY, no orders will be placed. "
            "Categories: %s  poll=%ds", sorted(set(CATEGORY_PREFIXES.values())), POLL_INTERVAL_SECS,
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
        log_dir / "macro_mlb_quote_logger.log", maxBytes=5 * 1024 * 1024, backupCount=2,
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
    _lock = acquire_single_instance_lock("macro_mlb_quote_logger")  # noqa: F841

    from src.trading.polymarket_us.trading_client import PolymarketUSTradingClient
    client = PolymarketUSTradingClient()
    try:
        MacroMlbQuoteLogger(client).run_forever()
    finally:
        client.close()


if __name__ == "__main__":
    main()

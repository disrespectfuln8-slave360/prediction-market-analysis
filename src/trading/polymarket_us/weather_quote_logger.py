"""Read-only quote logger: polymarket.us daily-high-temperature markets.

We have historical settlement outcomes and historical forecast data, but no
historical record of what the MARKET was actually quoting at decision time.
That gap can only be closed going forward, so this logs one row per active
bucket per poll — best bid/ask and depth — for a future backtest. It makes
no prediction and takes no position; it is the observational twin of the
(now-retired) dead-bid monitor and reuses that monitor's discovery/polling
skeleton, with the dead-bid/opportunity logic itself stripped out.

Settlement sources, verified from each market family's own description
text on 2026-08-02 (all settle on the NWS Climatological Report (Daily)):

    nychigh -> Central Park       KNYC
    miahigh -> Miami Intl         KMIA
    mdwhigh -> Chicago Midway     KMDW
    laxhigh -> Los Angeles Intl   KLAX
    sfohigh -> SF Intl            KSFO

Bucket bounds are parsed from market TITLES ("79 or below", "80 to 81",
"88 or above" — bounds inclusive), never from slugs: slug semantics are
inconsistent (`lt76f` means <=75 but `gte76lt77f` includes 77).

Output: logs/weather_quote_logger.log (human) and
data/polymarket_us/weather_quotes.jsonl (one JSON row per bucket per poll).

READ-ONLY: this module only ever calls list_all_markets/get_bbo. It must
never place, cancel, or otherwise touch an order — the account it shares a
client with is live and actively trading a separate strategy.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# city slug fragment -> NWS station (kept alongside the slug fragment for
# documentation/parity with the retired monitor; the station itself isn't
# queried here — this logger reads only the market book, not observations)
CITY_STATIONS = {
    "nychigh": ("KNYC", "America/New_York"),
    "miahigh": ("KMIA", "America/New_York"),
    "mdwhigh": ("KMDW", "America/Chicago"),
    "laxhigh": ("KLAX", "America/Los_Angeles"),
    "sfohigh": ("KSFO", "America/Los_Angeles"),
}

POLL_INTERVAL_SECS = 180
OUT_PATH = Path("data/polymarket_us/weather_quotes.jsonl")

_TITLE_BETWEEN = re.compile(r"^(\d+)\s+to\s+(\d+)$")
_TITLE_BELOW = re.compile(r"^(\d+)\s+or\s+below$")
_TITLE_ABOVE = re.compile(r"^(\d+)\s+or\s+above$")


def local_today(city: str) -> str:
    """Today's date in `city`'s own local timezone -- shared with
    src/trading/polymarket_us/apex/weather_model.py, which needs the same
    per-city "what date is it there right now" logic to find today's open
    bucket markets."""
    tz = ZoneInfo(CITY_STATIONS[city][1])
    return datetime.now(tz).strftime("%Y-%m-%d")


def parse_bucket_bounds(title: str) -> tuple | None:
    """(low, high) inclusive integer F bounds; None end = unbounded."""
    t = title.strip()
    m = _TITLE_BETWEEN.match(t)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = _TITLE_BELOW.match(t)
    if m:
        return None, int(m.group(1))
    m = _TITLE_ABOVE.match(t)
    if m:
        return int(m.group(1)), None
    return None


class WeatherQuoteLogger:
    def __init__(self, client):
        self.client = client  # PolymarketUSTradingClient (market data only)
        # city -> local date string its discovered buckets belong to
        self._bucket_date: dict[str, str] = {}
        # city -> list of (slug, low, high)
        self._buckets: dict[str, list] = {}

    # ── Discovery ────────────────────────────────────────────────────────────

    def _local_today(self, city: str) -> str:
        return local_today(city)

    def _discover(self) -> None:
        """One full catalog scan; refresh each city whose local date rolled."""
        need = [c for c in CITY_STATIONS if self._bucket_date.get(c) != self._local_today(c)]
        if not need:
            return
        logger.info("Discovering buckets for %s (full catalog scan)...", need)
        markets = self.client.list_all_markets(active=True, closed=False)
        for city in need:
            today = self._local_today(city)
            found = []
            for m in markets:
                if not m.slug.startswith(f"tc-temp-{city}-{today}-"):
                    continue
                bounds = parse_bucket_bounds(m.title or "")
                if bounds is None:
                    logger.warning("Unparseable bucket title %r on %s — skipping", m.title, m.slug)
                    continue
                found.append((m.slug, bounds[0], bounds[1]))
            self._buckets[city] = found
            self._bucket_date[city] = today
            logger.info("%s %s: %d buckets", city, today, len(found))

    # ── Measurement cycle ────────────────────────────────────────────────────

    def _write_row(self, row: dict) -> None:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with OUT_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def run_cycle(self) -> None:
        self._discover()
        now_iso = datetime.now(timezone.utc).isoformat()

        for city, buckets in self._buckets.items():
            if not buckets:
                continue
            logged = 0
            for slug, low, high in buckets:
                try:
                    md = self.client.get_bbo(slug).get("marketData", {})
                except Exception as exc:
                    logger.debug("BBO failed for %s: %s", slug, exc)
                    continue

                def cents(key, data=md):
                    v = (data.get(key) or {}).get("value")
                    return None if v is None else round(float(v) * 100)

                self._write_row(
                    {
                        "ts": now_iso,
                        "city": city,
                        "slug": slug,
                        "low": low,
                        "high": high,
                        "bid": cents("bestBid"),
                        "ask": cents("bestAsk"),
                        "bid_depth": md.get("bidDepth"),
                        "ask_depth": md.get("askDepth"),
                    }
                )
                logged += 1

            logger.info("[%s] logged %d/%d buckets", city, logged, len(buckets))

    def run_forever(self, stop_event=None) -> None:
        logger.info(
            "Weather quote logger started — READ-ONLY, no orders will be placed. Cities: %s  poll=%ds",
            list(CITY_STATIONS),
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
        log_dir / "weather_quote_logger.log",
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

    _lock = acquire_single_instance_lock("weather_quote_logger")  # noqa: F841

    from src.trading.polymarket_us.trading_client import PolymarketUSTradingClient

    client = PolymarketUSTradingClient()
    try:
        WeatherQuoteLogger(client).run_forever()
    finally:
        client.close()


if __name__ == "__main__":
    main()

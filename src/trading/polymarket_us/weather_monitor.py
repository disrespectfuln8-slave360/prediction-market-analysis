"""Read-only monitor: NWS observations vs polymarket.us daily-high books.

Measures — without trading — how often intraday temperature markets stay
bid above zero after the observed running max has already killed them.

The trade this is sizing up: a daily high can only rise, so once the
running max at the settlement station reaches X, every bucket whose upper
bound is below X is mathematically dead. If such a bucket still has a YES
bid, buying short collects that bid at settlement, risk-free modulo the
settlement source being read correctly. Whether that lag actually exists
(and for how long, at what depth) is exactly what this monitor records.
Run it through at least one full trading day before any trading decision.

Settlement sources, verified from each market family's own description
text on 2026-08-02 (all settle on the NWS Climatological Report (Daily)):

    nychigh -> Central Park       KNYC
    miahigh -> Miami Intl         KMIA
    mdwhigh -> Chicago Midway     KMDW
    laxhigh -> Los Angeles Intl   KLAX

Bucket bounds are parsed from market TITLES ("79 or below", "80 to 81",
"88 or above" — bounds inclusive), never from slugs: slug semantics are
inconsistent (`lt76f` means <=75 but `gte76lt77f` includes 77), and a
mis-parsed bound turns "risk-free" into a confidently wrong bet.

A dead-bucket call needs the observed max to STRICTLY exceed the bucket's
inclusive upper bound in whole degrees F: the Climatological Report rounds
to integer F, and an observation of 84.0F only guarantees the report reads
>= 84, which does not kill an "84 to 85" bucket. floor(max_obs) > upper is
the safe test, and slack beyond that is recorded so the eventual trading
rule can demand a margin.

Output: logs/weather_monitor.log (human) and
data/polymarket_us/weather_opportunities.jsonl (one JSON row per dead-bid
sighting, plus hourly book snapshots for later calibration).
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import httpx

from src.trading.polymarket_us.strategy import taker_fee_cents

logger = logging.getLogger(__name__)

# city slug fragment -> (NWS station, IANA timezone)
CITY_STATIONS = {
    "nychigh": ("KNYC", "America/New_York"),
    "miahigh": ("KMIA", "America/New_York"),
    "mdwhigh": ("KMDW", "America/Chicago"),
    "laxhigh": ("KLAX", "America/Los_Angeles"),
}

NWS_BASE = "https://api.weather.gov"
NWS_USER_AGENT = "prediction-market-analysis-weather-monitor (damianvela559@gmail.com)"

POLL_INTERVAL_SECS = 180
SNAPSHOT_EVERY_CYCLES = 20  # ~hourly at the default interval
MIN_INTERESTING_BID_CENTS = 2  # below this a dead bid isn't worth logging
OUT_PATH = Path("data/polymarket_us/weather_opportunities.jsonl")

_TITLE_BETWEEN = re.compile(r"^(\d+)\s+to\s+(\d+)$")
_TITLE_BELOW = re.compile(r"^(\d+)\s+or\s+below$")
_TITLE_ABOVE = re.compile(r"^(\d+)\s+or\s+above$")


def parse_bucket_bounds(title: str) -> Optional[tuple]:
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


class WeatherMonitor:
    def __init__(self, client):
        self.client = client  # PolymarketUSTradingClient (market data only)
        self._nws = httpx.Client(
            base_url=NWS_BASE, headers={"User-Agent": NWS_USER_AGENT}, timeout=15.0,
        )
        # city -> local date string its discovered buckets belong to
        self._bucket_date: dict[str, str] = {}
        # city -> list of (slug, low, high)
        self._buckets: dict[str, list] = {}
        self._cycle = 0

    # ── Discovery ────────────────────────────────────────────────────────────

    def _local_today(self, city: str) -> str:
        tz = ZoneInfo(CITY_STATIONS[city][1])
        return datetime.now(tz).strftime("%Y-%m-%d")

    def _discover(self) -> None:
        """One full catalog scan; refresh each city whose local date rolled.

        The markets API ignores its category filter (verified live), so
        discovery has to scan everything — done only on date change, with
        per-slug BBO polling in between.
        """
        need = [c for c in CITY_STATIONS if self._bucket_date.get(c) != self._local_today(c)]
        if not need:
            return
        logger.info("Discovering buckets for %s (full catalog scan)...", need)
        markets = self.client.list_all_markets(active=True, closed=False)
        for city in need:
            today = self._local_today(city)
            found = []
            for m in markets:
                if not m.slug.startswith("tc-temp-%s-%s-" % (city, today)):
                    continue
                bounds = parse_bucket_bounds(m.title or "")
                if bounds is None:
                    logger.warning("Unparseable bucket title %r on %s — skipping",
                                   m.title, m.slug)
                    continue
                found.append((m.slug, bounds[0], bounds[1]))
            self._buckets[city] = found
            self._bucket_date[city] = today
            logger.info("%s %s: %d buckets", city, today, len(found))

    # ── Observations ─────────────────────────────────────────────────────────

    def _running_max_f(self, city: str) -> Optional[float]:
        """Max observed temperature (F) since local midnight, or None."""
        station, tzname = CITY_STATIONS[city]
        tz = ZoneInfo(tzname)
        midnight = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)
        try:
            r = self._nws.get(
                "/stations/%s/observations" % station,
                params={"start": midnight.astimezone(timezone.utc).isoformat(), "limit": 500},
            )
            r.raise_for_status()
            temps = []
            for feat in r.json().get("features", []):
                val = (feat.get("properties", {}).get("temperature") or {}).get("value")
                if val is not None:
                    temps.append(float(val) * 9.0 / 5.0 + 32.0)
            return max(temps) if temps else None
        except Exception as exc:
            logger.warning("NWS fetch failed for %s: %s", station, exc)
            return None

    # ── Measurement cycle ────────────────────────────────────────────────────

    def _write_row(self, row: dict) -> None:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with OUT_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def run_cycle(self) -> None:
        self._discover()
        self._cycle += 1
        snapshot_cycle = (self._cycle % SNAPSHOT_EVERY_CYCLES) == 1
        now_iso = datetime.now(timezone.utc).isoformat()

        for city, buckets in self._buckets.items():
            if not buckets:
                continue
            max_f = self._running_max_f(city)
            if max_f is None:
                continue
            dead_line = []
            for slug, low, high in buckets:
                try:
                    md = self.client.get_bbo(slug).get("marketData", {})
                except Exception as exc:
                    logger.debug("BBO failed for %s: %s", slug, exc)
                    continue

                def cents(key):
                    v = (md.get(key) or {}).get("value")
                    return None if v is None else round(float(v) * 100)

                bid, ask = cents("bestBid"), cents("bestAsk")
                dead = high is not None and math.floor(max_f) > high

                if snapshot_cycle:
                    self._write_row({
                        "type": "snapshot", "ts": now_iso, "city": city, "slug": slug,
                        "low": low, "high": high, "bid": bid, "ask": ask,
                        "bid_depth": md.get("bidDepth"), "ask_depth": md.get("askDepth"),
                        "obs_max_f": round(max_f, 1), "dead": dead,
                    })

                if dead and bid is not None and bid >= MIN_INTERESTING_BID_CENTS:
                    # Buying short pays the bucket's bid at settlement; the
                    # short leg's entry cost is (100 - bid).
                    net = bid - taker_fee_cents(100 - bid, 1)
                    row = {
                        "type": "opportunity", "ts": now_iso, "city": city, "slug": slug,
                        "high": high, "bid": bid, "bid_depth": md.get("bidDepth"),
                        "obs_max_f": round(max_f, 1),
                        "slack_f": round(max_f - (high + 1), 1),
                        "net_cents_per_ct": net,
                    }
                    self._write_row(row)
                    dead_line.append("%s bid=%dc depth=%s net=%+dc" % (
                        slug.rsplit("-", 1)[-1], bid, md.get("bidDepth"), net))

            if dead_line:
                logger.info("[%s] obs_max=%.1fF  DEAD-BID: %s", city, max_f, "  ".join(dead_line))
            else:
                logger.info("[%s] obs_max=%.1fF  no dead bids", city, max_f)

    def run_forever(self, stop_event=None) -> None:
        logger.info("Weather monitor started — READ-ONLY, no orders will be placed. "
                    "Cities: %s  poll=%ds", list(CITY_STATIONS), POLL_INTERVAL_SECS)
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
        self._nws.close()


def main():
    from logging.handlers import RotatingFileHandler

    from dotenv import load_dotenv
    load_dotenv()

    log_dir = Path(__file__).resolve().parents[3] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_dir / "weather_monitor.log", maxBytes=5 * 1024 * 1024, backupCount=2,
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
    _lock = acquire_single_instance_lock("weather_monitor")  # noqa: F841

    from src.trading.polymarket_us.trading_client import PolymarketUSTradingClient
    client = PolymarketUSTradingClient()
    try:
        WeatherMonitor(client).run_forever()
    finally:
        client.close()


if __name__ == "__main__":
    main()

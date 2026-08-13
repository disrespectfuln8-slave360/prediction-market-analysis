"""One-shot backfill: resolved polymarket.us daily-high weather buckets x NOAA NBM forecasts.

Joins every resolved (city, date) bucket-market settlement with the NBM
forecast that was live "the evening before" that date, and writes the result
to data/polymarket_us/weather_history/weather_history.parquet for the
separate calibration analysis (src/analysis/polymarket_us/weather_forecast_calibration.py)
to score. Run once: python scripts/backfill_weather_history.py

Market discovery
-----------------
The bulk listing endpoint (list_all_markets with closed=True) does not
return resolved weather markets under any filter combination (confirmed
live) — only active/open markets show up there. So each city-day's bucket
ladder is found by probing get_market() on slugs of the confirmed live
pattern tc-temp-{city}-{date}-gte{N}lt{N+1}f (title "N to N+1", inclusive,
a 2-degree-wide bucket) and catching the 404 a nonexistent slug returns
(confirmed live: get_market -> httpx.HTTPStatusError, response.status_code
== 404). Buckets partition contiguously in 2-degree steps (verified live
against 08/05 and 08/06 listings for all 5 cities), so once ANY bucket is
found for a city-day, walking outward by 2F in both directions enumerates
the whole ladder in O(buckets) probes instead of sweeping the full 0-130F
range. The walk seeds its starting guess from the previous day's ladder for
that city (day-to-day highs rarely jump far); the very first date per city,
or any day the seeded guess misses entirely, falls back to an exhaustive
1F-step sweep of the full range so an unexpected parity shift (bucket
anchors are NOT always even — e.g. NYC uses even N, Chicago uses odd N,
both verified live) still gets found correctly, just slower for that one
day.

NBM forecast cycle choice (READ THIS BEFORE CHANGING lead_hours numbers)
--------------------------------------------------------------------------
Per station, we want a forecast issued "the evening before" the settlement
date's local midnight, with roughly 12-18h of lead time — recent enough to
be a real forecast, but locked in before the day it's predicting begins.
NBM NBS text bulletins are published *hourly* (confirmed live: blend_nbstx
exists at every UTC hour, not just synoptic 00/06/12/18Z), so instead of
snapping to a 6-hourly cycle we just take the exact midpoint of that
window: NBM_LEAD_HOURS = 15.0, i.e. cycle_utc = (local midnight starting
`date`) - 15h. Concretely, for the Apr29-Aug4 2026 backfill window (all
inside US DST) this resolves to the 13Z cycle of the day before for
nychigh/miahigh (Eastern), 14Z for mdwhigh (Central), and 16Z for
laxhigh/sfohigh (Pacific) — verified live end-to-end against 2026-08-03
NYC (13Z 08/02 cycle's TXN=81F for local date 08/03 matched the 12Z 08/03
cycle's TXN=81F for the same date, and the market settled the "80 to 81"
bucket that same day). The 15h figure is computed via zoneinfo against
each station's actual local midnight, not hardcoded per city, so it stays
correct if the backfill window is ever extended across a DST boundary.

NBM text bulletin parsing
--------------------------
Each station's block starts with a line matching " {ICAO}    NBM ...". The
FHR row underneath gives forecast hour offsets from the cycle in a fixed
5-char-label + 3-char-field layout (verified live via direct byte-offset
inspection, not guessed). The TXN row directly below is NOT the deterministic
max alone — it alternates between the 24h max ("TX", at forecast hours whose
valid time lands on 00Z) and the overnight min ("TN", at 12Z), matching the
literal row name (TX + N -> TXN). Only the 00Z slots are max-temp forecasts;
this is what "TXN row is the deterministic forecast max" from earlier
research meant in practice, and picking blindly would silently use the
overnight low half the time. XND, directly beneath, is that slot's std dev.
The 00Z valid-hour rule holds regardless of the cycle's own hour (verified
against both a 12Z and a 13Z cycle file), so no per-cycle special-casing is
needed.

Large-file handling
--------------------
Each blend_nbstx.t{HH}z bulletin bundles all CONUS stations and runs
15-30MB. Since nychigh/miahigh share a timezone (and so a cycle) and
laxhigh/sfohigh do too, at most 3 distinct (date, cycle-hour) files are
needed per backfill date. Each is downloaded once to the OS temp dir,
all needed stations are extracted from it immediately, and the file is
deleted before moving to the next date — never more than ~3 files
(~90MB) resident on disk at a time, and nothing is left behind even if
the run is interrupted (best-effort cleanup in a finally).

READ-ONLY on the polymarket.us API: only get_market and get_settlement are
called. No order-placing method is imported, let alone called.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from src.trading.polymarket_us.models import Market, _amount_to_cents
from src.trading.polymarket_us.nbm_forecast import cleanup_nbm_cache, fetch_nbm_forecast
from src.trading.polymarket_us.trading_client import PolymarketUSTradingClient
from src.trading.polymarket_us.weather_quote_logger import CITY_STATIONS, parse_bucket_bounds

logger = logging.getLogger("backfill_weather_history")

# Verified live: nothing on this platform resolves before this date.
START_DATE = "2026-04-29"

BUCKET_WIDTH = 2  # tc-temp-...-gte{N}lt{N+1}f covers N and N+1 -> 2F wide
BUCKET_LOW_F = 0
BUCKET_HIGH_F = 132  # a little past the "0-130F is safe" guidance, cheap insurance

NBM_CACHE_DIR = Path(tempfile.gettempdir()) / "pmus_weather_backfill_nbm_cache"

OUT_DIR = Path("data/polymarket_us/weather_history")
OUT_PATH = OUT_DIR / "weather_history.parquet"

SCHEMA_COLUMNS = [
    "city", "station", "date", "bucket_slug", "low", "high",
    "settlement", "forecast_mu", "forecast_sigma", "forecast_cycle", "lead_hours",
]


# ── polymarket.us bucket discovery ──────────────────────────────────────────

def _mid_slug(city: str, d: str, n: int) -> str:
    return f"tc-temp-{city}-{d}-gte{n}lt{n + 1}f"


def _low_slug(city: str, d: str, n: int) -> str:
    return f"tc-temp-{city}-{d}-lt{n}f"


def _high_slug(city: str, d: str, n: int) -> str:
    return f"tc-temp-{city}-{d}-gte{n}f"


def _probe(client: PolymarketUSTradingClient, slug: str) -> Optional[Market]:
    try:
        return client.get_market(slug)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return None
        raise


def _radius_candidates(center: int, lo: int, hi: int, step: int):
    """center, center+step, center-step, center+2*step, ... clipped to [lo, hi]."""
    if lo <= center <= hi:
        yield center
    r = step
    while center - r >= lo or center + r <= hi:
        for cand in (center + r, center - r):
            if lo <= cand <= hi:
                yield cand
        r += step


def _find_seed_n(client: PolymarketUSTradingClient, city: str, d: str, guess_n: Optional[int]):
    """Find any one valid middle bucket for this city-day; return (n, Market) or (None, None)."""
    if guess_n is not None:
        for n in _radius_candidates(guess_n, BUCKET_LOW_F, BUCKET_HIGH_F - BUCKET_WIDTH, BUCKET_WIDTH):
            m = _probe(client, _mid_slug(city, d, n))
            if m is not None:
                return n, m
    # No guess, or the guessed parity missed entirely (bucket anchors aren't
    # always the same parity day to day — verified live) — fall back to a
    # full 1F-step sweep so we don't miss a ladder anchored on the other parity.
    for n in range(BUCKET_LOW_F, BUCKET_HIGH_F - BUCKET_WIDTH + 1):
        m = _probe(client, _mid_slug(city, d, n))
        if m is not None:
            return n, m
    return None, None


def discover_day_buckets(client: PolymarketUSTradingClient, city: str, d: str, guess_n: Optional[int]):
    """Return (markets, next_guess_n) or None if no bucket could be found for this city-day."""
    n0, m0 = _find_seed_n(client, city, d, guess_n)
    if n0 is None:
        return None

    mid_buckets = {n0: m0}
    n = n0
    while True:
        prev = n - BUCKET_WIDTH
        if prev < BUCKET_LOW_F:
            break
        m = _probe(client, _mid_slug(city, d, prev))
        if m is None:
            break
        mid_buckets[prev] = m
        n = prev
    n_min = n

    n = n0
    while True:
        nxt = n + BUCKET_WIDTH
        if nxt > BUCKET_HIGH_F:
            break
        m = _probe(client, _mid_slug(city, d, nxt))
        if m is None:
            break
        mid_buckets[nxt] = m
        n = nxt
    n_max = n

    markets = []
    low_m = _probe(client, _low_slug(city, d, n_min))
    if low_m is not None:
        markets.append(low_m)
    else:
        logger.warning("%s %s: no unbounded-low bucket found at n=%d", city, d, n_min)

    markets.extend(mid_buckets[k] for k in sorted(mid_buckets))

    high_n = n_max + BUCKET_WIDTH
    high_m = _probe(client, _high_slug(city, d, high_n))
    if high_m is not None:
        markets.append(high_m)
    else:
        logger.warning("%s %s: no unbounded-high bucket found at n=%d", city, d, high_n)

    return markets, n_min


def _extract_settlement(payload: Optional[dict]) -> Optional[int]:
    """0/1 winning-outcome flag from a /settlement response.

    Mirrors paper_client._settlement_to_cents's defensive shape-handling
    (observed live: bare {"settlement": 1}, not an Amount dict) but returns
    the raw 0/1 outcome instead of cents, since that's what this schema wants.
    """
    if not payload:
        return None
    for key in ("settlement", "settlementPrice"):
        if key in payload:
            val = payload[key]
            if isinstance(val, dict):
                cents = _amount_to_cents(val)
                return None if cents is None else (1 if cents >= 50 else 0)
            try:
                return 1 if float(val) >= 0.5 else 0
            except (TypeError, ValueError):
                return None
    return None


# ── Orchestration ────────────────────────────────────────────────────────────

def _date_range(start: str, end_exclusive: date) -> list[str]:
    d = datetime.strptime(start, "%Y-%m-%d").date()
    out = []
    while d < end_exclusive:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def process_city_day(
    client: PolymarketUSTradingClient,
    nbm_http: httpx.Client,
    nbm_cache: dict,
    city: str,
    station: str,
    d: str,
    guess_n: Optional[int],
    bucket_issues: Counter,
) -> tuple[list[dict], Optional[int], str]:
    """Returns (rows, next_guess_n, status) — status is "ok", "no_buckets",
    "no_forecast", or "no_resolved_buckets"."""
    discovery = discover_day_buckets(client, city, d, guess_n)
    if discovery is None:
        return [], guess_n, "no_buckets"
    markets, next_guess_n = discovery

    forecast = fetch_nbm_forecast(nbm_http, nbm_cache, city, d, CITY_STATIONS, NBM_CACHE_DIR)
    if forecast is None:
        return [], next_guess_n, "no_forecast"

    rows = []
    for m in markets:
        if not m.closed:
            bucket_issues["not yet closed"] += 1
            continue
        bounds = parse_bucket_bounds(m.title or "")
        if bounds is None:
            bucket_issues["unparseable title"] += 1
            logger.warning("Unparseable bucket title %r on %s", m.title, m.slug)
            continue
        try:
            settle_payload = client.get_settlement(m.slug)
        except Exception as exc:
            bucket_issues["settlement fetch failed"] += 1
            logger.warning("get_settlement failed for %s: %s", m.slug, exc)
            continue
        settlement = _extract_settlement(settle_payload)
        if settlement is None:
            bucket_issues["settlement missing/unparseable"] += 1
            continue

        low, high = bounds
        rows.append({
            "city": city,
            "station": station,
            "date": d,
            "bucket_slug": m.slug,
            "low": float(low) if low is not None else float("nan"),
            "high": float(high) if high is not None else float("nan"),
            "settlement": int(settlement),
            "forecast_mu": forecast["forecast_mu"],
            "forecast_sigma": forecast["forecast_sigma"],
            "forecast_cycle": forecast["forecast_cycle"],
            "lead_hours": forecast["lead_hours"],
        })

    if not rows:
        return [], next_guess_n, "no_resolved_buckets"
    return rows, next_guess_n, "ok"


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    dates = _date_range(START_DATE, datetime.now(timezone.utc).date())
    if not dates:
        print("No dates in [START_DATE, today) — nothing to backfill.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    guess_n: dict[str, Optional[int]] = dict.fromkeys(CITY_STATIONS)
    city_day_status: Counter = Counter()
    bucket_issues: Counter = Counter()

    nbm_http = httpx.Client(timeout=120.0, follow_redirects=True)
    total = len(dates) * len(CITY_STATIONS)

    with PolymarketUSTradingClient() as client:
        try:
            with tqdm(total=total, desc="Backfilling weather history", unit="city-day") as pbar:
                for d in dates:
                    nbm_cache: dict = {}
                    for city, (station, _tzname) in CITY_STATIONS.items():
                        try:
                            day_rows, guess_n[city], status = process_city_day(
                                client, nbm_http, nbm_cache, city, station, d, guess_n[city], bucket_issues,
                            )
                        except Exception as exc:
                            logger.error("Unexpected error on %s %s: %s", city, d, exc, exc_info=True)
                            status = "unexpected_error"
                            day_rows = []
                        rows.extend(day_rows)
                        city_day_status[status] += 1
                        pbar.update(1)
                        pbar.set_postfix(rows=len(rows))
                    cleanup_nbm_cache(nbm_cache)
        finally:
            nbm_http.close()
            shutil.rmtree(NBM_CACHE_DIR, ignore_errors=True)

    df = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    # Atomic write: a run that dies partway through (network blip, killed
    # process -- this backfill runs for hours against a live API and has
    # been interrupted mid-run in practice) must never leave a truncated
    # or half-written parquet file where Apex's calibration gate expects a
    # complete one.
    tmp_path = OUT_PATH.with_suffix(".tmp")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, OUT_PATH)

    processed = sum(city_day_status.values())
    ok = city_day_status.get("ok", 0)
    print(f"\nCity-days processed: {processed:,} ({len(CITY_STATIONS)} cities x {len(dates)} dates)")
    print(f"City-days with >=1 resolved row: {ok:,}")
    skipped = processed - ok
    print(f"City-days skipped/incomplete: {skipped:,}")
    for reason, n in city_day_status.most_common():
        if reason != "ok":
            print(f"  {reason}: {n:,}")
    if bucket_issues:
        print("Per-bucket issues (within otherwise-processed city-days):")
        for reason, n in bucket_issues.most_common():
            print(f"  {reason}: {n:,}")
    print(f"Rows written: {len(df):,} -> {OUT_PATH}")
    print(f"Date range covered: {dates[0]} .. {dates[-1]}")


if __name__ == "__main__":
    main()

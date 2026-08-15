"""Incremental backfill: resolved Kalshi daily-high-temperature markets x NOAA
NBM forecasts.

Same purpose and shape as polymarket_us's scripts/backfill_weather_history.py
(and the same two bugs already fixed there, deliberately not reintroduced
here): incremental (resumes from the day after the latest date already in
weather_history.parquet, appends rather than rebuilds) and recurring
(--loop mode with a real rotating log file, since a hidden detached launch
has no console to write to).

Market discovery is simpler here than the polymarket_us version needed --
Kalshi's own market snapshots (data/kalshi/markets/*.parquet, kept fresh by
run_all.py's indexer) already carry settled results for every KXHIGH*/
KXHIGHT* market, so there is no need to probe individual tickers and catch
404s to find what exists. Each pass just re-reads the existing snapshot for
newly-resolved rows in the new date range and joins them against a NOAA NBM
forecast for that (series, date).

IMPORTANT: Kalshi changed the weather settlement source from NWS to "The
Weather Company" on 2026-08-14, system-wide (confirmed live via the public
/series endpoint). This backfill does not distinguish settlement era --  it
just keeps accumulating real (forecast, settlement) pairs forever, the same
way it would after any other regime change, so the calibration table
self-corrects as fresh post-2026-08-14 data accumulates. See
src/trading/kalshi/weather_model.py's module docstring for the full caveat.

One-shot: python scripts/backfill_kalshi_weather_history.py
Recurring (daemon, re-runs every RELOOP_INTERVAL_SECS): add --loop
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import httpx
import pandas as pd
from dotenv import load_dotenv

from src.trading.kalshi.weather_model import CITY_STATIONS, parse_bucket_title, series_of
from src.trading.polymarket_us.nbm_forecast import cleanup_nbm_cache, fetch_nbm_forecast

logger = logging.getLogger("backfill_kalshi_weather_history")

# Verified live: the earliest close_time across the whole KXHIGH*/KXHIGHT*
# family in the local markets snapshot.
START_DATE = "2026-04-25"

# Same reasoning as polymarket_us's version: frequent enough that a day's
# settlement lands well within KalshiWeatherGroundingEngine's own
# STALE_HISTORY_DAYS=3 gate, infrequent enough not to re-hammer the NBM
# archive for data that mostly hasn't changed since the last pass.
RELOOP_INTERVAL_SECS = 6 * 3600

MARKETS_GLOB = "data/kalshi/markets/*.parquet"
NBM_CACHE_DIR = Path(__import__("tempfile").gettempdir()) / "kalshi_weather_backfill_nbm_cache"

OUT_DIR = Path("data/kalshi/weather_history")
OUT_PATH = OUT_DIR / "weather_history.parquet"

SCHEMA_COLUMNS = [
    "series", "station", "date", "ticker", "low", "high",
    "settlement", "forecast_mu", "forecast_sigma", "forecast_cycle", "lead_hours",
]


def _resume_start_date() -> str:
    """Day after the latest date already in weather_history.parquet, or
    START_DATE if that file doesn't exist yet."""
    if not OUT_PATH.exists():
        return START_DATE
    try:
        existing_max = pd.read_parquet(OUT_PATH, columns=["date"])["date"].max()
    except Exception as exc:
        logger.warning("Could not read existing %s (%s) — resuming from START_DATE", OUT_PATH, exc)
        return START_DATE
    if pd.isna(existing_max):
        return START_DATE
    next_day = datetime.strptime(str(existing_max), "%Y-%m-%d").date() + timedelta(days=1)
    return next_day.strftime("%Y-%m-%d")


def _fetch_settled_rows(start_date: str, end_date_exclusive: str) -> pd.DataFrame:
    con = duckdb.connect()
    prefixes = " OR ".join(f"ticker LIKE '{s}-%'" for s in CITY_STATIONS)
    return con.execute(f"""
        SELECT ticker, title, result, close_time
        FROM '{MARKETS_GLOB}'
        WHERE ({prefixes})
          AND result IN ('yes', 'no')
          AND close_time >= TIMESTAMP '{start_date}'
          AND close_time < TIMESTAMP '{end_date_exclusive}'
    """).df()


def run_once() -> None:
    start = _resume_start_date()
    today = datetime.now(timezone.utc).date()
    end_exclusive = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    if start >= end_exclusive:
        print(f"No new dates since {start} — nothing to backfill.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    settled = _fetch_settled_rows(start, end_exclusive)
    print(f"{len(settled)} newly-settled rows found ({start} .. {today})")
    if settled.empty:
        return

    parsed_rows = []
    for r in settled.itertuples(index=False):
        series = series_of(r.ticker)
        parsed = parse_bucket_title(r.title)
        if parsed is None:
            logger.warning("Unparseable bucket title %r on %s", r.title, r.ticker)
            continue
        low, high, date_str = parsed
        parsed_rows.append({
            "series": series, "ticker": r.ticker, "low": low, "high": high,
            "date": date_str, "settlement": 1 if r.result == "yes" else 0,
        })
    print(f"{len(parsed_rows)} rows parsed")

    nbm_http = httpx.Client(timeout=120.0, follow_redirects=True)
    raw_cache: dict = {}
    rows = []
    try:
        for pr in parsed_rows:
            station, _tz = CITY_STATIONS.get(pr["series"], (None, None))
            if station is None:
                continue
            forecast = fetch_nbm_forecast(nbm_http, raw_cache, pr["series"], pr["date"], CITY_STATIONS, NBM_CACHE_DIR)
            if forecast is None:
                continue
            rows.append({
                "series": pr["series"], "station": station, "date": pr["date"], "ticker": pr["ticker"],
                "low": pr["low"] if pr["low"] is not None else float("nan"),
                "high": pr["high"] if pr["high"] is not None else float("nan"),
                "settlement": pr["settlement"],
                "forecast_mu": forecast["forecast_mu"], "forecast_sigma": forecast["forecast_sigma"],
                "forecast_cycle": forecast["forecast_cycle"], "lead_hours": forecast["lead_hours"],
            })
    finally:
        cleanup_nbm_cache(raw_cache)
        nbm_http.close()

    print(f"{len(rows)} rows matched to a real NOAA forecast")
    if not rows:
        return

    new_df = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    # Append rather than rebuild -- _resume_start_date() already guarantees
    # `settled` doesn't overlap what's in OUT_PATH, so this can't introduce
    # duplicate rows.
    if OUT_PATH.exists():
        df = pd.concat([pd.read_parquet(OUT_PATH), new_df], ignore_index=True)
    else:
        df = new_df
    # Atomic write: a run killed partway through must never leave a
    # truncated parquet where the calibration engine expects a complete one.
    tmp_path = OUT_PATH.with_suffix(".tmp")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, OUT_PATH)

    print(f"New rows written this run: {len(new_df):,}  (total in file: {len(df):,}) -> {OUT_PATH}")


def main():
    from logging.handlers import RotatingFileHandler

    load_dotenv()
    log_dir = Path(__file__).resolve().parents[1] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "backfill_kalshi_weather_history.log", maxBytes=5 * 1024 * 1024, backupCount=2,
        encoding="utf-8",
    )
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S",
        handlers=[file_handler, logging.StreamHandler()],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--loop", action="store_true",
        help=f"Keep running, re-checking for newly-resolved days every {RELOOP_INTERVAL_SECS}s "
             "instead of exiting after one pass",
    )
    args = parser.parse_args()

    if not args.loop:
        run_once()
        return

    logger.info("Running in --loop mode, re-checking every %ds", RELOOP_INTERVAL_SECS)
    while True:
        try:
            run_once()
        except Exception as exc:
            logger.error("Backfill pass failed: %s", exc, exc_info=True)
        time.sleep(RELOOP_INTERVAL_SECS)


if __name__ == "__main__":
    main()

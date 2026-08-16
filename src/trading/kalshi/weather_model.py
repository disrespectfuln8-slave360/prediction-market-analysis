"""Grounded probability engine for Kalshi daily-high-temperature markets
(the KXHIGH*/KXHIGHT* series family).

Same shape as polymarket_us/apex/weather_model.py -- a NOAA NBM forecast run
through a continuity-corrected Normal-CDF model, only usable to trade on if
a per-city, per-decile calibration table says the model is actually
trustworthy in that range. Reuses that module's forecast-fetching machinery
(bucket_probability, fetch_nbm_forecast) directly since none of it is
polymarket.us-specific -- it's just NOAA NBM parsing.

Scoped per (series, decile), never pooled across cities -- Apex's own
calibration table made exactly that mistake once already (see its
_build_calibration_table docstring, fixed 2026-08-15); building this one
without it from the start.

IMPORTANT CAVEAT: Kalshi changed its weather settlement source from the
National Weather Service to "The Weather Company" on 2026-08-14 (confirmed
live via the public /trade-api/v2/series/{ticker} endpoint, system-wide
across every KXHIGH*/KXHIGHT* series). The backfilled history this table
starts from is entirely from the retired NWS era -- it's the best available
prior, not a guarantee the same calibration holds under the new source.
Nothing here distinguishes settlement source; the calibration table is just
a rolling window of real (forecast, settlement) pairs, so it self-corrects
as fresh Weather-Company-settled data accumulates over time the same way it
would after any other regime change. Treat early trades against
NWS-only-backed deciles with extra caution until enough post-2026-08-14 data
exists to re-check them independently.

Data source (written by scripts/backfill_kalshi_weather_history.py):
    data/kalshi/weather_history/weather_history.parquet
"""
from __future__ import annotations

import logging
import re
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import pandas as pd

from src.trading.polymarket_us.nbm_forecast import bucket_probability, cleanup_nbm_cache, fetch_nbm_forecast

logger = logging.getLogger(__name__)

DEFAULT_HISTORY_PATH = Path("data/kalshi/weather_history/weather_history.parquet")
NBM_CACHE_DIR = Path(tempfile.gettempdir()) / "kalshi_weather_bot_nbm_cache"

# Same conservative defaults Apex already validated for this exact modeling
# approach (NOAA NBM forecast + continuity-corrected Normal CDF) -- not
# re-derived here, just inherited until there's a reason to tune them
# differently for Kalshi's own settlement behavior.
MIN_SAMPLE_FLOOR = 25
MAX_CALIBRATION_ERROR_PCT = 10.0
STALE_HISTORY_DAYS = 3

# series ticker prefix -> (NOAA station, IANA timezone). Cities Apex also
# trades on polymarket.us reuse the exact same station it already validated
# live; the rest are best-effort primary-airport ASOS codes, NOT verified
# against Kalshi's own settlement methodology (old NWS or new Weather
# Company) -- worth confirming against the contract_terms PDF before
# trusting any single-city number here in isolation.
CITY_STATIONS: dict[str, tuple[str, str]] = {
    "KXHIGHLAX": ("KLAX", "America/Los_Angeles"),
    "KXHIGHTSFO": ("KSFO", "America/Los_Angeles"),
    "KXHIGHNY": ("KNYC", "America/New_York"),
    "KXHIGHMIA": ("KMIA", "America/New_York"),
    "KXHIGHCHI": ("KMDW", "America/Chicago"),
    "KXHIGHTATL": ("KATL", "America/New_York"),
    "KXHIGHTDC": ("KDCA", "America/New_York"),
    "KXHIGHTBOS": ("KBOS", "America/New_York"),
    "KXHIGHPHIL": ("KPHL", "America/New_York"),
    "KXHIGHTMIN": ("KMSP", "America/Chicago"),
    "KXHIGHTOKC": ("KOKC", "America/Chicago"),
    "KXHIGHTSATX": ("KSAT", "America/Chicago"),
    "KXHIGHAUS": ("KAUS", "America/Chicago"),
    "KXHIGHTDAL": ("KDAL", "America/Chicago"),
    "KXHIGHTHOU": ("KHOU", "America/Chicago"),
    "KXHIGHTNOLA": ("KMSY", "America/Chicago"),
    "KXHIGHDEN": ("KDEN", "America/Denver"),
    "KXHIGHTPHX": ("KPHX", "America/Phoenix"),
    "KXHIGHTLV": ("KLAS", "America/Los_Angeles"),
    "KXHIGHTSEA": ("KSEA", "America/Los_Angeles"),
}

_RANGE_RE = re.compile(r"be\s+(\d+)-(\d+)°\s+on\s+(\w+\s+\d+,\s+\d+)")
_BELOW_RE = re.compile(r"be\s+<(\d+)°\s+on\s+(\w+\s+\d+,\s+\d+)")
_ABOVE_RE = re.compile(r"be\s+>(\d+)°\s+on\s+(\w+\s+\d+,\s+\d+)")


def parse_bucket_title(title: str) -> Optional[tuple[Optional[float], Optional[float], str]]:
    """(low, high, date "YYYY-MM-DD") from a KXHIGH*/KXHIGHT* market title,
    or None if it doesn't match the known shape. Both bounds None-able --
    the tail buckets ("<N", ">N") are unbounded on one side, same convention
    as polymarket.us's weather buckets."""
    m = _RANGE_RE.search(title)
    if m:
        low, high, datestr = m.groups()
        return float(low), float(high), datetime.strptime(datestr, "%b %d, %Y").strftime("%Y-%m-%d")
    m = _BELOW_RE.search(title)
    if m:
        high, datestr = m.groups()
        return None, float(high), datetime.strptime(datestr, "%b %d, %Y").strftime("%Y-%m-%d")
    m = _ABOVE_RE.search(title)
    if m:
        low, datestr = m.groups()
        return float(low), None, datetime.strptime(datestr, "%b %d, %Y").strftime("%Y-%m-%d")
    return None


def series_of(ticker: str) -> str:
    return ticker.split("-")[0]


def _decile_low(prob: float) -> int:
    pct = prob * 100.0
    return int(min(np.floor(pct / 10.0) * 10.0, 90.0))


@dataclass(frozen=True)
class WeatherBucketGrounding:
    series: str
    date: str
    low: Optional[float]
    high: Optional[float]
    model_prob: float
    forecast_mu: float
    forecast_sigma: float
    forecast_cycle: str
    calibration_trustworthy: bool
    calibration_note: str
    calibration_sample_count: int
    calibration_trustworthy_no: bool
    calibration_note_no: str
    calibration_sample_count_no: int


def _build_calibration_table(history: pd.DataFrame) -> dict[tuple[str, int], dict]:
    """(series, bucket_low) -> {count, actual_rate, mean_predicted}. Scoped
    per series (city), never pooled -- see this module's docstring."""
    rows = []
    for row in history.itertuples(index=False):
        mp = bucket_probability(row.forecast_mu, row.forecast_sigma, row.low, row.high)
        if mp is None:
            continue
        rows.append({"series": row.series, "model_prob_pct": mp * 100.0, "settlement": int(row.settlement)})
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    df["bucket_low"] = np.minimum(np.floor(df["model_prob_pct"] / 10.0) * 10.0, 90.0).astype(int)
    grouped = df.groupby(["series", "bucket_low"]).agg(
        count=("settlement", "size"),
        wins=("settlement", "sum"),
        mean_predicted=("model_prob_pct", "mean"),
    )
    grouped["actual_rate"] = 100.0 * grouped["wins"] / grouped["count"]
    return {
        (str(series), int(bucket_low)): {
            "count": int(r["count"]),
            "actual_rate": float(r["actual_rate"]),
            "mean_predicted": float(r["mean_predicted"]),
        }
        for (series, bucket_low), r in grouped.iterrows()
    }


class KalshiWeatherGroundingEngine:
    def __init__(self, history_path: Path = DEFAULT_HISTORY_PATH, nbm_cache_dir: Path = NBM_CACHE_DIR):
        self.history_path = history_path
        self.nbm_cache_dir = nbm_cache_dir
        self._nbm_http = httpx.Client(timeout=120.0, follow_redirects=True)
        self._calibration: dict[tuple[str, int], dict] = {}
        self._history_max_date: Optional[str] = None
        self._forecast_cache: dict[tuple, Optional[dict]] = {}  # (series, date) -> forecast dict or None
        self._lock = threading.Lock()
        self.reload_calibration()

    def close(self) -> None:
        self._nbm_http.close()

    def forecast_for(self, series: str, date_str: str) -> Optional[dict]:
        """NBM forecast for `series`'s city on `date_str`, cached for the
        process lifetime -- same reasoning as Apex's own forecast_for: the
        cycle this repo picks is a fixed historical cycle by the time a
        market for that date is tradeable, so it never changes underfoot
        within the same day."""
        key = (series, date_str)
        with self._lock:
            if key in self._forecast_cache:
                return self._forecast_cache[key]
        raw_cache: dict = {}
        forecast = fetch_nbm_forecast(self._nbm_http, raw_cache, series, date_str, CITY_STATIONS, self.nbm_cache_dir)
        cleanup_nbm_cache(raw_cache)
        with self._lock:
            self._forecast_cache[key] = forecast
        if forecast is None:
            logger.warning("[kalshi-weather] No NBM forecast available for %s %s", series, date_str)
        return forecast

    def reload_calibration(self) -> None:
        if not self.history_path.exists():
            self._calibration = {}
            self._history_max_date = None
            logger.warning(
                "[kalshi-weather] No history at %s -- run "
                "scripts/backfill_kalshi_weather_history.py before trading; "
                "every bucket will be treated as ungrounded until then.",
                self.history_path,
            )
            return
        try:
            history = pd.read_parquet(self.history_path)
        except Exception as exc:
            logger.warning("[kalshi-weather] Failed to read history at %s: %s", self.history_path, exc)
            self._calibration = {}
            self._history_max_date = None
            return
        history = history.dropna(subset=["forecast_mu", "forecast_sigma", "settlement"])
        self._calibration = _build_calibration_table(history)
        self._history_max_date = str(history["date"].max()) if not history.empty else None
        logger.info(
            "[kalshi-weather] Calibration table loaded: %d resolved rows across %d (series, decile) bucket(s), latest date %s",
            len(history), len(self._calibration), self._history_max_date,
        )

    def history_is_stale(self, today: Optional[str] = None) -> bool:
        if self._history_max_date is None:
            return True
        today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            age = (datetime.strptime(today, "%Y-%m-%d") - datetime.strptime(self._history_max_date, "%Y-%m-%d")).days
        except ValueError:
            return True
        return age > STALE_HISTORY_DAYS

    def _trustworthiness(self, series: str, model_prob: float) -> tuple[bool, str, int]:
        bucket_low = _decile_low(model_prob)
        info = self._calibration.get((series, bucket_low))
        if info is None or info["count"] < MIN_SAMPLE_FLOOR:
            count = info["count"] if info else 0
            return False, f"only {count} historical {series} sample(s) in the {bucket_low}-{bucket_low + 10}% range (need {MIN_SAMPLE_FLOOR})", count
        error = abs(info["actual_rate"] - info["mean_predicted"])
        if error > MAX_CALIBRATION_ERROR_PCT:
            return False, (
                f"{series} model measured {error:.1f}pp off actual in the {bucket_low}-{bucket_low + 10}% range "
                f"({info['count']} samples) — exceeds the {MAX_CALIBRATION_ERROR_PCT:.0f}pp tolerance"
            ), info["count"]
        return True, f"calibration OK: {info['count']} {series} samples, {error:.1f}pp measured error", info["count"]

    def evaluate_bucket(self, series: str, date_str: str, low: Optional[float], high: Optional[float]) -> Optional[WeatherBucketGrounding]:
        """Fetch (or reuse the cached) forecast for `series`/`date_str` and
        evaluate this bucket against it. Returns None if no forecast is
        available -- the caller (the bot) never sees an ungrounded bucket."""
        forecast = self.forecast_for(series, date_str)
        if forecast is None:
            return None
        return self.evaluate(
            series, date_str, low, high,
            forecast["forecast_mu"], forecast["forecast_sigma"], forecast["forecast_cycle"],
        )

    def evaluate(self, series: str, date_str: str, low: Optional[float], high: Optional[float],
                 forecast_mu: float, forecast_sigma: float, forecast_cycle: str) -> Optional[WeatherBucketGrounding]:
        mp = bucket_probability(forecast_mu, forecast_sigma, low, high)
        if mp is None:
            return None
        trustworthy, note, count = self._trustworthiness(series, mp)
        trustworthy_no, note_no, count_no = self._trustworthiness(series, 1.0 - mp)
        return WeatherBucketGrounding(
            series=series, date=date_str, low=low, high=high, model_prob=mp,
            forecast_mu=forecast_mu, forecast_sigma=forecast_sigma, forecast_cycle=forecast_cycle,
            calibration_trustworthy=trustworthy, calibration_note=note, calibration_sample_count=count,
            calibration_trustworthy_no=trustworthy_no, calibration_note_no=note_no,
            calibration_sample_count_no=count_no,
        )

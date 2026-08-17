"""Grounded probability engine for polymarket.us daily-high-temperature markets.

This is the one place in this bot where a probability estimate comes from
somewhere other than an LLM's judgment: the NOAA NBM forecast (mean, std)
run through the same continuity-corrected Normal-CDF model that
src/analysis/polymarket_us/weather_forecast_calibration.py already scores
against real settlement history (Brier score, ECE, MCE — see that module).
A model_prob is only usable to trade on if the historical calibration table
says the model is actually trustworthy in that probability range; outside
that, this returns "ungrounded" rather than a number for the caller to bet
on faith.

No other market category on polymarket.us has an equivalent grounded model
in this repo (see strategy.py's note on the removed crypto/macro mispricing
bots — subjective probability with no supporting data was tried here and
didn't work), so Apex is deliberately scoped to only these markets for now.
"""
from __future__ import annotations

import logging
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import pandas as pd

from src.trading.polymarket_us.models import Market
from src.trading.polymarket_us.nbm_forecast import bucket_probability, cleanup_nbm_cache, fetch_nbm_forecast
from src.trading.polymarket_us.weather_quote_logger import CITY_STATIONS, local_today, parse_bucket_bounds

logger = logging.getLogger(__name__)

DEFAULT_HISTORY_PATH = Path("data/polymarket_us/weather_history/weather_history.parquet")
NBM_CACHE_DIR = Path(tempfile.gettempdir()) / "pmus_apex_nbm_cache"

# Mirrors weather_forecast_calibration.py's MIN_SAMPLE_FLOOR — the same
# sample-size floor this codebase already trusts before leaning on a
# bucket's stats for anything.
MIN_SAMPLE_FLOOR = 25

# Largest acceptable |actual_rate - mean_predicted| (percentage points) in a
# model_prob's decile bucket for that estimate to be considered trustworthy.
# Not a formally derived number — a conservative line under which "the model
# is measurably wrong in this range" should block trading rather than a
# smaller, noisier miscalibration that's within the sample's own error bars.
MAX_CALIBRATION_ERROR_PCT = 10.0

# If the backfilled history hasn't been refreshed in this many days, treat
# grounding as stale rather than trade on a calibration table that may no
# longer reflect current conditions. scripts/backfill_weather_history.py
# needs to be re-run periodically to keep this current.
STALE_HISTORY_DAYS = 3

# TEMPORARY, data-motivated shrinkage on live trade sizing — 2026-08-16,
# after two straight days of every open longshot bucket losing. Checked the
# standardized residual z = (actual - forecast_mu) / forecast_sigma across
# real settlements: the full history (Apr-Aug, n=328) is close to perfectly
# calibrated (std(z)=1.05), but the last 15 days alone (n=67) run tighter
# (std(z)=0.87), and 2026-08-15 alone (n=5, too small to trust on its own,
# but consistent with the 15-day trend) tighter still (std(z)=0.57) --
# real outcomes have been landing closer to forecast_mu than the model
# assumes. A bucket several degrees off the mean is a tail event under a
# Normal(mu, sigma); SHRINKING sigma pulls that same bucket further into
# the tail and correctly lowers its estimated probability (verified
# numerically before applying this: a bucket at mu+4.5..5.5 with sigma=2.0
# scores 0.92%; at sigma=2.0*1.15 it roughly DOUBLES to 1.68% -- inflating
# sigma would have made these exact losing trades look even more
# attractive, backwards from what the data is saying). Applied only where
# model_prob feeds a live trade decision (evaluate_bucket) -- deliberately
# NOT applied in _build_calibration_table, which must keep measuring
# historical accuracy against the real, un-shrunk sigma or it would corrupt
# its own read of what actually happened. Revisit once more real
# settlements accumulate past this stretch; 0.87 is the 15-day std(z)
# directly (the more robust of the two recent-window estimates, not the
# noisier single-day 0.57 figure).
LIVE_SIGMA_SHRINKAGE = 0.87


@dataclass(frozen=True)
class BucketGrounding:
    city: str
    date: str
    low: Optional[float]
    high: Optional[float]
    model_prob: float
    forecast_mu: float
    forecast_sigma: float
    forecast_cycle: str
    calibration_trustworthy: bool  # YES side: model_prob's own decile
    calibration_note: str
    calibration_sample_count: int
    calibration_trustworthy_no: bool  # NO side: (1 - model_prob)'s decile -- a
    # different decile than the YES side whenever model_prob isn't ~50%, so it
    # needs its own independent calibration check rather than reusing the YES
    # side's flag (a bucket calibrated as trustworthy at 5% says nothing about
    # whether the complementary 95% decile has ever been checked).
    calibration_note_no: str
    calibration_sample_count_no: int


def _bucket_low_pct(prob: float) -> int:
    """Same decile bucketing as weather_forecast_calibration.py._bucket_by_decile."""
    pct = prob * 100.0
    return int(min(np.floor(pct / 10.0) * 10.0, 90.0))


def _build_calibration_table(history: pd.DataFrame) -> dict[tuple[str, int], dict]:
    """(city, bucket_low) -> {count, actual_rate, mean_predicted}.

    Scoped per city, not pooled across all five -- checked live 2026-08-15
    against the real backfilled history: the pooled 60-70% decile measured
    12.5pp off (92 samples, well above the floor), and splitting by city
    showed why -- LAX alone was 16.4pp off (52 samples) and Miami 10.8pp off
    (29 samples), both plausibly explained by daily-high volatility (marine
    layer, afternoon convection) the NBM's forecast_sigma may be
    underestimating for those two stations specifically. NYC/Chicago/SFO had
    too few rows in that range to say anything either way. Pooling meant
    every city's 60-70% bucket got judged by LAX and Miami's miscalibration
    (or, just as bad, the reverse: a real problem in one city hiding behind
    other cities' good calibration in the same aggregate). Same lesson as
    the Kalshi MLB-totals/macro pooling bug and the sports-props
    never-pool-categories design, just found here for the first time."""
    rows = []
    for row in history.itertuples(index=False):
        mp = bucket_probability(row.forecast_mu, row.forecast_sigma, row.low, row.high)
        if mp is None:
            continue
        rows.append({"city": row.city, "model_prob_pct": mp * 100.0, "settlement": int(row.settlement)})
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    df["bucket_low"] = np.minimum(np.floor(df["model_prob_pct"] / 10.0) * 10.0, 90.0).astype(int)
    grouped = df.groupby(["city", "bucket_low"]).agg(
        count=("settlement", "size"),
        wins=("settlement", "sum"),
        mean_predicted=("model_prob_pct", "mean"),
    )
    grouped["actual_rate"] = 100.0 * grouped["wins"] / grouped["count"]
    return {
        (str(city), int(bucket_low)): {
            "count": int(r["count"]),
            "actual_rate": float(r["actual_rate"]),
            "mean_predicted": float(r["mean_predicted"]),
        }
        for (city, bucket_low), r in grouped.iterrows()
    }


class WeatherGroundingEngine:
    def __init__(
        self,
        history_path: Path = DEFAULT_HISTORY_PATH,
        nbm_cache_dir: Path = NBM_CACHE_DIR,
    ):
        self.history_path = history_path
        self.nbm_cache_dir = nbm_cache_dir
        self._nbm_http = httpx.Client(timeout=120.0, follow_redirects=True)
        self._calibration: dict[tuple[str, int], dict] = {}
        self._history_max_date: Optional[str] = None
        self._forecast_cache: dict[tuple, Optional[dict]] = {}  # (city, date) -> forecast dict or None
        self._lock = threading.Lock()
        self.reload_calibration()

    def close(self) -> None:
        self._nbm_http.close()

    # ── Calibration table ────────────────────────────────────────────────────

    def reload_calibration(self) -> None:
        """Recompute the calibration table from the on-disk parquet. Cheap
        enough (a few hundred to a few thousand rows) to call once at
        startup and once per scan day without caching machinery."""
        if not self.history_path.exists():
            self._calibration = {}
            self._history_max_date = None
            logger.warning(
                "[apex] No weather history at %s — run scripts/backfill_weather_history.py "
                "before trading; every bucket will be treated as ungrounded until then.",
                self.history_path,
            )
            return
        try:
            history = pd.read_parquet(self.history_path)
        except Exception as exc:
            logger.warning("[apex] Failed to read weather history at %s: %s", self.history_path, exc)
            self._calibration = {}
            self._history_max_date = None
            return
        history = history.dropna(subset=["forecast_mu", "forecast_sigma", "settlement"])
        self._calibration = _build_calibration_table(history)
        self._history_max_date = str(history["date"].max()) if not history.empty else None
        logger.info(
            "[apex] Calibration table loaded: %d resolved rows across %d (city, decile) bucket(s), latest date %s",
            len(history), len(self._calibration), self._history_max_date,
        )

    def history_is_stale(self, today: str) -> bool:
        if self._history_max_date is None:
            return True
        try:
            age = (datetime.strptime(today, "%Y-%m-%d") - datetime.strptime(self._history_max_date, "%Y-%m-%d")).days
        except ValueError:
            return True
        return age > STALE_HISTORY_DAYS

    def _trustworthiness(self, city: str, model_prob: float) -> tuple[bool, str, int]:
        bucket_low = _bucket_low_pct(model_prob)
        info = self._calibration.get((city, bucket_low))
        if info is None or info["count"] < MIN_SAMPLE_FLOOR:
            count = info["count"] if info else 0
            return False, f"only {count} historical {city} sample(s) in the {bucket_low}-{bucket_low + 10}% range (need {MIN_SAMPLE_FLOOR})", count
        error = abs(info["actual_rate"] - info["mean_predicted"])
        if error > MAX_CALIBRATION_ERROR_PCT:
            return False, (
                f"{city} model measured {error:.1f}pp off actual in the {bucket_low}-{bucket_low + 10}% range "
                f"({info['count']} samples) — exceeds the {MAX_CALIBRATION_ERROR_PCT:.0f}pp tolerance"
            ), info["count"]
        return True, f"calibration OK: {info['count']} {city} samples, {error:.1f}pp measured error", info["count"]

    # ── Live forecast ────────────────────────────────────────────────────────

    def _local_today(self, city: str) -> str:
        return local_today(city)

    def forecast_for(self, city: str, date_str: str) -> Optional[dict]:
        """NBM forecast for `city` on `date_str`, cached for the process
        lifetime — the cycle this repo picks (see nbm_forecast.py) is a fixed
        historical cycle by the time a market for that date is tradeable, so
        it never changes underfoot within the same day."""
        key = (city, date_str)
        with self._lock:
            if key in self._forecast_cache:
                return self._forecast_cache[key]
        raw_cache: dict = {}
        forecast = fetch_nbm_forecast(
            self._nbm_http, raw_cache, city, date_str, CITY_STATIONS, self.nbm_cache_dir,
        )
        cleanup_nbm_cache(raw_cache)
        with self._lock:
            self._forecast_cache[key] = forecast
        if forecast is None:
            logger.warning("[apex] No NBM forecast available for %s %s", city, date_str)
        return forecast

    # ── Bucket discovery ─────────────────────────────────────────────────────

    def discover_open_buckets(self, markets: list[Market]) -> dict[str, list[tuple[str, Optional[float], Optional[float], Market]]]:
        """Today's (per-city local date) open bucket markets, grouped by city.
        Each entry is (slug, low, high, Market)."""
        out: dict[str, list] = {}
        for city in CITY_STATIONS:
            today = self._local_today(city)
            prefix = f"tc-temp-{city}-{today}-"
            found = []
            for m in markets:
                if not m.slug.startswith(prefix):
                    continue
                bounds = parse_bucket_bounds(m.title or "")
                if bounds is None:
                    continue
                found.append((m.slug, bounds[0], bounds[1], m))
            if found:
                out[city] = found
        return out

    def evaluate_bucket(self, city: str, low: Optional[float], high: Optional[float]) -> Optional[BucketGrounding]:
        date_str = self._local_today(city)
        forecast = self.forecast_for(city, date_str)
        if forecast is None:
            return None
        # See LIVE_SIGMA_SHRINKAGE's comment -- live trade sizing only, the
        # raw forecast_sigma below (stored on the returned grounding, and
        # everywhere else in this class) is untouched.
        live_sigma = forecast["forecast_sigma"] * LIVE_SIGMA_SHRINKAGE
        mp = bucket_probability(forecast["forecast_mu"], live_sigma, low, high)
        if mp is None:
            return None
        trustworthy, note, count = self._trustworthiness(city, mp)
        trustworthy_no, note_no, count_no = self._trustworthiness(city, 1.0 - mp)
        return BucketGrounding(
            city=city, date=date_str, low=low, high=high, model_prob=mp,
            forecast_mu=forecast["forecast_mu"], forecast_sigma=forecast["forecast_sigma"],
            forecast_cycle=forecast["forecast_cycle"],
            calibration_trustworthy=trustworthy, calibration_note=note, calibration_sample_count=count,
            calibration_trustworthy_no=trustworthy_no, calibration_note_no=note_no,
            calibration_sample_count_no=count_no,
        )

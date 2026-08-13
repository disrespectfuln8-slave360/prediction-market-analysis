"""NOAA NBM forecast fetch/parse for polymarket.us daily-high-temperature markets.

Extracted from scripts/backfill_weather_history.py so the same live-verified
parsing logic has one home — src/trading/polymarket_us/apex/weather_model.py
needs it for TODAY's forecast (an open market's grounding), and the backfill
script needs it for past dates (resolved markets' calibration history). Both
call fetch_nbm_forecast(); nothing about the parsing differs between the two
uses, so duplicating it would just be two copies to keep in sync.

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
correct if it's ever used across a DST boundary.

For a market that is still OPEN when queried (today's date), this same
formula picks a cycle that is already in the past (12-18h before local
midnight of a day that has already started), so it is always fetchable —
there is no "forecast not issued yet" case for the current day.

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
"""
from __future__ import annotations

import logging
import math
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import httpx

logger = logging.getLogger(__name__)

NBM_BASE_URL = "https://noaa-nbm-grib2-pds.s3.amazonaws.com"
NBM_LEAD_HOURS = 15.0  # see module docstring — midpoint of the 12-18h target window
NBM_LABEL_WIDTH = 5
NBM_FIELD_WIDTH = 3


def _target_cycle_utc(tzname: str, d: str) -> tuple[datetime, datetime]:
    tz = ZoneInfo(tzname)
    local_date = datetime.strptime(d, "%Y-%m-%d").date()
    local_midnight = datetime(local_date.year, local_date.month, local_date.day, tzinfo=tz)
    target_utc = local_midnight.astimezone(timezone.utc)
    cycle_utc = (target_utc - timedelta(hours=NBM_LEAD_HOURS)).replace(minute=0, second=0, microsecond=0)
    return cycle_utc, target_utc


def _nbm_cache_key(cycle_utc: datetime) -> tuple[str, str]:
    return cycle_utc.strftime("%Y%m%d"), cycle_utc.strftime("%H")


def _nbm_url(cycle_utc: datetime) -> str:
    yyyymmdd, hh = _nbm_cache_key(cycle_utc)
    return f"{NBM_BASE_URL}/blend.{yyyymmdd}/{hh}/text/blend_nbstx.t{hh}z"


def _download_cycle(nbm_http: httpx.Client, cycle_utc: datetime, cache_dir: Path, attempts: int = 3) -> Path:
    yyyymmdd, hh = _nbm_cache_key(cycle_utc)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"blend_nbstx_{yyyymmdd}_{hh}.txt"
    url = _nbm_url(cycle_utc)

    last_exc = None
    for attempt in range(attempts):
        try:
            with nbm_http.stream("GET", url) as r:
                r.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in r.iter_bytes(chunk_size=1 << 20):
                        f.write(chunk)
            return dest
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            last_exc = exc
            dest.unlink(missing_ok=True)
            if attempt < attempts - 1:
                time.sleep(2**attempt)
    raise last_exc


def _parse_fixed_row(line: str) -> tuple[str, list[str]]:
    label = line[:NBM_LABEL_WIDTH].strip()
    rest = line[NBM_LABEL_WIDTH:]
    vals = [rest[i : i + NBM_FIELD_WIDTH].strip() for i in range(0, len(rest), NBM_FIELD_WIDTH)]
    return label, vals


def _extract_station_txn(path: Path, station: str):
    """(fhr_vals, txn_vals, xnd_vals) for one station's block, or None if not found."""
    header_re = re.compile(rf"^ {re.escape(station)}\s+NBM")
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = iter(f)
        for line in lines:
            if not header_re.match(line):
                continue
            fhr = txn = xnd = None
            # FHR/TXN/XND always land in the first few rows after the header
            # (DT, UTC, FHR, TXN, XND, ...) — 10 is generous slack.
            for _ in range(10):
                row = next(lines, None)
                if row is None:
                    break
                label, vals = _parse_fixed_row(row.rstrip("\n"))
                if label == "FHR":
                    fhr = vals
                elif label == "TXN":
                    txn = vals
                elif label == "XND":
                    xnd = vals
                if fhr and txn and xnd:
                    break
            return fhr, txn, xnd
    return None


def _select_tx(fhr_vals, txn_vals, xnd_vals, cycle_utc: datetime, tz: ZoneInfo, target_date: date):
    """The (mu, sigma) TX/XND pair whose valid local date == target_date, or None.

    TXN alternates TX (max, valid hour == 00Z) and TN (min, valid hour ==
    12Z) — see module docstring. Only 00Z slots are candidates.
    """
    if not (fhr_vals and txn_vals and xnd_vals):
        return None
    for f_s, t_s, x_s in zip(fhr_vals, txn_vals, xnd_vals):
        if not f_s or not t_s:
            continue
        try:
            fhr = int(f_s)
        except ValueError:
            continue
        valid_utc = cycle_utc + timedelta(hours=fhr)
        if valid_utc.hour != 0:
            continue
        if valid_utc.astimezone(tz).date() != target_date:
            continue
        try:
            mu = float(t_s)
            sigma = float(x_s) if x_s else None
        except ValueError:
            return None
        return mu, sigma
    return None


def _get_or_download_cycle(nbm_http: httpx.Client, cache: dict, cycle_utc: datetime, cache_dir: Path) -> Optional[Path]:
    key = _nbm_cache_key(cycle_utc)
    if key in cache:
        return cache[key]  # may be None if this cycle already failed today
    try:
        path = _download_cycle(nbm_http, cycle_utc, cache_dir)
    except Exception as exc:
        logger.warning("NBM download failed for cycle %s: %s", key, exc)
        cache[key] = None
        return None
    cache[key] = path
    return path


def fetch_nbm_forecast(
    nbm_http: httpx.Client,
    cache: dict,
    city: str,
    d: str,
    city_stations: dict,
    cache_dir: Path,
) -> Optional[dict]:
    """(forecast_mu, forecast_sigma, forecast_cycle, lead_hours) for `city` on date `d`
    ("YYYY-MM-DD"), or None if the bulletin couldn't be fetched/parsed.

    `cache` is a dict the caller owns, keyed by (yyyymmdd, hh) NBM cycle —
    share one across a batch of calls that may hit the same cycle file
    (e.g. nychigh/miahigh share a timezone and so a cycle) to avoid
    redundant downloads, then clean it up with cleanup_nbm_cache().
    """
    station, tzname = city_stations[city]
    tz = ZoneInfo(tzname)
    cycle_utc, target_utc = _target_cycle_utc(tzname, d)

    path = _get_or_download_cycle(nbm_http, cache, cycle_utc, cache_dir)
    if path is None:
        return None

    try:
        parsed = _extract_station_txn(path, station)
    except OSError as exc:
        logger.warning("Failed reading cached NBM file for %s: %s", station, exc)
        return None
    if parsed is None:
        logger.warning("Station %s not found in NBM bulletin for cycle %s", station, cycle_utc.isoformat())
        return None

    target_date = datetime.strptime(d, "%Y-%m-%d").date()
    sel = _select_tx(*parsed, cycle_utc=cycle_utc, tz=tz, target_date=target_date)
    if sel is None or sel[1] is None:
        logger.warning("No usable TX/XND slot for %s %s in cycle %s", station, d, cycle_utc.isoformat())
        return None

    mu, sigma = sel
    lead_hours = (target_utc - cycle_utc).total_seconds() / 3600.0
    return {
        "forecast_mu": mu,
        "forecast_sigma": sigma,
        "forecast_cycle": cycle_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lead_hours": round(lead_hours, 2),
    }


def cleanup_nbm_cache(cache: dict) -> None:
    for path in cache.values():
        if path is not None:
            path.unlink(missing_ok=True)
    cache.clear()


def _is_unbounded(x) -> bool:
    """True for Python None (the live path -- parse_bucket_bounds()'s
    unbounded-end marker) and for float NaN (the historical path --
    backfill_weather_history.py writes unbounded ends to the parquet as
    `float("nan")`, since a real DataFrame column can't hold a mixed
    None/float column cleanly). A bare `x is None` check missed the NaN
    case entirely: `nan - 0.5` stays `nan`, and `max(0.0, min(1.0, nan))`
    silently resolves to exactly 1.0 (Python's min/max never compare True
    against NaN, so both calls just return their first argument) --
    every unbounded historical bucket read back a fabricated 100%
    probability regardless of the real forecast, corroding a third of the
    calibration table (737 of 2213 backfilled rows) into looking like the
    model is dramatically overconfident when the bug was in this bounds
    check, not the model.
    """
    if x is None:
        return True
    try:
        return math.isnan(x)
    except TypeError:
        return False


def bucket_probability(mu: float, sigma: Optional[float], low: Optional[float], high: Optional[float]) -> Optional[float]:
    """P(low <= actual_high <= high) under Normal(mu, sigma), scalar version of the
    same continuity-corrected model used by
    src/analysis/polymarket_us/weather_forecast_calibration.py._add_model_prob
    (settlement rounds to integer F, hence the +/-0.5F correction). Both
    None and NaN bounds mean unbounded (-inf/+inf) -- see _is_unbounded's
    docstring for why both forms have to be handled. Returns None if sigma
    is missing or not positive — a degenerate forecast is not something to
    size a real bet against.
    """
    if sigma is None or sigma <= 0:
        return None
    from scipy.stats import norm

    low_bound = -float("inf") if _is_unbounded(low) else low - 0.5
    high_bound = float("inf") if _is_unbounded(high) else high + 0.5
    prob = norm.cdf(high_bound, loc=mu, scale=sigma) - norm.cdf(low_bound, loc=mu, scale=sigma)
    return max(0.0, min(1.0, prob))

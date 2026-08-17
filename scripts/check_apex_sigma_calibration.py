"""Daily check: is Apex's live sigma-shrinkage still needed?

Context (2026-08-16/17): Apex's open weather positions lost two days in a
row (2026-08-14, 2026-08-15). Checked standardized residuals
z = (actual_temp - forecast_mu) / forecast_sigma against real settlements:
the full history (n=328) was close to perfectly calibrated (std(z)=1.05),
but the trailing ~15 days ran tighter (std(z)=0.87) -- real outcomes
landing closer to the forecast than the model assumed. Applied
LIVE_SIGMA_SHRINKAGE=0.87 in src/trading/polymarket_us/apex/weather_model.py,
live trade sizing only, as a temporary, data-motivated correction.

This script re-runs that same check daily as new settlements accumulate,
so a human can tell whether the trailing-window std(z) has drifted back
toward 1.0 (the shrinkage is no longer justified -- consider removing it)
or stayed low / dropped further (the regime effect is persisting or
worsening -- consider keeping or deepening it). It only reports numbers;
it never modifies LIVE_SIGMA_SHRINKAGE and never touches Apex's running
state (no pause/resume) -- that decision stays with a human.

Refreshes data/polymarket_us/weather_history/weather_history.parquet first
(same incremental backfill scripts/backfill_weather_history.py already
does), then recomputes the check against the fresh data.

Usage: python -m scripts.check_apex_sigma_calibration
(run from the repo root; see scripts/backfill_weather_history.py's own
docstring for why -- this needs `src` on the import path.)
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.trading.polymarket_us.apex.weather_model import LIVE_SIGMA_SHRINKAGE

logger = logging.getLogger("check_apex_sigma_calibration")

HISTORY_PATH = Path("data/polymarket_us/weather_history/weather_history.parquet")
TRAILING_WINDOW_DAYS = 15

# If the trailing-window std(z) climbs back above this, the shrinkage fix
# has likely stopped being justified -- worth a human re-check even if
# nothing else prompts one.
REVERT_THRESHOLD = 0.95


def _standardized_residuals(df: pd.DataFrame) -> pd.DataFrame:
    wins = df[df["settlement"] == 1].dropna(subset=["low", "high"]).copy()
    wins["actual_proxy"] = (wins["low"] + wins["high"]) / 2.0
    wins["z"] = (wins["actual_proxy"] - wins["forecast_mu"]) / wins["forecast_sigma"]
    wins["date"] = pd.to_datetime(wins["date"])
    return wins


def run_once() -> None:
    if not HISTORY_PATH.exists():
        logger.warning("No history at %s -- has the backfill ever run?", HISTORY_PATH)
        return

    df = pd.read_parquet(HISTORY_PATH)
    df = df.dropna(subset=["forecast_mu", "forecast_sigma", "settlement"])
    wins = _standardized_residuals(df)

    if wins.empty:
        logger.warning("No usable (bounded, settled) rows in %s", HISTORY_PATH)
        return

    full_mean, full_std, full_n = wins["z"].mean(), wins["z"].std(), len(wins)

    trailing = wins[wins["date"] >= (wins["date"].max() - pd.Timedelta(days=TRAILING_WINDOW_DAYS))]
    trail_mean = trailing["z"].mean() if len(trailing) else float("nan")
    trail_std = trailing["z"].std() if len(trailing) > 1 else float("nan")
    trail_n = len(trailing)

    logger.info(
        "Full history: n=%d mean(z)=%+.3f std(z)=%.3f  |  Trailing %dd: n=%d mean(z)=%+.3f std(z)=%.3f  |  current LIVE_SIGMA_SHRINKAGE=%.2f",
        full_n, full_mean, full_std, TRAILING_WINDOW_DAYS, trail_n, trail_mean, trail_std, LIVE_SIGMA_SHRINKAGE,
    )

    if trail_n < 20:
        logger.info("Trailing sample still thin (n=%d) -- treat std(z) as noisy until more settlements land.", trail_n)
        return

    if trail_std >= REVERT_THRESHOLD:
        logger.warning(
            "Trailing std(z)=%.3f is back near/above 1.0 -- the sigma-shrinkage fix may no longer be "
            "justified. Consider reviewing LIVE_SIGMA_SHRINKAGE in "
            "src/trading/polymarket_us/apex/weather_model.py.",
            trail_std,
        )
    elif trail_std < full_std - 0.15:
        logger.warning(
            "Trailing std(z)=%.3f is still notably tighter than the full-history std(z)=%.3f -- the "
            "recent-regime effect looks like it's persisting, not resolving.",
            trail_std, full_std,
        )
    else:
        logger.info("Trailing std(z)=%.3f -- no strong signal either way yet.", trail_std)


def main():
    from logging.handlers import RotatingFileHandler

    load_dotenv()
    log_dir = Path(__file__).resolve().parents[1] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "check_apex_sigma_calibration.log", maxBytes=2 * 1024 * 1024, backupCount=2,
        encoding="utf-8",
    )
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[file_handler, logging.StreamHandler()],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Refresh the history first so the check reflects newly-settled days,
    # not whatever was on disk from the last unrelated backfill run.
    try:
        from scripts.backfill_weather_history import run_once as refresh_history
        refresh_history()
    except Exception as exc:
        logger.warning("Backfill refresh failed (%s) -- checking against whatever's already on disk.", exc)

    try:
        run_once()
    except Exception as exc:
        logger.error("Sigma calibration check failed: %s", exc, exc_info=True)


if __name__ == "__main__":
    main()

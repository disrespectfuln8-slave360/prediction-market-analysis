"""Launch the NOAA-forecast-grounded weather-trading bot for Kalshi
(KXHIGH*/KXHIGHT* daily-high-temperature markets).

See src/trading/kalshi/weather_bot.py and weather_model.py for the full
design rationale -- why this runs as its own process rather than a thread
inside run_all.py, why it's scoped per (series/city, decile) rather than
pooled, and the open caveat about Kalshi's settlement source changing from
NWS to The Weather Company on 2026-08-14.

Paper mode by default, like every other bot launcher in this repo. Set
KALSHI_WEATHER_PAPER_TRADING=false to go live -- deliberately a separate
flag from KALSHI_PAPER_TRADING (crypto's) and
KALSHI_MACRO_MLB_TOTALS_PAPER_TRADING (macro/mlb_totals'): all three share
one live Kalshi account, so arming one must never silently arm another.

Prerequisite: scripts/backfill_kalshi_weather_history.py must have been run
at least once (and kept running with --loop to stay current) -- this bot
refuses to trade on stale or missing calibration history rather than fall
back to an ungrounded guess.

Usage
-----
    python run_kalshi_weather.py
"""
from __future__ import annotations

import logging
import signal
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_LOG_DIR = Path(__file__).resolve().parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_file_handler = RotatingFileHandler(
    _LOG_DIR / "run_kalshi_weather.log", maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(name)-20s]  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[_file_handler, logging.StreamHandler()],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from src.common.single_instance import acquire_single_instance_lock
from src.trading.kalshi.learning import PerformanceTracker
from src.trading.kalshi.trading_client import KalshiTradingClient
from src.trading.kalshi.weather_bot import KalshiWeatherBot
from src.trading.kalshi.weather_model import KalshiWeatherGroundingEngine
from src.trading.mode import is_paper_mode, log_mode_banner

# Deliberately not crypto's or macro/mlb_totals' paper ledger/learning
# files -- four bots sharing one paper ledger would stomp each other's
# simulated cash/positions the same way sharing one live account would.
WEATHER_PAPER_LEDGER_PATH = Path("data/kalshi/weather_paper_ledger.json")
WEATHER_LEARNING_PATH = Path("data/kalshi/weather_bot_learning.json")
WEATHER_PENDING_FILLS_PATH = Path("data/kalshi/weather_pending_fills.json")

# Real account balance_fraction budgets already sum to 100% across
# crypto/macro/mlb_totals (0.55 + 0.225 + 0.225) -- this adds a modest
# further carve-out on top rather than steal from an already-live-tuned
# bot. Soft cap in either case: each bot reads real-time balance
# independently, and a genuine over-commitment is backstopped by Kalshi's
# own insufficient_balance rejection on whichever order lands last. Kept
# deliberately small given this calibration table is currently backed
# entirely by the retired NWS settlement era (see weather_model.py).
WEATHER_BALANCE_FRACTION = 0.10


def make_kalshi_client():
    """Real client in live mode; paper-simulated wrapper otherwise (default)."""
    real = KalshiTradingClient()
    if is_paper_mode("KALSHI_WEATHER"):
        from src.trading.kalshi.paper_client import PaperKalshiTradingClient
        return PaperKalshiTradingClient(real_client=real, ledger_path=WEATHER_PAPER_LEDGER_PATH)
    return real


def main():
    # Held for the life of the process -- a second copy exits immediately
    # instead of racing this one for the same ledger/rate limits.
    _instance_lock = acquire_single_instance_lock("run_kalshi_weather")  # noqa: F841

    stop_event = threading.Event()
    log_mode_banner("run_kalshi_weather", "KALSHI_WEATHER")

    def _shutdown(sig, frame):
        logging.getLogger("launcher").info("Shutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    client = make_kalshi_client()
    grounding = KalshiWeatherGroundingEngine()
    tracker = PerformanceTracker(path=WEATHER_LEARNING_PATH)
    bot = KalshiWeatherBot(
        client=client, grounding=grounding, tracker=tracker,
        poll_interval=90, balance_fraction=WEATHER_BALANCE_FRACTION,
        pending_fills_path=WEATHER_PENDING_FILLS_PATH,
    )

    logging.getLogger("launcher").info("Kalshi weather-trading bot running. Ctrl+C to stop.")
    bot.run_forever(stop_event)

    client.close()
    logging.getLogger("launcher").info("Stopped.")


if __name__ == "__main__":
    main()

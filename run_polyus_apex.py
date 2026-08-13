"""Launch the Apex weather-trading bot for polymarket.us.

See src/trading/polymarket_us/apex/bot.py for the full design rationale --
why this only trades daily-high-temperature bucket markets, why the LLM
step (used only when ANTHROPIC_API_KEY is set) never touches the buy/pass
decision or position sizing, and how the bankroll/risk rules are enforced.

Paper mode by default, like every other bot launcher in this repo. Set
POLYMARKET_US_APEX_PAPER_TRADING=false to go live -- deliberately a separate
flag from POLYMARKET_US_PAPER_TRADING (the ladder bot's flag): the two bots
share one live account, so arming one must never silently arm the other.

Prerequisite: scripts/backfill_weather_history.py must have been run at
least once (and re-run periodically to stay current) -- Apex refuses to
trade on stale or missing calibration history rather than fall back to an
ungrounded guess.

Usage
-----
    python run_polyus_apex.py
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
    _LOG_DIR / "run_polyus_apex.log", maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8",
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
from src.trading.mode import is_paper_mode, log_mode_banner
from src.trading.polymarket_us.apex.bot import ApexBot
from src.trading.polymarket_us.apex.memory import ApexMemory
from src.trading.polymarket_us.apex.weather_model import WeatherGroundingEngine
from src.trading.polymarket_us.trading_client import PolymarketUSTradingClient

# Deliberately not the ladder bot's paper_client.DEFAULT_LEDGER_PATH -- the
# two bots would otherwise both write data/polymarket_us/paper_ledger.json
# and stomp each other's simulated cash/positions.
APEX_PAPER_LEDGER_PATH = Path("data/polymarket_us/apex_paper_ledger.json")

# Apex's OWN bankroll/trade ledger also needs a separate file per mode --
# without this, switching POLYMARKET_US_APEX_PAPER_TRADING between runs
# would mix simulated and real trade history and bankroll into one file.
APEX_MEMORY_PAPER_PATH = Path("data/polymarket_us/apex_memory_paper.json")
APEX_MEMORY_LIVE_PATH = Path("data/polymarket_us/apex_memory_live.json")


def make_polymarket_us_client():
    """Real client in live mode; paper-simulated wrapper otherwise (default)."""
    real = PolymarketUSTradingClient()
    if is_paper_mode("POLYMARKET_US_APEX"):
        from src.trading.polymarket_us.paper_client import PaperPolymarketUSTradingClient
        return PaperPolymarketUSTradingClient(real_client=real, ledger_path=APEX_PAPER_LEDGER_PATH)
    return real


def make_apex_memory() -> ApexMemory:
    path = APEX_MEMORY_PAPER_PATH if is_paper_mode("POLYMARKET_US_APEX") else APEX_MEMORY_LIVE_PATH
    return ApexMemory(path=path)


def main():
    # Held for the life of the process -- a second copy exits immediately
    # instead of racing this one for the same ledger/rate limits.
    _instance_lock = acquire_single_instance_lock("run_polyus_apex")  # noqa: F841

    stop_event = threading.Event()
    log_mode_banner("run_polyus_apex", "POLYMARKET_US_APEX")

    def _shutdown(sig, frame):
        logging.getLogger("launcher").info("Shutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    client = make_polymarket_us_client()
    memory = make_apex_memory()
    grounding = WeatherGroundingEngine()
    bot = ApexBot(client=client, memory=memory, grounding=grounding, poll_interval=300)

    logging.getLogger("launcher").info("Apex weather-trading bot running. Ctrl+C to stop.")
    bot.run_forever(stop_event)

    grounding.close()
    client.close()
    logging.getLogger("launcher").info("Stopped.")


if __name__ == "__main__":
    main()

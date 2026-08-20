"""Launch the NFL-props trading bot for polymarket.us.

Split out from the shared sports-props bot on 2026-08-19 so NFL gets its
own bankroll, ledger, and scheduled task instead of blending its wins and
losses into MLB's single tracked number -- mirrors what the tennis bot
already does. Reuses SportsPropsBot unmodified (it already keys
calibration purely on (sport, stat), so an NFL-only instance is exactly
as safe as the combined one was) via allowed_sports=("nfl",), and reuses
SportsPropCalibrationEngine unmodified too, just pointed at NFL's own
quote/settlement files (nfl_prop_quote_logger.py) instead of the MLB
pair. See src/trading/polymarket_us/sports_props_bot.py for the full
design rationale shared with the MLB bot -- portfolio construction
(one open position per game), the correlated-bucket guard, and why real
trades stay gated by calibration_trustworthy until the data matures.

This account is shared with the ladder bot, Apex, MLB sports-props, and
tennis-props -- arming this bot must never silently arm any of the
others, hence its own separate paper-mode flag.

Paper mode by default, like every other bot launcher in this repo. Set
POLYMARKET_US_NFL_PROPS_PAPER_TRADING=false to go live.

Prerequisite: nfl_prop_quote_logger.py must be running (or have run
recently) to populate data/polymarket_us/nfl_prop_quotes.jsonl and
nfl_prop_settlements.jsonl -- this bot refuses to trade on stale or
missing calibration history rather than fall back to an ungrounded guess.

Usage
-----
    python run_polyus_nfl_props.py
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
    _LOG_DIR / "run_polyus_nfl_props.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
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
from src.trading.polymarket_us.apex.memory import ApexMemory
from src.trading.polymarket_us.nfl_prop_quote_logger import QUOTES_PATH as NFL_QUOTES_PATH
from src.trading.polymarket_us.nfl_prop_quote_logger import SETTLEMENTS_PATH as NFL_SETTLEMENTS_PATH
from src.trading.polymarket_us.sports_prop_calibration import SportsPropCalibrationEngine
from src.trading.polymarket_us.sports_props_bot import SportsPropsBot
from src.trading.polymarket_us.trading_client import PolymarketUSTradingClient

# Deliberately not any other bot's ledger paths -- five bots sharing one
# paper ledger file would stomp each other's simulated cash/positions the
# same way sharing one live account would.
NFL_PROPS_PAPER_LEDGER_PATH = Path("data/polymarket_us/nfl_props_paper_ledger.json")

NFL_PROPS_MEMORY_PAPER_PATH = Path("data/polymarket_us/nfl_props_memory_paper.json")
NFL_PROPS_MEMORY_LIVE_PATH = Path("data/polymarket_us/nfl_props_memory_live.json")


def make_polymarket_us_client():
    """Real client in live mode; paper-simulated wrapper otherwise (default)."""
    real = PolymarketUSTradingClient()
    if is_paper_mode("POLYMARKET_US_NFL_PROPS"):
        from src.trading.polymarket_us.paper_client import PaperPolymarketUSTradingClient

        return PaperPolymarketUSTradingClient(real_client=real, ledger_path=NFL_PROPS_PAPER_LEDGER_PATH)
    return real


def make_nfl_props_memory() -> ApexMemory:
    path = NFL_PROPS_MEMORY_PAPER_PATH if is_paper_mode("POLYMARKET_US_NFL_PROPS") else NFL_PROPS_MEMORY_LIVE_PATH
    return ApexMemory(path=path, label="nfl_props")


def main():
    # Held for the life of the process -- a second copy exits immediately
    # instead of racing this one for the same ledger/rate limits.
    _instance_lock = acquire_single_instance_lock("run_polyus_nfl_props")  # noqa: F841

    stop_event = threading.Event()
    log_mode_banner("run_polyus_nfl_props", "POLYMARKET_US_NFL_PROPS")

    def _shutdown(sig, frame):
        logging.getLogger("launcher").info("Shutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    client = make_polymarket_us_client()
    memory = make_nfl_props_memory()
    calibration = SportsPropCalibrationEngine(quotes_path=NFL_QUOTES_PATH, settlements_path=NFL_SETTLEMENTS_PATH)
    bot = SportsPropsBot(
        client=client, memory=memory, calibration=calibration, poll_interval=300, allowed_sports=("nfl",)
    )

    logging.getLogger("launcher").info("NFL-props trading bot running. Ctrl+C to stop.")
    bot.run_forever(stop_event)

    client.close()
    logging.getLogger("launcher").info("Stopped.")


if __name__ == "__main__":
    main()

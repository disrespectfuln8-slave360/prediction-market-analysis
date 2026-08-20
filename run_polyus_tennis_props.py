"""Launch the tennis-props trading bot for polymarket.us.

See src/trading/polymarket_us/tennis_props_bot.py for the full design
rationale -- why this only trades asc-*/tsc-* tennis TOTALS markets (never
spreads, see that module's docstring for the real, unresolved sign-
convention finding 2026-08-18), why it's grounded in the market's own
historical price-level accuracy rather than an independent forecast (there
isn't one), and why real trades stay gated by TennisPropCalibrationEngine's
own trustworthiness check until the underlying data matures.

Paper mode by default, like every other bot launcher in this repo. Set
POLYMARKET_US_TENNIS_PROPS_PAPER_TRADING=false to go live -- deliberately a
separate flag from every other bot's own flag on this account (ladder,
Apex, sports-props): four bots now share one live account, so arming one
must never silently arm the others.

Prerequisite: tennis_prop_quote_logger.py must be running (or have run
recently) to populate data/polymarket_us/tennis_prop_quotes.jsonl and
tennis_prop_settlements.jsonl -- this bot refuses to trade on stale or
missing calibration history rather than fall back to an ungrounded guess.

Usage
-----
    python run_polyus_tennis_props.py
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
    _LOG_DIR / "run_polyus_tennis_props.log",
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
from src.trading.polymarket_us.tennis_prop_calibration import TennisPropCalibrationEngine
from src.trading.polymarket_us.tennis_props_bot import TennisPropsBot
from src.trading.polymarket_us.trading_client import PolymarketUSTradingClient

# Deliberately not any other bot's ledger paths -- four bots sharing one
# paper ledger file would stomp each other's simulated cash/positions the
# same way sharing one live account would.
TENNIS_PROPS_PAPER_LEDGER_PATH = Path("data/polymarket_us/tennis_props_paper_ledger.json")

TENNIS_PROPS_MEMORY_PAPER_PATH = Path("data/polymarket_us/tennis_props_memory_paper.json")
TENNIS_PROPS_MEMORY_LIVE_PATH = Path("data/polymarket_us/tennis_props_memory_live.json")


def make_polymarket_us_client():
    """Real client in live mode; paper-simulated wrapper otherwise (default)."""
    real = PolymarketUSTradingClient()
    if is_paper_mode("POLYMARKET_US_TENNIS_PROPS"):
        from src.trading.polymarket_us.paper_client import PaperPolymarketUSTradingClient

        return PaperPolymarketUSTradingClient(real_client=real, ledger_path=TENNIS_PROPS_PAPER_LEDGER_PATH)
    return real


def make_tennis_props_memory() -> ApexMemory:
    path = (
        TENNIS_PROPS_MEMORY_PAPER_PATH if is_paper_mode("POLYMARKET_US_TENNIS_PROPS") else TENNIS_PROPS_MEMORY_LIVE_PATH
    )
    return ApexMemory(path=path, label="tennis_props")


def main():
    # Held for the life of the process -- a second copy exits immediately
    # instead of racing this one for the same ledger/rate limits.
    _instance_lock = acquire_single_instance_lock("run_polyus_tennis_props")  # noqa: F841

    stop_event = threading.Event()
    log_mode_banner("run_polyus_tennis_props", "POLYMARKET_US_TENNIS_PROPS")

    def _shutdown(sig, frame):
        logging.getLogger("launcher").info("Shutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    client = make_polymarket_us_client()
    memory = make_tennis_props_memory()
    calibration = TennisPropCalibrationEngine()
    bot = TennisPropsBot(client=client, memory=memory, calibration=calibration, poll_interval=300)

    logging.getLogger("launcher").info("Tennis-props trading bot running. Ctrl+C to stop.")
    bot.run_forever(stop_event)

    client.close()
    logging.getLogger("launcher").info("Stopped.")


if __name__ == "__main__":
    main()

"""Launch the sports-props trading bot for polymarket.us -- MLB only.

See src/trading/polymarket_us/sports_props_bot.py for the full design
rationale -- why this only trades astatc-mlb-* player-prop markets,
why it's grounded in the market's own historical price-level accuracy
rather than an independent forecast (there isn't one), and why real
trades stay gated by SportsPropCalibrationEngine's own trustworthiness
check until the underlying data matures.

NFL runs as its own separate bot (run_polyus_nfl_props.py) as of
2026-08-19 -- own bankroll, own ledger, own scheduled task -- rather than
sharing this one's bankroll the way it briefly did. The two share this
same live account (like the ladder bot and Apex also do), which is why
SportsPropsBot's own live-position reconciliation is scoped to
allowed_sports: a real NFL position on the account is real and tracked,
just by the other process, not this one.

Paper mode by default, like every other bot launcher in this repo. Set
POLYMARKET_US_SPORTS_PROPS_PAPER_TRADING=false to go live -- deliberately
a separate flag from POLYMARKET_US_PAPER_TRADING (the ladder bot's) and
POLYMARKET_US_APEX_PAPER_TRADING (Apex's): all three bots share one live
account, so arming one must never silently arm the others.

Prerequisite: sports_prop_quote_logger.py must be running (or have run
recently) to populate data/polymarket_us/sports_prop_quotes.jsonl and
sports_prop_settlements.jsonl -- this bot refuses to trade on stale or
missing calibration history rather than fall back to an ungrounded guess.

The calibration_trustworthy gate is the real safety net here, not the
paper/live flag: as of 2026-08-15 the collected data is only hours old,
nowhere near enough for any decile to be trusted for real money yet. This
launcher defaults to paper mode regardless, on top of that gate, the same
belt-and-suspenders posture as every other bot in this repo.

Usage
-----
    python run_polyus_sports_props.py
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
    _LOG_DIR / "run_polyus_sports_props.log", maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8",
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
from src.trading.polymarket_us.sports_prop_calibration import SportsPropCalibrationEngine
from src.trading.polymarket_us.sports_props_bot import SportsPropsBot
from src.trading.polymarket_us.trading_client import PolymarketUSTradingClient

# Deliberately not the ladder bot's or Apex's ledger paths -- three bots
# sharing one paper ledger file would stomp each other's simulated
# cash/positions the same way sharing one live account would.
SPORTS_PROPS_PAPER_LEDGER_PATH = Path("data/polymarket_us/sports_props_paper_ledger.json")

# This bot's OWN bankroll/trade ledger also needs a separate file per mode
# -- without this, switching POLYMARKET_US_SPORTS_PROPS_PAPER_TRADING
# between runs would mix simulated and real trade history and bankroll
# into one file.
SPORTS_PROPS_MEMORY_PAPER_PATH = Path("data/polymarket_us/sports_props_memory_paper.json")
SPORTS_PROPS_MEMORY_LIVE_PATH = Path("data/polymarket_us/sports_props_memory_live.json")


def make_polymarket_us_client():
    """Real client in live mode; paper-simulated wrapper otherwise (default)."""
    real = PolymarketUSTradingClient()
    if is_paper_mode("POLYMARKET_US_SPORTS_PROPS"):
        from src.trading.polymarket_us.paper_client import PaperPolymarketUSTradingClient
        return PaperPolymarketUSTradingClient(real_client=real, ledger_path=SPORTS_PROPS_PAPER_LEDGER_PATH)
    return real


def make_sports_props_memory() -> ApexMemory:
    path = SPORTS_PROPS_MEMORY_PAPER_PATH if is_paper_mode("POLYMARKET_US_SPORTS_PROPS") else SPORTS_PROPS_MEMORY_LIVE_PATH
    return ApexMemory(path=path, label="sports_props")


def main():
    # Held for the life of the process -- a second copy exits immediately
    # instead of racing this one for the same ledger/rate limits.
    _instance_lock = acquire_single_instance_lock("run_polyus_sports_props")  # noqa: F841

    stop_event = threading.Event()
    log_mode_banner("run_polyus_sports_props", "POLYMARKET_US_SPORTS_PROPS")

    def _shutdown(sig, frame):
        logging.getLogger("launcher").info("Shutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    client = make_polymarket_us_client()
    memory = make_sports_props_memory()
    calibration = SportsPropCalibrationEngine()
    bot = SportsPropsBot(client=client, memory=memory, calibration=calibration, poll_interval=300, allowed_sports=("mlb",))

    logging.getLogger("launcher").info("Sports-props trading bot running. Ctrl+C to stop.")
    bot.run_forever(stop_event)

    client.close()
    logging.getLogger("launcher").info("Stopped.")


if __name__ == "__main__":
    main()

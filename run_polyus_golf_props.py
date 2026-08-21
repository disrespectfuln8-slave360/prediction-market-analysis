"""Launch the golf-props trading bot for polymarket.us.

Scoped to PGA Tour round-leader markets only (tec-pga-*-r1l/r2l/r3l-*) --
see src/trading/polymarket_us/golf_prop_quote_logger.py's docstring for the
real market survey behind that scope: outright-winner markets are
structurally longshot (median 4c across the field, will never clear this
account's 30-70c no-longshot gate), and LPGA/LIV markets were confirmed
illiquid (placeholder quotes, zero real shares traded).

Reuses SportsPropsBot unmodified via allowed_sports=("golf",) -- the same
pattern NFL-props uses -- and SportsPropCalibrationEngine pointed at golf's
own quote/settlement files. "matchup" is the tournament name standing in
for a 2-team matchup; the one-position-per-(sport,matchup,date) guard and
the correlated-bucket guard both apply the same way they do for MLB/NFL's
real matchups, since a single tournament-round is exactly the kind of
shared, correlated event those guards exist to protect against.

This account is shared with the ladder bot, Apex, MLB sports-props,
NFL-props, and tennis-props -- arming this bot must never silently arm any
of the others, hence its own separate paper-mode flag.

No independent settlement-verification module exists for golf yet (unlike
MLB/NFL's real ESPN box-score cross-check) -- this trusts the exchange's
own settlement outcome. A real, known gap, not an oversight.

Paper mode by default, like every other bot launcher in this repo. Set
POLYMARKET_US_GOLF_PROPS_PAPER_TRADING=false to go live.

Prerequisite: golf_prop_quote_logger.py must be running (or have run
recently) to populate data/polymarket_us/golf_prop_quotes.jsonl and
golf_prop_settlements.jsonl -- this bot refuses to trade on stale or
missing calibration history rather than fall back to an ungrounded guess.

Usage
-----
    python run_polyus_golf_props.py
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
    _LOG_DIR / "run_polyus_golf_props.log",
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
from src.trading.polymarket_us.golf_prop_quote_logger import QUOTES_PATH as GOLF_QUOTES_PATH
from src.trading.polymarket_us.golf_prop_quote_logger import SETTLEMENTS_PATH as GOLF_SETTLEMENTS_PATH
from src.trading.polymarket_us.sports_prop_calibration import SportsPropCalibrationEngine
from src.trading.polymarket_us.sports_props_bot import SportsPropsBot
from src.trading.polymarket_us.trading_client import PolymarketUSTradingClient

# Deliberately not any other bot's ledger paths -- six bots sharing one
# paper ledger file would stomp each other's simulated cash/positions the
# same way sharing one live account would.
GOLF_PROPS_PAPER_LEDGER_PATH = Path("data/polymarket_us/golf_props_paper_ledger.json")

GOLF_PROPS_MEMORY_PAPER_PATH = Path("data/polymarket_us/golf_props_memory_paper.json")
GOLF_PROPS_MEMORY_LIVE_PATH = Path("data/polymarket_us/golf_props_memory_live.json")


def make_polymarket_us_client():
    """Real client in live mode; paper-simulated wrapper otherwise (default)."""
    real = PolymarketUSTradingClient()
    if is_paper_mode("POLYMARKET_US_GOLF_PROPS"):
        from src.trading.polymarket_us.paper_client import PaperPolymarketUSTradingClient

        return PaperPolymarketUSTradingClient(real_client=real, ledger_path=GOLF_PROPS_PAPER_LEDGER_PATH)
    return real


def make_golf_props_memory() -> ApexMemory:
    path = GOLF_PROPS_MEMORY_PAPER_PATH if is_paper_mode("POLYMARKET_US_GOLF_PROPS") else GOLF_PROPS_MEMORY_LIVE_PATH
    return ApexMemory(path=path, label="golf_props")


def main():
    # Held for the life of the process -- a second copy exits immediately
    # instead of racing this one for the same ledger/rate limits.
    _instance_lock = acquire_single_instance_lock("run_polyus_golf_props")  # noqa: F841

    stop_event = threading.Event()
    log_mode_banner("run_polyus_golf_props", "POLYMARKET_US_GOLF_PROPS")

    def _shutdown(sig, frame):
        logging.getLogger("launcher").info("Shutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    client = make_polymarket_us_client()
    memory = make_golf_props_memory()
    calibration = SportsPropCalibrationEngine(quotes_path=GOLF_QUOTES_PATH, settlements_path=GOLF_SETTLEMENTS_PATH)
    bot = SportsPropsBot(
        client=client, memory=memory, calibration=calibration, poll_interval=300, allowed_sports=("golf",)
    )

    logging.getLogger("launcher").info("Golf-props trading bot running. Ctrl+C to stop.")
    bot.run_forever(stop_event)

    client.close()
    logging.getLogger("launcher").info("Stopped.")


if __name__ == "__main__":
    main()

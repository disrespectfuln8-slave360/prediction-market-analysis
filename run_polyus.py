"""Launch the polymarket.us ladder-arbitrage bot.

Bot design
----------
polymarket.us (the regulated US product, Key ID + Secret Key auth — a
different product from the international polymarket.com CLOB that
run_poly.py trades) has no crypto/short-duration markets: scanning all
~9,500 active markets found none of Kalshi's KXBTC-style recurring
brackets, so the convergence-bot approach in run_poly.py/run_all.py
doesn't apply here.

Instead this trades a structural feature specific to this catalog: many
player-stat markets come in threshold "ladders" (e.g. `...-gte1`,
`...-gte2`, `...-gte3` for "1+ goals", "2+ goals", "3+ goals"). Since
gte(n) implies gte(n-1), price(gte(n)) must be non-increasing in n — a
math constraint, not a probability estimate, so (unlike the removed
crypto/macro mispricing bots) it needs no historical calibration to
trade safely. See src/trading/polymarket_us/strategy.py for the payoff
proof. Checked live prices: 0 violations found at scan time (rare but
real when they occur — the bot just watches continuously).

Usage
-----
    python run_polyus.py
"""
from __future__ import annotations

import logging
import signal
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# The process writes its own log rather than relying on a shell redirect.
# The scheduled task used to run this through `cmd.exe /c ... >> run_polyus.log`,
# and stopping that task reaped only the cmd.exe wrapper — the python
# grandchild survived, kept the single-instance lock, and left Task Scheduler
# reporting "Ready" while the bot was still trading. Owning the log here means
# the task can invoke python directly, so stopping the task stops the bot.
_LOG_DIR = Path(__file__).resolve().parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Rotating, because this runs unattended for days at a time and an unbounded
# log is a slow-motion disk problem (run_all.log reached 15 MB).
_file_handler = RotatingFileHandler(
    _LOG_DIR / "run_polyus.log", maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8",
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
from src.trading.polymarket_us.bot import LadderArbBot
from src.trading.polymarket_us.trading_client import PolymarketUSTradingClient


def make_polymarket_us_client():
    """Real client in live mode; paper-simulated wrapper otherwise (default)."""
    real = PolymarketUSTradingClient()
    if is_paper_mode("POLYMARKET_US"):
        from src.trading.polymarket_us.paper_client import PaperPolymarketUSTradingClient
        return PaperPolymarketUSTradingClient(real_client=real)
    return real


def main():
    # Held for the life of the process — a second copy exits immediately
    # instead of racing this one for the same ledger/rate limits.
    _instance_lock = acquire_single_instance_lock("run_polyus")  # noqa: F841

    stop_event = threading.Event()
    log_mode_banner("run_polyus", "POLYMARKET_US")

    def _shutdown(sig, frame):
        logging.getLogger("launcher").info("Shutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    client = make_polymarket_us_client()
    bot = LadderArbBot(client=client, poll_interval=60, max_open_legs=20)

    logging.getLogger("launcher").info("Ladder-arbitrage bot running. Ctrl+C to stop.")
    bot.run_forever(stop_event)

    client.close()
    logging.getLogger("launcher").info("Stopped.")


if __name__ == "__main__":
    main()

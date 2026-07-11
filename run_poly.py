"""Launch the Polymarket convergence bot and a background trade indexer.

Bot design
----------
The crypto/macro mispricing bots (mirrored from run_all.py's Kalshi design)
were removed here — checked the live Polymarket catalog and found none of
its crypto/macro/politics-tagged markets are short-duration recurring
brackets like Kalshi's KXBTC series (nearest close was 82 days out, median
~175 days). Widening the close-time window to fit made the strategy return
"opportunities" with implausible 40-60pp edges: the win-rate calibration
table is built from ~2,100 resolved markets of every type mixed together
(sports, politics, weather, ...), which reflects short-duration mispricing,
not the very different kind of uncertainty in a 6-month-out speculative
forecast. Applying it there isn't a real edge, it's a mismatched model.

The convergence bot survives: it doesn't use that price-based win-rate
table at all (live spot price + a volatility diffusion model instead), so
it isn't affected by this issue. It just won't fire until Polymarket lists
short-duration crypto threshold markets, since none currently exist.

A single indexer thread refreshes local trade data every 30 minutes.

Usage
-----
    python run_poly.py
"""
from __future__ import annotations

import logging
import signal
import threading

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(name)-20s]  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Every single outbound HTTP request logs at INFO by default — fine for a
# short foreground run, but unbounded noise for a process meant to run for
# days unattended. Bot decisions/fills/errors are logged separately at the
# application level and aren't affected by this.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from src.common.event_exposure import EventExposureTracker
from src.common.single_instance import acquire_single_instance_lock
from src.trading.mode import is_paper_mode, log_mode_banner
from src.trading.polymarket.bot import run_indexer_loop
from src.trading.polymarket.convergence_bot import ConvergenceBot
from src.trading.polymarket.trading_client import PolymarketTradingClient

# Max total spend on any single correlated real-world event per day, across
# every bot sharing the tracker. 3x a single max bet: enough room for a
# couple of legitimate related bets without letting correlated risk stack
# unbounded.
MAX_EVENT_EXPOSURE_CENTS = 1500

CRYPTO_TAGS = ["crypto"]


def make_polymarket_client() -> PolymarketTradingClient:
    """Real client in live mode; paper-simulated wrapper otherwise (default)."""
    real = PolymarketTradingClient()
    if is_paper_mode("POLYMARKET"):
        from src.trading.polymarket.paper_client import PaperPolymarketTradingClient
        return PaperPolymarketTradingClient(real_client=real)
    return real


def main():
    # Held for the life of the process — a second copy exits immediately
    # instead of racing this one for the same ledger/rate limits.
    _instance_lock = acquire_single_instance_lock("run_poly")  # noqa: F841

    stop_event = threading.Event()
    log_mode_banner("run_poly", "POLYMARKET")

    def _shutdown(sig, frame):
        logging.getLogger("launcher").info("Shutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    poly_client = make_polymarket_client()

    # Caps total daily spend on any one correlated real-world event.
    event_tracker = EventExposureTracker(MAX_EVENT_EXPOSURE_CENTS)

    # Shared indexer — keeps local trade data fresh.
    indexer_thread = threading.Thread(
        target=run_indexer_loop,
        args=(stop_event, 1800),
        daemon=True,
        name="indexer",
    )
    indexer_thread.start()

    # Convergence bot: bets crypto threshold markets whose outcome is >=93%
    # certain from live spot prices.
    def _run_convergence(stop_event):
        bot = ConvergenceBot(
            client=poly_client,
            tag_slugs=CRYPTO_TAGS,
            min_prob=0.93,
            max_entry_price=92,
            min_edge_pp=3.0,
            min_close_secs=90,
            max_close_secs=600,
            max_spend_cents=500,
            poll_interval=20,
            event_tracker=event_tracker,
        )
        bot.run(stop_event)

    conv_thread = threading.Thread(
        target=_run_convergence, args=(stop_event,), name="convergence", daemon=True
    )
    conv_thread.start()

    logging.getLogger("launcher").info(
        "1 convergence bot + 1 indexer running. Ctrl+C to stop.",
    )

    conv_thread.join()

    poly_client.close()
    logging.getLogger("launcher").info("All stopped.")


if __name__ == "__main__":
    main()

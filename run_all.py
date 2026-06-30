"""Launch multiple trading bots in parallel threads.

Each bot targets a different market segment and runs concurrently.
All bots share the same strategy parameters but have separate API clients
and rate limiters so they don't block each other.

Usage:
    uv run run_all.py
"""

from __future__ import annotations

import logging
import os
import signal
import threading

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(name)-20s]  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from src.trading.kalshi.bot import TradingBot
from src.trading.kalshi.strategy import MispricingStrategy
from src.trading.kalshi.trading_client import KalshiTradingClient

# ── Bot definitions ──────────────────────────────────────────────────────────
# Each entry becomes its own thread with its own client + strategy instance.

BOTS = [
    {
        "name": "btc",
        "ticker_filter": "BTC",
        "poll_interval": 30,
        "max_open_positions": 10,
        "min_edge_pp": 5.0,
        "max_spend_cents": 500,
    },
    {
        "name": "general",
        "ticker_filter": None,   # all markets
        "poll_interval": 60,
        "max_open_positions": 20,
        "min_edge_pp": 5.0,
        "max_spend_cents": 500,
    },
]

# ─────────────────────────────────────────────────────────────────────────────


def _run_bot(cfg: dict, stop_event: threading.Event):
    logger = logging.getLogger(cfg["name"])
    strategy = MispricingStrategy(
        min_edge_pp=cfg["min_edge_pp"],
        max_spend_cents=cfg["max_spend_cents"],
        ticker_filter=cfg["ticker_filter"],
    )
    with KalshiTradingClient() as client:
        bot = TradingBot(
            client=client,
            strategy=strategy,
            poll_interval=cfg["poll_interval"],
            max_open_positions=cfg["max_open_positions"],
        )
        # Patch the bot's _running flag to respect the shared stop event
        import time
        bot._running = True
        bot._last_reload = 0.0
        logger.info("Started (poll=%ds, filter=%s)", cfg["poll_interval"], cfg["ticker_filter"] or "all")
        while not stop_event.is_set():
            try:
                bot._scan_once()
            except Exception as exc:
                logger.error("Scan error: %s", exc, exc_info=True)
            stop_event.wait(timeout=cfg["poll_interval"])
        logger.info("Stopped.")


def main():
    stop_event = threading.Event()

    def _shutdown(sig, frame):
        logging.getLogger("launcher").info("Shutting down all bots...")
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    threads = []
    for cfg in BOTS:
        t = threading.Thread(
            target=_run_bot,
            args=(cfg, stop_event),
            name=cfg["name"],
            daemon=True,
        )
        t.start()
        threads.append(t)

    logging.getLogger("launcher").info("All %d bots running. Press Ctrl+C to stop.", len(threads))

    for t in threads:
        t.join()

    logging.getLogger("launcher").info("All bots stopped.")


if __name__ == "__main__":
    main()

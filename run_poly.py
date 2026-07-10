"""Launch the Polymarket trading bots: crypto + macro mispricing bots,
a convergence bot, and a background trade indexer.

Bot design (mirrors run_all.py for Kalshi)
-------------------------------------------
crypto (15s poll, 55% of balance)
    Tags   : crypto-tagged markets
    Win rates: computed from crypto-keyword-matched historical trades

macro  (30s poll, 45% of balance)
    Tags   : politics/economy-tagged markets
    Win rates: computed from politics/economy-keyword-matched historical trades

Because Position/Order carry no tag, slot accounting is account-wide across
both bots by design (see the "DEDUP ACROSS BOTS" note in bot.py) — balance
fractions are what keep the two bots from racing each other for the same
dollars.

A single shared indexer thread refreshes local trade data every 30 minutes.

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
from src.trading.mode import is_paper_mode, log_mode_banner
from src.trading.polymarket.bot import TradingBot, run_indexer_loop
from src.trading.polymarket.convergence_bot import ConvergenceBot
from src.trading.polymarket.learning import PerformanceTracker
from src.trading.polymarket.strategy import MispricingStrategy
from src.trading.polymarket.trading_client import PolymarketTradingClient

# Max total spend on any single correlated real-world event per day, across
# every bot sharing the tracker. 3x a single max bet: enough room for a
# couple of legitimate related bets without letting correlated risk stack
# unbounded.
MAX_EVENT_EXPOSURE_CENTS = 1500


def make_polymarket_client() -> PolymarketTradingClient:
    """Real client in live mode; paper-simulated wrapper otherwise (default)."""
    real = PolymarketTradingClient()
    if is_paper_mode():
        from src.trading.polymarket.paper_client import PaperPolymarketTradingClient
        return PaperPolymarketTradingClient(real_client=real)
    return real

# ── Market universe split ──────────────────────────────────────────────────────

CRYPTO_TAGS = ["crypto"]
MACRO_TAGS = ["economy", "politics", "elections"]

# ── Bot configurations ────────────────────────────────────────────────────────

BOTS = [
    {
        "name": "crypto",
        "tag_slugs": CRYPTO_TAGS,
        "keywords": ["bitcoin", "ethereum", "solana", "dogecoin", "crypto"],
        "poll_interval": 15,
        "max_open_positions": 40,
        "balance_fraction": 0.55,
        "min_edge_pp": 0.5,
        "max_spend_cents": 500,
        "min_historical_trades": 30,
        "min_entry_price": 18,
        "max_entry_price": 55,
        "min_close_seconds": 60,
        "max_close_seconds": 3600,
    },
    {
        "name": "macro",
        "tag_slugs": MACRO_TAGS,
        "keywords": ["fed", "inflation", "election", "president", "congress", "recession"],
        "poll_interval": 30,
        "max_open_positions": 40,
        "balance_fraction": 0.45,
        "min_edge_pp": 0.5,
        "max_spend_cents": 500,
        "min_historical_trades": 30,
        "min_entry_price": 18,
        "max_entry_price": 55,
        "min_close_seconds": 60,
        "max_close_seconds": 3600,
    },
]

# ─────────────────────────────────────────────────────────────────────────────


def _run_bot(
    cfg: dict,
    stop_event: threading.Event,
    tracker: PerformanceTracker,
    client: PolymarketTradingClient,
    event_tracker: EventExposureTracker,
):
    log = logging.getLogger(cfg["name"])
    strategy = MispricingStrategy(
        keywords=cfg["keywords"],
        min_edge_pp=cfg["min_edge_pp"],
        max_spend_cents=cfg["max_spend_cents"],
        min_historical_trades=cfg["min_historical_trades"],
        min_entry_price=cfg["min_entry_price"],
        max_entry_price=cfg["max_entry_price"],
        min_close_seconds=cfg["min_close_seconds"],
        max_close_seconds=cfg["max_close_seconds"],
        tracker=tracker,
    )
    bot = TradingBot(
        client=client,
        strategy=strategy,
        poll_interval=cfg["poll_interval"],
        max_open_positions=cfg["max_open_positions"],
        balance_fraction=cfg["balance_fraction"],
        tag_slugs=cfg["tag_slugs"],
        tracker=tracker,
        event_tracker=event_tracker,
    )
    log.info(
        "Started — tags=%s  poll=%ds  balance_fraction=%.0f%%",
        cfg["tag_slugs"],
        cfg["poll_interval"],
        cfg["balance_fraction"] * 100,
    )
    while not stop_event.is_set():
        try:
            bot._scan_once()
        except Exception as exc:
            log.error("Scan error: %s", exc, exc_info=True)
        stop_event.wait(timeout=cfg["poll_interval"])
    log.info("Stopped.")


def main():
    stop_event = threading.Event()
    log_mode_banner("run_poly")

    def _shutdown(sig, frame):
        logging.getLogger("launcher").info("Shutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Shared learning tracker — persists across restarts, shared by both bots.
    tracker = PerformanceTracker()

    # One shared Polymarket client for every thread in this process. In
    # paper mode this matters a lot: each PaperPolymarketTradingClient keeps
    # its own in-memory ledger, so giving every thread a separate instance
    # means their balance/position writes to the same ledger file race and
    # clobber each other. One shared instance (with its own internal lock)
    # is the single source of truth, exactly like the real account is for
    # the real client.
    poly_client = make_polymarket_client()

    # Caps total daily spend on any one correlated real-world event, shared
    # across crypto/macro/convergence.
    event_tracker = EventExposureTracker(MAX_EVENT_EXPOSURE_CENTS)

    # Shared indexer — keeps local trade data fresh for both bots.
    indexer_thread = threading.Thread(
        target=run_indexer_loop,
        args=(stop_event, 1800),
        daemon=True,
        name="indexer",
    )
    indexer_thread.start()

    threads = []
    for cfg in BOTS:
        t = threading.Thread(
            target=_run_bot,
            args=(cfg, stop_event, tracker, poly_client, event_tracker),
            name=cfg["name"],
            daemon=True,
        )
        t.start()
        threads.append(t)

    # Convergence bot: bets crypto threshold markets whose outcome is >=93%
    # certain from live spot prices. Independent signal source, shares the
    # same Polymarket client/ledger.
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
    threads.append(conv_thread)

    logging.getLogger("launcher").info(
        "%d bots + 1 convergence + 1 indexer running. Ctrl+C to stop.",
        len(BOTS),
    )

    for t in threads:
        t.join()

    poly_client.close()
    logging.getLogger("launcher").info("All stopped.")


if __name__ == "__main__":
    main()

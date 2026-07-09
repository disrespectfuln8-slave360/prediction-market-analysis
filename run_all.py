"""Launch two non-racing trading bots with disjoint market universes.

Bot design
----------
crypto (30s poll, 55% of balance)
    Markets : KXBTC, KXETH, KXSOL, KXDOGE
    Win rates: computed from crypto-only historical trades

macro  (90s poll, 45% of balance)
    Markets : equity indices + macro + politics
    Win rates: computed from macro/politics historical trades

Because each bot only sees its own market series, they can never compete for
the same ticker.  Balance fractions prevent them from racing for the same
dollars even if both scan at the same moment.

A single shared indexer thread refreshes local trade data every 30 minutes so
win rates improve continuously without duplication.

Usage
-----
    python run_all.py
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

from src.trading.kalshi.bot import TradingBot, run_indexer_loop
from src.trading.kalshi.convergence_bot import ConvergenceBot
from src.trading.kalshi.learning import PerformanceTracker
from src.trading.kalshi.strategy import MispricingStrategy
from src.trading.kalshi.trading_client import KalshiTradingClient
from src.trading.mode import is_paper_mode, log_mode_banner
from src.trading.polymarket.arb_scanner import run_arb_loop
from src.trading.polymarket.trading_client import PolymarketTradingClient


def make_kalshi_client() -> KalshiTradingClient:
    """Real client in live mode; paper-simulated wrapper otherwise (default)."""
    real = KalshiTradingClient()
    if is_paper_mode():
        from src.trading.kalshi.paper_client import PaperKalshiTradingClient
        return PaperKalshiTradingClient(real_client=real)
    return real


def make_polymarket_client() -> PolymarketTradingClient:
    """Real client in live mode; paper-simulated wrapper otherwise (default)."""
    real = PolymarketTradingClient()
    if is_paper_mode():
        from src.trading.polymarket.paper_client import PaperPolymarketTradingClient
        return PaperPolymarketTradingClient(real_client=real)
    return real

# ── Sports series watched by the arb scanner (Kalshi ↔ Polymarket) ───────────
# The arb scanner owns these series exclusively. They are intentionally excluded
# from MACRO_SERIES below so the macro bot and arb scanner never compete for
# the same ticker.
ARB_SERIES = ["KXATPMATCH", "KXWTAMATCH", "KXMLBGAME", "KXUFCFIGHT"]

# ── Market universe split (must be disjoint) ──────────────────────────────────

CRYPTO_SERIES = [
    "KXBTC", "KXETH", "KXSOL", "KXDOGE",            # hourly price brackets
    "KXBTC15M", "KXETH15M", "KXSOL15M",             # 15-min price brackets (4x more opportunities/hr)
    "KXBTCD",                                         # daily BTC price brackets
]

MACRO_SERIES = [
    "KXINX", "KXNDX", "KXSPY", "KXDJI",            # equity indices
    "INX", "NDX", "NASDAQ",                          # additional index series
    "FED", "FOMC", "CPI", "JOBS", "GDP", "UNRATE",  # macro
    "PRES", "CONGRESS", "SENATE", "HOUSE",           # politics
    # Note: KXMLBGAME, KXATPMATCH, KXWTAMATCH, KXUFCFIGHT are owned by the
    # arb scanner (ARB_SERIES) and must not appear here to avoid competing orders.
    "KXMLBTOTAL", "KXMLBSPREAD",                    # MLB totals/spreads (not in ARB_SERIES)
]

# ── Bot configurations ────────────────────────────────────────────────────────

BOTS = [
    {
        "name": "crypto",
        "series_list": CRYPTO_SERIES,
        "ticker_prefixes": CRYPTO_SERIES,
        "poll_interval": 15,                 # scan every 15s — catch fast-moving opportunities
        "max_open_positions": 40,            # allow deep position stack
        "balance_fraction": 0.55,
        "min_edge_pp": 0.5,
        "max_spend_cents": 500,              # up to $5 per trade
        "max_spread": 15,
        "min_historical_trades": 30,
        "min_entry_price": 18,
        "max_entry_price": 55,
        "min_close_seconds": 60,
        "max_close_seconds": 3600,           # only bet on markets closing within 1 hour
    },
    {
        "name": "macro",
        "series_list": MACRO_SERIES,
        "ticker_prefixes": MACRO_SERIES,
        "poll_interval": 30,
        "max_open_positions": 40,
        "balance_fraction": 0.45,
        "min_edge_pp": 0.5,
        "max_spend_cents": 500,              # up to $5 per trade
        "max_spread": 15,
        "min_historical_trades": 30,
        "min_entry_price": 18,
        "max_entry_price": 55,
        "min_close_seconds": 60,
        "max_close_seconds": 3600,           # only bet on markets closing within 1 hour
    },
]

# ─────────────────────────────────────────────────────────────────────────────


def _run_bot(cfg: dict, stop_event: threading.Event, tracker: PerformanceTracker, client: KalshiTradingClient):
    log = logging.getLogger(cfg["name"])
    strategy = MispricingStrategy(
        ticker_prefixes=cfg["ticker_prefixes"],
        min_edge_pp=cfg["min_edge_pp"],
        max_spend_cents=cfg["max_spend_cents"],
        max_spread=cfg["max_spread"],
        min_historical_trades=cfg["min_historical_trades"],
        min_close_seconds=cfg["min_close_seconds"],
        max_close_seconds=cfg["max_close_seconds"],
        min_entry_price=cfg["min_entry_price"],
        max_entry_price=cfg["max_entry_price"],
        tracker=tracker,
    )
    bot = TradingBot(
        client=client,
        strategy=strategy,
        poll_interval=cfg["poll_interval"],
        max_open_positions=cfg["max_open_positions"],
        balance_fraction=cfg["balance_fraction"],
        series_list=cfg["series_list"],
        tracker=tracker,
    )
    log.info(
        "Started — series=%s  poll=%ds  balance_fraction=%.0f%%",
        cfg["name"],
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
    log_mode_banner("run_all")

    def _shutdown(sig, frame):
        logging.getLogger("launcher").info("Shutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Shared learning tracker — persists across restarts, shared by both bots.
    tracker = PerformanceTracker()

    # One shared Kalshi client for every thread in this process (crypto bot,
    # macro bot, convergence bot, arb bot). In paper mode this matters a lot:
    # each PaperKalshiTradingClient keeps its own in-memory ledger, so giving
    # every thread a separate instance means their balance/position writes to
    # the same ledger file race and clobber each other. One shared instance
    # (with its own internal lock) is the single source of truth, exactly
    # like the real Kalshi account is for the real client. It also naturally
    # centralizes live-mode rate-limiting across all four bots instead of
    # each one throttling independently against the same account.
    kalshi_client = make_kalshi_client()
    poly_client = make_polymarket_client()

    # One shared indexer — keeps local trade data fresh for both bots.
    # Only scan crypto+macro series to avoid iterating over 9M+ esports markets.
    all_series = CRYPTO_SERIES + MACRO_SERIES
    indexer = threading.Thread(
        target=run_indexer_loop,
        args=(stop_event, 1800, all_series),
        daemon=True,
        name="indexer",
    )
    indexer.start()

    threads = []
    for cfg in BOTS:
        t = threading.Thread(
            target=_run_bot,
            args=(cfg, stop_event, tracker, kalshi_client),
            name=cfg["name"],
            daemon=True,
        )
        t.start()
        threads.append(t)

    # Cross-platform arb scanner: Kalshi ↔ Polymarket sports markets.
    def _run_arb(stop_event):
        log = logging.getLogger("arb")
        log.info("Arb scanner started — watching %s", ARB_SERIES)
        run_arb_loop(
            kalshi_client, stop_event, ARB_SERIES,
            execute_kalshi=True, poly_trading_client=poly_client,
        )

    arb_thread = threading.Thread(
        target=_run_arb, args=(stop_event,), name="arb", daemon=True
    )
    arb_thread.start()
    threads.append(arb_thread)

    # Convergence bot: bets crypto brackets whose outcome is ≥93% certain
    # from live spot prices. Independent signal source from the other bots,
    # shares the same Kalshi client/ledger.
    def _run_convergence(stop_event):
        bot = ConvergenceBot(
            client=kalshi_client,
            min_prob=0.93,
            max_entry_price=92,
            min_edge_pp=3.0,
            min_close_secs=90,
            max_close_secs=600,
            max_spend_cents=500,
            poll_interval=20,
        )
        bot.run(stop_event)

    conv_thread = threading.Thread(
        target=_run_convergence, args=(stop_event,), name="convergence", daemon=True
    )
    conv_thread.start()
    threads.append(conv_thread)

    logging.getLogger("launcher").info(
        "%d bots + 1 convergence + 1 indexer + 1 arb scanner running. Ctrl+C to stop.",
        len(BOTS),
    )

    for t in threads:
        t.join()

    kalshi_client.close()
    poly_client.close()
    logging.getLogger("launcher").info("All stopped.")


if __name__ == "__main__":
    main()

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
from pathlib import Path

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
from src.common.reservation import TickerReservation
from src.common.single_instance import acquire_single_instance_lock
from src.indexers.kalshi.market_snapshots import run_snapshot_loop
from src.trading.kalshi.bot import TradingBot, run_indexer_loop
from src.trading.kalshi.convergence_bot import ConvergenceBot
from src.trading.kalshi.learning import PerformanceTracker
from src.trading.kalshi.strategy import MispricingStrategy
from src.trading.kalshi.trading_client import KalshiTradingClient
from src.trading.kalshi.weather_model import CITY_STATIONS as WEATHER_SERIES
from src.trading.mode import is_paper_mode, log_mode_banner
from src.trading.polymarket.arb_scanner import run_arb_loop
from src.trading.polymarket.trading_client import PolymarketTradingClient

# Max total spend on any single correlated real-world event per day, across
# every bot sharing the tracker — e.g. one BTC close hour's price brackets,
# one game's markets. Originally set as 3x the $5 max bet; now 15x the $1
# max bet since it wasn't rescaled with the per-trade size change — still
# a real cap, just one that now allows more individual bets to stack on the
# same event before it kicks in.
MAX_EVENT_EXPOSURE_CENTS = 1500


def make_kalshi_client(platform: str = "KALSHI") -> KalshiTradingClient:
    """Real client in live mode; paper-simulated wrapper otherwise (default).

    `platform` selects which *_PAPER_TRADING flag gates this instance. Crypto
    stayed in paper mode (real backtest showed it losing money on noise --
    2026-08-15) while macro/mlb_totals went live on their own separate flag,
    so they can no longer share one blanket "KALSHI" switch -- see each
    BOTS entry's "paper_platform" below."""
    real = KalshiTradingClient()
    if is_paper_mode(platform):
        from src.trading.kalshi.paper_client import PaperKalshiTradingClient
        return PaperKalshiTradingClient(real_client=real)
    return real


def make_polymarket_client() -> PolymarketTradingClient:
    """Real client in live mode; paper-simulated wrapper otherwise (default).

    Gated on its own POLYMARKET_PAPER_TRADING flag, independent of Kalshi's —
    there's no funded international Polymarket account, so this stays in
    paper mode regardless of whether Kalshi goes live.
    """
    real = PolymarketTradingClient()
    if is_paper_mode("POLYMARKET"):
        from src.trading.polymarket.paper_client import PaperPolymarketTradingClient
        return PaperPolymarketTradingClient(real_client=real)
    return real

# ── Sports series watched by the arb scanner (Kalshi ↔ Polymarket) ───────────
# The arb scanner owns these series exclusively. They are intentionally excluded
# from MACRO_SERIES below so the macro bot and arb scanner never compete for
# the same ticker.
ARB_SERIES = ["KXATPMATCH", "KXWTAMATCH", "KXMLBGAME", "KXUFCFIGHT"]

# ── Global same-market lock scan (yes_ask + no_ask < $1, guaranteed profit) ──
# find_locks() already runs every cycle inside each of crypto/macro/mlb_totals,
# but only against that bot's own narrow series_list -- roughly 24 series
# total. Checked live 2026-08-15: it has NEVER once found a real lock across
# this account's entire trading history, and a truly unbounded scan
# (list_liquid_open_markets's own docstring: "50k+ markets, most illiquid
# esports") would be far too expensive to poll regularly. This widens
# coverage to every series already scanned by SOMETHING in this process
# (crypto+macro+mlb_totals) plus the weather series nothing else watches yet
# -- meaningfully broader without the unbounded-scan cost. Deliberately
# excludes ARB_SERIES (the arb scanner's own exclusive territory, same
# reasoning as MACRO_SERIES above) to avoid two threads competing for the
# same ticker. Defined below, once CRYPTO_SERIES/MACRO_SERIES/MLB_TOTALS_SERIES exist.

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
]

# KXMLBTOTAL/KXMLBSPREAD used to live in MACRO_SERIES above -- added there
# only because they're "not in ARB_SERIES", never because anyone checked
# they'd calibrate against the same win-rate table as FOMC/CPI/elections.
# They don't: split out and checked separately (2026-08-15), the combined
# "macro" win-rate table showed 15-31pp gaps between quoted price and real
# outcome at several price points. Isolating each half showed why -- real
# macro events (FED, SENATE, HOUSE, KXINX, ...) barely clear the sample
# floor at all (1-3 price points with 30+ trades), while MLB totals/spreads
# have plenty of raw trade volume but are badly miscalibrated on their own
# (still 15-31pp off in isolation) -- likely correlated trades within the
# same game being counted as independent samples, not yet confirmed. Pooling
# them meant real macro's already-thin signal was further diluted by a
# differently-broken sports pool, and neither category's problem was visible
# until they were split apart.
MLB_TOTALS_SERIES = ["KXMLBTOTAL", "KXMLBSPREAD"]

GLOBAL_LOCK_SERIES = CRYPTO_SERIES + MACRO_SERIES + MLB_TOTALS_SERIES + list(WEATHER_SERIES)

# ── Bot configurations ────────────────────────────────────────────────────────

BOTS = [
    {
        "name": "crypto",
        # Real backtest showed crypto losing money on noise at current
        # settings (-$28.36, 35% win rate; break-even only appeared at a
        # much stricter edge threshold that found zero qualifying real
        # trades, ever). Stays on its own flag, paper by default, until it
        # actually shows a real edge rather than market efficiency noise.
        "paper_platform": "KALSHI",
        "series_list": CRYPTO_SERIES,
        "ticker_prefixes": CRYPTO_SERIES,
        "poll_interval": 15,                 # scan every 15s — catch fast-moving opportunities
        "max_open_positions": 40,            # allow deep position stack
        "balance_fraction": 0.55,
        "min_edge_pp": 0.5,
        "max_spend_cents": 100,               # up to $1 per trade
        "max_spread": 15,
        "min_historical_trades": 30,
        "min_entry_price": 18,
        "max_entry_price": 55,
        "min_close_seconds": 60,
        "max_close_seconds": 3600,           # only bet on markets closing within 1 hour
    },
    {
        "name": "macro",
        # Live 2026-08-15, on its own flag shared with mlb_totals (not
        # crypto's) -- this is the category that was actually diagnosed and
        # fixed this session (the 15-31pp macro miscalibration turned out to
        # be MLB totals/spreads pooled in here; see MLB_TOTALS_SERIES note).
        "paper_platform": "KALSHI_MACRO_MLB_TOTALS",
        "series_list": MACRO_SERIES,
        "ticker_prefixes": MACRO_SERIES,
        "poll_interval": 30,
        "max_open_positions": 40,
        "balance_fraction": 0.225,
        "min_edge_pp": 0.5,
        "max_spend_cents": 100,               # up to $1 per trade
        "max_spread": 15,
        "min_historical_trades": 30,
        "min_entry_price": 18,
        "max_entry_price": 55,
        "min_close_seconds": 60,
        "max_close_seconds": 3600,           # only bet on markets closing within 1 hour
    },
    {
        # Split out from "macro" 2026-08-15 -- see MLB_TOTALS_SERIES comment
        # above. Same baseline tuning as macro for now; not yet re-tuned
        # against its own (still-unexplained) 15-31pp miscalibration.
        "name": "mlb_totals",
        # Live 2026-08-15, same flag as macro -- see that entry's note.
        "paper_platform": "KALSHI_MACRO_MLB_TOTALS",
        "series_list": MLB_TOTALS_SERIES,
        "ticker_prefixes": MLB_TOTALS_SERIES,
        "poll_interval": 30,
        "max_open_positions": 40,
        "balance_fraction": 0.225,
        "min_edge_pp": 0.5,
        "max_spend_cents": 100,               # up to $1 per trade
        "max_spread": 15,
        "min_historical_trades": 30,
        "min_entry_price": 18,
        "max_entry_price": 55,
        "min_close_seconds": 60,
        "max_close_seconds": 3600,           # only bet on markets closing within 1 hour
    },
]

# ─────────────────────────────────────────────────────────────────────────────


def _run_bot(
    cfg: dict,
    stop_event: threading.Event,
    tracker: PerformanceTracker,
    client: KalshiTradingClient,
    reservation: TickerReservation,
    event_tracker: EventExposureTracker,
):
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
        reservation=reservation,
        event_tracker=event_tracker,
        pending_fills_path=Path(f"data/kalshi/pending_fills_{cfg['name']}.json"),
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
    # Held for the life of the process — a second copy of run_all.py exits
    # immediately instead of racing this one for the same files/rate limits.
    _instance_lock = acquire_single_instance_lock("run_all")  # noqa: F841

    stop_event = threading.Event()
    for platform in sorted({cfg["paper_platform"] for cfg in BOTS}):
        log_mode_banner(f"run_all-{platform.lower()}", platform)
    log_mode_banner("run_all-poly-leg", "POLYMARKET")

    def _shutdown(sig, frame):
        logging.getLogger("launcher").info("Shutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Shared learning tracker — persists across restarts, shared by all bots.
    tracker = PerformanceTracker()

    # One shared Kalshi client per paper_platform group (not one for the
    # whole process anymore -- crypto and macro/mlb_totals can now be in
    # different modes at once, see BOTS above). Within a group this still
    # matters a lot in paper mode: each PaperKalshiTradingClient keeps its
    # own in-memory ledger at the same default path, so two separate paper
    # instances in the same group would race and clobber each other's
    # writes. One shared instance per group (with its own internal lock) is
    # that group's single source of truth, and for a live group it also
    # centralizes rate-limiting across its bots instead of each one
    # throttling independently against the same account.
    kalshi_clients = {
        platform: make_kalshi_client(platform)
        for platform in {cfg["paper_platform"] for cfg in BOTS}
    }
    # The arb scanner watches a disjoint series list (sports, not
    # crypto/macro/mlb_totals) and has its own live/paper safety gate below
    # -- deliberately kept on the plain "KALSHI" flag/client (crypto's),
    # unchanged by the macro/mlb_totals split, since going live there is a
    # separate decision this doesn't make on its own.
    kalshi_client = kalshi_clients["KALSHI"]
    poly_client = make_polymarket_client()

    # Shared across crypto/macro/convergence: closes the race window between
    # two threads both seeing an unheld ticker (reservation), and caps total
    # daily spend on any one correlated real-world event (event_tracker).
    reservation = TickerReservation()
    event_tracker = EventExposureTracker(MAX_EVENT_EXPOSURE_CENTS)

    # One shared indexer — keeps local trade data fresh for all bots.
    # Only scan crypto+macro+mlb_totals series to avoid iterating over 9M+ esports markets.
    all_series = CRYPTO_SERIES + MACRO_SERIES + MLB_TOTALS_SERIES
    indexer = threading.Thread(
        target=run_indexer_loop,
        args=(stop_event, 1800, all_series),
        daemon=True,
        name="indexer",
    )
    indexer.start()

    # Polls OPEN markets every 5 min to build a point-in-time liquidity time
    # series (bid/ask/volume/open_interest) — data/kalshi/markets never
    # captured this (see src/indexers/kalshi/market_snapshots.py docstring),
    # which made any liquidity-based strategy feature untestable. Writes to
    # a separate directory; does not touch the existing win-rate calibration data.
    snapshot_indexer = threading.Thread(
        target=run_snapshot_loop,
        args=(stop_event, 300, all_series),
        daemon=True,
        name="market_snapshots",
    )
    snapshot_indexer.start()

    threads = []
    for cfg in BOTS:
        t = threading.Thread(
            target=_run_bot,
            args=(cfg, stop_event, tracker, kalshi_clients[cfg["paper_platform"]], reservation, event_tracker),
            name=cfg["name"],
            daemon=True,
        )
        t.start()
        threads.append(t)

    # Cross-platform arb scanner: Kalshi ↔ Polymarket sports markets.
    # Only safe to auto-execute if EITHER leg can't actually place a real
    # order (Kalshi still paper — nothing real happens either way) OR BOTH
    # legs are genuinely live (Polymarket also live means it's actually
    # funded and can really hedge). The dangerous combination is Kalshi live
    # + Polymarket paper: a real Kalshi order with no funded account on the
    # other side to hedge it — a naked directional bet, not the risk-free
    # arb this is meant to be. There's no funded international Polymarket
    # account (confirmed), so Polymarket stays in paper mode regardless of
    # Kalshi, and this correctly disables auto-execution once Kalshi goes live.
    def _run_arb(stop_event):
        log = logging.getLogger("arb")
        can_execute = is_paper_mode("KALSHI") or not is_paper_mode("POLYMARKET")
        log.info(
            "Arb scanner started — watching %s (auto-execute=%s)",
            ARB_SERIES, can_execute,
        )
        run_arb_loop(
            kalshi_client, stop_event, ARB_SERIES,
            execute_kalshi=can_execute, poly_trading_client=poly_client,
        )

    arb_thread = threading.Thread(
        target=_run_arb, args=(stop_event,), name="arb", daemon=True
    )
    arb_thread.start()
    threads.append(arb_thread)

    # Global same-market lock scanner: risk-free (yes_ask+no_ask<$1) checked
    # across every series any bot in this process already watches, plus
    # weather -- see GLOBAL_LOCK_SERIES above for why this exists and why
    # it's bounded rather than unbounded. On the crypto/paper client
    # deliberately, not the live macro/mlb_totals one -- this is new code
    # (list_liquid_open_markets has never been called with a list this wide
    # before), and a guaranteed-profit-by-construction check is still not a
    # reason to skip proving new code paper-side first. Point it at
    # kalshi_clients["KALSHI_MACRO_MLB_TOTALS"] once it's shown real, correct
    # behavior here.
    def _run_global_lock_scanner(stop_event):
        log = logging.getLogger("global_locks")
        strategy = MispricingStrategy()  # find_locks() needs no win-rate/prefix config at all
        placer = TradingBot(client=kalshi_client, strategy=strategy, reservation=reservation)
        while not stop_event.is_set():
            try:
                markets = kalshi_client.list_liquid_open_markets(series_list=GLOBAL_LOCK_SERIES)
                held = {p.ticker for p in kalshi_client.get_positions()} | {o.ticker for o in kalshi_client.get_open_orders()}
                locks = strategy.find_locks(markets, held)
                if locks:
                    log.info("Found %d GLOBAL lock opportunity(s) across %d open markets (%d series)",
                              len(locks), len(markets), len(GLOBAL_LOCK_SERIES))
                    placer._place_locks(locks, held, budget_cents=500)
                else:
                    log.debug("Scanned %d open markets across %d series, no locks", len(markets), len(GLOBAL_LOCK_SERIES))
            except Exception as exc:
                log.error("Global lock scan error: %s", exc, exc_info=True)
            stop_event.wait(timeout=45)

    lock_scanner_thread = threading.Thread(
        target=_run_global_lock_scanner, args=(stop_event,), name="global_locks", daemon=True
    )
    lock_scanner_thread.start()
    threads.append(lock_scanner_thread)

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
            max_spend_cents=100,
            poll_interval=20,
            reservation=reservation,
            event_tracker=event_tracker,
        )
        bot.run(stop_event)

    conv_thread = threading.Thread(
        target=_run_convergence, args=(stop_event,), name="convergence", daemon=True
    )
    conv_thread.start()
    threads.append(conv_thread)

    logging.getLogger("launcher").info(
        "%d bots + 1 convergence + 1 indexer + 1 arb scanner + 1 global lock scanner running. Ctrl+C to stop.",
        len(BOTS),
    )

    for t in threads:
        t.join()

    for client in kalshi_clients.values():
        client.close()
    poly_client.close()
    logging.getLogger("launcher").info("All stopped.")


if __name__ == "__main__":
    main()

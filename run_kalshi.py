"""Launch the Kalshi calibration-mispricing bot.

Read `docs/kalshi-bot-operating-manual.md` before running this, and §6 of
it before ever setting KALSHI_PAPER_TRADING=false. Short version:

  - The prior generation of these bots traded a funded account and was
    deliberately stopped and deleted (commit a55b95d). Rebuilding the code
    did not re-authorize live trading; §6 is what does.
  - The edge thesis is an ASSUMPTION (that a historical win-rate-by-price
    table predicts future outcomes), not a proven one. Every gate in
    src/trading/kalshi/risk.py is conservative for that reason.
  - With no data/kalshi/{trades,markets} present, the calibration table is
    empty and the bot finds zero opportunities. That is correct behaviour,
    not a bug to route around: no table means no thesis.

Paper mode by default, like every other bot launcher in this repo. The
KALSHI_PAPER_TRADING flag is deliberately separate from every other
platform's -- they share one operator but not one account, and arming one
must never silently arm another (see src/trading/mode.py for the incident
that rule comes from).

Usage
-----
    python run_kalshi.py
"""

from __future__ import annotations

import logging
import signal
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# The process owns its own log rather than relying on a shell redirect, so
# that stopping a scheduled task actually stops the bot instead of reaping
# only a cmd.exe wrapper and orphaning the python grandchild (see
# run_polyus.py for the incident this pattern comes from).
_LOG_DIR = Path(__file__).resolve().parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_file_handler = RotatingFileHandler(
    _LOG_DIR / "run_kalshi.log",
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

from src.common.event_exposure import EventExposureTracker  # noqa: E402
from src.common.reservation import TickerReservation  # noqa: E402
from src.common.single_instance import acquire_single_instance_lock  # noqa: E402
from src.trading.kalshi.bot import TradingBot, run_indexer_loop  # noqa: E402
from src.trading.kalshi.learning import PerformanceTracker  # noqa: E402
from src.trading.kalshi.paper_client import PaperKalshiTradingClient  # noqa: E402
from src.trading.kalshi.risk import RiskLedger  # noqa: E402
from src.trading.kalshi.strategy import MispricingStrategy  # noqa: E402
from src.trading.kalshi.trading_client import KalshiTradingClient  # noqa: E402
from src.trading.mode import is_paper_mode, log_mode_banner  # noqa: E402

logger = logging.getLogger("run_kalshi")

PLATFORM = "KALSHI"

# Scoped to liquid, short-duration series. Deliberately NOT the full market
# list: most of Kalshi's ~50k markets are illiquid, and an illiquid market's
# "edge" is mostly a stale quote.
SERIES = ["KXBTC", "KXBTCD", "KXBTC15M", "KXETH", "KXETHD", "KXETH15M"]

# Per-event daily exposure cap. Incident #3 in the manual: adjacent,
# mutually exclusive brackets of one BTC close hour accumulated across
# scans until several guaranteed-losing legs were held at once.
MAX_EVENT_CENTS = 500


def main() -> int:
    # Held for the life of the process — releasing this handle releases the
    # lock, so a second copy would start racing this one for the same
    # ledger/learning files and rate-limit budget.
    _instance_lock = acquire_single_instance_lock("run_kalshi")  # noqa: F841

    paper = is_paper_mode(PLATFORM)
    log_mode_banner("kalshi-mispricing", PLATFORM)

    try:
        real_client = KalshiTradingClient()
    except RuntimeError as exc:
        # A missing local credential is a setup problem, not a crash — say so
        # plainly and exit non-zero instead of dumping a traceback.
        logger.error("%s", exc)
        return 1
    client = PaperKalshiTradingClient(real_client=real_client) if paper else real_client

    tracker = PerformanceTracker()
    strategy = MispricingStrategy(ticker_prefixes=SERIES, tracker=tracker)

    starting_bankroll = client.get_balance()
    risk_ledger = RiskLedger(starting_bankroll_cents=starting_bankroll)
    logger.info(
        "Starting bankroll $%.2f — daily loss limit %.0f%%, drawdown kill switch %.0f%%",
        starting_bankroll / 100,
        risk_ledger.daily_loss_limit_fraction * 100,
        risk_ledger.max_drawdown_fraction * 100,
    )

    bot = TradingBot(
        client=client,
        strategy=strategy,
        series_list=SERIES,
        tracker=tracker,
        reservation=TickerReservation(),
        event_tracker=EventExposureTracker(max_event_cents=MAX_EVENT_CENTS),
        pending_fills_path=Path("data/kalshi/pending_fills.json"),
        risk_ledger=risk_ledger,
    )

    stop_event = threading.Event()

    def _handle_signal(signum, _frame):
        logger.info("Received signal %s — shutting down after this scan.", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # The calibration table is only as fresh as the data behind it. A live
    # incident (documented in bot.run_indexer_loop) ran six weeks of trades
    # off a snapshot that was never refreshed after its initial backfill.
    indexer = threading.Thread(
        target=run_indexer_loop,
        args=(stop_event,),
        kwargs={"ticker_prefixes": SERIES},
        daemon=True,
    )
    indexer.start()

    try:
        bot.run_forever(stop_event=stop_event)
    finally:
        stop_event.set()
        client.close()
        if paper and real_client is not client:
            real_client.close()
        logger.info("Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

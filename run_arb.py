"""Run only the cross-platform arb scanner (Kalshi <-> Polymarket).

No betting bots. Places both legs directly when a guaranteed arbitrage is
found (combined cost < 95c on the same event same day).
"""
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
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from src.trading.kalshi.trading_client import KalshiTradingClient
from src.trading.mode import is_paper_mode, log_mode_banner
from src.trading.polymarket.arb_scanner import run_arb_loop
from src.trading.polymarket.trading_client import PolymarketTradingClient

ARB_SERIES = ["KXATPMATCH", "KXWTAMATCH", "KXMLBGAME", "KXUFCFIGHT"]


def make_kalshi_client() -> KalshiTradingClient:
    real = KalshiTradingClient()
    if is_paper_mode():
        from src.trading.kalshi.paper_client import PaperKalshiTradingClient
        return PaperKalshiTradingClient(real_client=real)
    return real


def make_polymarket_client() -> PolymarketTradingClient:
    real = PolymarketTradingClient()
    if is_paper_mode():
        from src.trading.polymarket.paper_client import PaperPolymarketTradingClient
        return PaperPolymarketTradingClient(real_client=real)
    return real


def main():
    stop_event = threading.Event()
    log_mode_banner("run_arb")

    def _shutdown(sig, frame):
        logging.getLogger("launcher").info("Shutting down arb scanner...")
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log = logging.getLogger("launcher")
    log.info("Arb scanner started — watching %s", ARB_SERIES)
    log.info("Scans every 5 min. Both legs auto-executed when a guaranteed pair is found.")

    with make_kalshi_client() as client, make_polymarket_client() as poly_client:
        run_arb_loop(
            client, stop_event, ARB_SERIES,
            execute_kalshi=True, poly_trading_client=poly_client,
        )

    log.info("Arb scanner stopped.")


if __name__ == "__main__":
    main()

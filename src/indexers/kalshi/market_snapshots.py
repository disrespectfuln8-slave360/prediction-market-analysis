"""Live time-series indexer for OPEN Kalshi markets.

KalshiMarketsIndexer (markets.py) and backfill_history.py both only ever
fetch already-FINALIZED markets, and ParquetStorage.append_markets dedupes
permanently per ticker — correct for their purpose (one-shot win-rate
calibration data), but it means data/kalshi/markets/*.parquet has never
captured a market's bid/ask/volume/open_interest while it was still open
and trading. Every historical snapshot was written weeks after the fact,
during a backfill, making point-in-time liquidity features impossible to
test (confirmed by direct query: every ticker has exactly one row, whose
_fetched_at postdates 100% of that ticker's own trades).

This indexer polls currently-open markets on a short interval and appends
every poll with dedupe=False, to a SEPARATE directory
(data/kalshi/market_snapshots/) so the existing calibration table and
win-rate query are completely unaffected. This can only start collecting
the right data going forward — it cannot retroactively fix historical gaps.
"""

from __future__ import annotations

from pathlib import Path

from src.common.indexer import Indexer
from src.common.storage import ParquetStorage
from src.indexers.kalshi.client import KalshiClient
from src.indexers.kalshi.models import Market

DATA_DIR = Path("data/kalshi/market_snapshots")


class KalshiMarketSnapshotIndexer(Indexer):
    """Fetches currently-open markets for the given series and appends a
    fresh snapshot row per ticker per run (no cross-run dedup)."""

    def __init__(self, ticker_prefixes: list[str] | None = None):
        super().__init__(
            name="kalshi_market_snapshots",
            description="Polls open Kalshi markets to build a point-in-time liquidity time series",
        )
        self.ticker_prefixes = ticker_prefixes or []

    def _fetch_open_markets(self, client: KalshiClient) -> list[Market]:
        all_markets: list[Market] = []
        seen: set[str] = set()
        for prefix in self.ticker_prefixes:
            cursor = None
            while True:
                params: dict = {"limit": 200, "status": "open", "series_ticker": prefix}
                if cursor:
                    params["cursor"] = cursor
                try:
                    data = client.http.get("/markets", params=params)
                except Exception:
                    break
                for m in data.get("markets", []):
                    market = Market.from_dict(m)
                    if market.ticker not in seen:
                        all_markets.append(market)
                        seen.add(market.ticker)
                cursor = data.get("cursor")
                if not cursor:
                    break
        return all_markets

    def run(self) -> int:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        client = KalshiClient()
        storage = ParquetStorage(data_dir=DATA_DIR)
        try:
            markets = self._fetch_open_markets(client)
            if not markets:
                return 0
            storage.append_markets(markets, dedupe=False)
            return len(markets)
        finally:
            client.close()


def run_snapshot_loop(stop_event, interval: int, ticker_prefixes: list[str]):
    """Background loop — poll open markets every `interval` seconds.

    Intended to run in its own daemon thread, same pattern as
    src.trading.kalshi.bot.run_indexer_loop.
    """
    import logging

    logger = logging.getLogger(__name__)
    while True:
        try:
            n = KalshiMarketSnapshotIndexer(ticker_prefixes=ticker_prefixes).run()
            logger.info("Market snapshot: captured %d open markets", n)
        except Exception as exc:
            logger.warning("Market snapshot error: %s", exc)
        if stop_event.wait(timeout=interval):
            break

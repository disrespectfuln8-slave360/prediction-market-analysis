"""Indexer for Kalshi markets data."""

from pathlib import Path
from typing import Optional

from src.common.indexer import Indexer
from src.common.storage import ParquetStorage
from src.indexers.kalshi.client import KalshiClient

DATA_DIR = Path("data/kalshi/markets")
CURSOR_FILE = Path("data/kalshi/.backfill_cursor")


class KalshiMarketsIndexer(Indexer):
    """Fetches and stores Kalshi markets data."""

    def __init__(
        self,
        min_close_ts: Optional[int] = None,
        max_close_ts: Optional[int] = None,
    ):
        super().__init__(
            name="kalshi_markets",
            description="Backfills Kalshi markets data to parquet files",
        )
        self._min_close_ts = min_close_ts
        self._max_close_ts = max_close_ts

    def run(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)

        client = KalshiClient()
        storage = ParquetStorage(data_dir=DATA_DIR)

        cursor = None
        if CURSOR_FILE.exists():
            cursor = CURSOR_FILE.read_text().strip() or None
            if cursor:
                print(f"Resuming from cursor: {cursor[:20]}...")

        total_fetched = 0
        total_stored = 0
        for markets, next_cursor in client.iter_markets(
            limit=1000,
            cursor=cursor,
            min_close_ts=self._min_close_ts,
            max_close_ts=self._max_close_ts,
        ):
            if markets:
                stored_this_batch = storage.append_markets(markets)
                total_fetched += len(markets)
                total_stored += stored_this_batch
                print(f"Fetched {len(markets)} markets (total fetched: {total_fetched}, "
                      f"total newly stored: {total_stored})")

            if next_cursor:
                CURSOR_FILE.write_text(next_cursor)
            else:
                if CURSOR_FILE.exists():
                    CURSOR_FILE.unlink()
                break

        print(f"\nBackfill complete: {total_fetched} markets fetched, {total_stored} newly stored")

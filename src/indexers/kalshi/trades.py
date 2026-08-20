"""Indexer for Kalshi trades data."""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from tqdm import tqdm

from src.common.indexer import Indexer
from src.indexers.kalshi.client import KalshiClient

DATA_DIR = Path("data/kalshi/trades")
MARKETS_DIR = Path("data/kalshi/markets")
CURSOR_FILE = Path("data/kalshi/.backfill_trades_cursor")


class KalshiTradesIndexer(Indexer):
    """Fetches and stores Kalshi trades data."""

    def __init__(
        self,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
        max_workers: int = 10,
        ticker_prefixes: Optional[list] = None,
    ):
        super().__init__(
            name="kalshi_trades",
            description="Backfills Kalshi trades data to parquet files",
        )
        self._min_ts = min_ts
        self._max_ts = max_ts
        self._max_workers = max_workers
        self._ticker_prefixes = ticker_prefixes

    def run(self) -> None:
        BATCH_SIZE = 10000
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Load existing trade IDs for deduplication
        existing_trade_ids: set[str] = set()
        existing_tickers: set[str] = set()
        parquet_files = list(DATA_DIR.glob("trades_*.parquet"))
        if parquet_files:
            print("Loading existing trades for deduplication...")
            try:
                result = duckdb.sql(f"SELECT DISTINCT trade_id, ticker FROM '{DATA_DIR}/trades_*.parquet'").fetchall()
                for trade_id, ticker in result:
                    existing_trade_ids.add(trade_id)
                    existing_tickers.add(ticker)
                print(f"Found {len(existing_trade_ids)} existing trades")
            except Exception:
                pass

        prefix_clause = ""
        if self._ticker_prefixes:
            conditions = " OR ".join(f"ticker LIKE '{p}%'" for p in self._ticker_prefixes)
            prefix_clause = f"AND ({conditions})"

        try:
            all_tickers = duckdb.sql(f"""
                SELECT DISTINCT ticker FROM '{MARKETS_DIR}/markets_*_*.parquet'
                WHERE volume >= 100
                {prefix_clause}
                ORDER BY ticker
            """).fetchall()
            all_tickers = [row[0] for row in all_tickers]
        except Exception as e:
            print(f"No markets data found ({e}); nothing to process")
            return
        print(f"Found {len(all_tickers)} unique markets")

        # When a time window is set (incremental update), re-scan all tickers for
        # new trades — existing_trade_ids handles per-record dedup.
        # When doing a full backfill (no time window), skip already-processed tickers.
        if self._min_ts is not None:
            tickers_to_process = all_tickers
            print(f"Incremental update: scanning all {len(tickers_to_process)} tickers for new trades")
        else:
            tickers_to_process = [t for t in all_tickers if t not in existing_tickers]
            print(
                f"Skipped {len(all_tickers) - len(tickers_to_process)} already processed, "
                f"{len(tickers_to_process)} to fetch"
            )

        if not tickers_to_process:
            print("Nothing to process")
            return

        all_trades: list[dict] = []
        total_trades_saved = 0
        next_chunk_idx = 0

        # Calculate next chunk index
        if parquet_files:
            indices = []
            for f in parquet_files:
                parts = f.stem.split("_")
                if len(parts) >= 2:
                    try:
                        indices.append(int(parts[1]))
                    except ValueError:
                        pass
            if indices:
                next_chunk_idx = max(indices) + BATCH_SIZE

        def save_batch(trades_batch: list[dict]) -> int:
            nonlocal next_chunk_idx
            if not trades_batch:
                return 0
            chunk_path = DATA_DIR / f"trades_{next_chunk_idx}_{next_chunk_idx + BATCH_SIZE}.parquet"
            df = pd.DataFrame(trades_batch)
            df.to_parquet(chunk_path)
            next_chunk_idx += BATCH_SIZE
            return len(trades_batch)

        def fetch_ticker_trades(client: KalshiClient, ticker: str) -> list[dict]:
            """Fetch trades for a single ticker."""
            trades = client.get_market_trades(
                ticker,
                verbose=False,
                min_ts=self._min_ts,
                max_ts=self._max_ts,
            )
            if not trades:
                return []
            fetched_at = datetime.now(timezone.utc)
            return [{**asdict(t), "_fetched_at": fetched_at} for t in trades if t.trade_id not in existing_trade_ids]

        # Concurrent fetching. Share one client so its rate limiter is enforced
        # across all worker threads instead of each thread getting its own budget.
        # Disable the progress bar when not attached to a terminal (e.g. a
        # bot's background indexer thread with output redirected to a file)
        # — each tick would otherwise be written as its own line, producing
        # unbounded log growth over a long-running process.
        pbar = tqdm(total=len(tickers_to_process), desc="Fetching trades", disable=not sys.stdout.isatty())
        shared_client = KalshiClient()
        try:
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = {
                    executor.submit(fetch_ticker_trades, shared_client, ticker): ticker for ticker in tickers_to_process
                }

                for future in as_completed(futures):
                    ticker = futures[future]
                    try:
                        trades_data = future.result()
                        if trades_data:
                            all_trades.extend(trades_data)

                        pbar.update(1)
                        pbar.set_postfix(buffer=len(all_trades), saved=total_trades_saved, last=ticker[-20:])

                        # Save in batches
                        while len(all_trades) >= BATCH_SIZE:
                            saved = save_batch(all_trades[:BATCH_SIZE])
                            total_trades_saved += saved
                            all_trades = all_trades[BATCH_SIZE:]

                    except Exception as e:
                        pbar.update(1)
                        tqdm.write(f"Error fetching {ticker}: {e}")
        finally:
            shared_client.close()
            pbar.close()

        # Save remaining
        if all_trades:
            total_trades_saved += save_batch(all_trades)

        print(
            f"\nBackfill trades complete: {len(tickers_to_process)} markets processed, "
            f"{total_trades_saved} trades saved"
        )

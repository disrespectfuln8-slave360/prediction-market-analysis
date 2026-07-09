"""
Fetch months of historical Kalshi markets + trades for win-rate model training.

Uses the /events API with series_ticker + with_nested_markets=true — fetches
only the series the bots trade, gets 36k markets per API call instead of 1k,
and completes Phase 1 in minutes instead of hours.

Run once: python backfill_history.py
"""
from __future__ import annotations
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import duckdb
import pandas as pd
from tqdm import tqdm

from src.common.client import HttpClient
from src.common.storage import ParquetStorage
from src.indexers.kalshi.client import KalshiClient, KALSHI_API_HOST
from src.indexers.kalshi.models import Market

MARKETS_DIR = Path("data/kalshi/markets")
TRADES_DIR  = Path("data/kalshi/trades")

# Only fetch the series the bots actually trade
BOT_SERIES = [
    "KXBTC", "KXETH", "KXSOL", "KXDOGE",
    "KXINX", "KXNDX", "KXSPY", "KXDJI",
    "INX", "NDX", "NASDAQ", "FED", "FOMC", "CPI", "JOBS", "GDP", "UNRATE",
]

MAX_TRADE_WORKERS = 5
TRADE_BATCH_SIZE  = 10_000


def _client(rate: float = 4, retries: int = 15) -> KalshiClient:
    c = KalshiClient()
    c.http = HttpClient(base_url=KALSHI_API_HOST, rate_limit=rate, max_retries=retries)
    return c


# ── PHASE 1: market metadata via events API ───────────────────────────────────

def fetch_series_markets(series: str, client: KalshiClient, storage: ParquetStorage) -> int:
    """Fetch all finalized markets for one series via the events endpoint."""
    total = 0
    cursor = None
    while True:
        params = {
            "limit": 200,
            "series_ticker": series,
            "with_nested_markets": "true",
        }
        if cursor:
            params["cursor"] = cursor

        data = client.http.get("/events", params=params)
        events = data.get("events", [])
        cursor = data.get("cursor")

        # Parse and filter to finalized markets only
        markets: list[Market] = []
        for evt in events:
            for m in evt.get("markets", []):
                if m.get("result") in ("yes", "no") and m.get("status") == "finalized":
                    try:
                        markets.append(Market.from_dict(m))
                    except Exception:
                        pass

        if markets:
            storage.append_markets(markets)
            total += len(markets)

        if not cursor:
            break

    return total


def fetch_historical_markets() -> int:
    print(f"Phase 1: fetching finalized markets for {len(BOT_SERIES)} series via events API")
    MARKETS_DIR.mkdir(parents=True, exist_ok=True)
    client  = _client(rate=4)
    storage = ParquetStorage(data_dir=MARKETS_DIR)
    grand_total = 0
    try:
        for series in BOT_SERIES:
            n = fetch_series_markets(series, client, storage)
            print(f"  {series}: {n:,} finalized markets", flush=True)
            grand_total += n
    finally:
        client.close()
    print(f"\nPhase 1 done: {grand_total:,} finalized markets fetched\n")
    return grand_total


# ── PHASE 2: trades for newly discovered markets ──────────────────────────────

def fetch_missing_trades() -> int:
    print("Phase 2: fetching trades for finalized bot-series markets")

    # Load existing trade tickers
    existing_ids:     set[str] = set()
    existing_tickers: set[str] = set()
    if list(TRADES_DIR.glob("*.parquet")):
        rows = duckdb.sql(
            f"SELECT DISTINCT trade_id, ticker FROM '{TRADES_DIR}/trades_*.parquet'"
        ).fetchall()
        for tid, tk in rows:
            existing_ids.add(tid)
            existing_tickers.add(tk)
    print(f"  Already have trades for {len(existing_tickers):,} tickers")

    # Find finalized bot-series markets missing trades
    if not list(MARKETS_DIR.glob("*.parquet")):
        print("  No finalized markets on disk yet — run Phase 1 first. Nothing to fetch.")
        return 0

    series_filter = " OR ".join(f"ticker LIKE '{s}%'" for s in BOT_SERIES)
    rows = duckdb.sql(f"""
        SELECT DISTINCT ticker FROM '{MARKETS_DIR}/markets_*.parquet'
        WHERE status='finalized' AND result IN ('yes','no')
          AND volume >= 100
          AND ({series_filter})
    """).fetchall()
    all_tickers = [r[0] for r in rows]
    to_fetch    = [t for t in all_tickers if t not in existing_tickers]
    print(f"  {len(all_tickers):,} finalized liquid markets, {len(to_fetch):,} need trades")

    if not to_fetch:
        print("  Nothing to fetch.")
        return 0

    # Determine next chunk index
    existing_files = list(TRADES_DIR.glob("*.parquet"))
    next_idx = 0
    if existing_files:
        idxs = []
        for f in existing_files:
            parts = f.stem.split("_")
            if len(parts) >= 2:
                try: idxs.append(int(parts[1]))
                except ValueError: pass
        if idxs:
            next_idx = max(idxs) + TRADE_BATCH_SIZE

    buffer: list[dict] = []
    total_saved = 0
    chunk_idx   = next_idx

    def save_batch(batch: list[dict]):
        nonlocal chunk_idx, total_saved
        if not batch:
            return
        path = TRADES_DIR / f"trades_{chunk_idx}_{chunk_idx+TRADE_BATCH_SIZE}.parquet"
        pd.DataFrame(batch).to_parquet(path)
        chunk_idx   += TRADE_BATCH_SIZE
        total_saved += len(batch)

    def fetch_ticker(ticker: str) -> list[dict]:
        c = _client(rate=2)
        try:
            trades = c.get_market_trades(ticker, verbose=False)
            if not trades:
                return []
            ts = datetime.utcnow()
            return [
                {**asdict(t), "_fetched_at": ts}
                for t in trades
                if t.trade_id not in existing_ids
            ]
        finally:
            c.close()

    TRADES_DIR.mkdir(parents=True, exist_ok=True)
    pbar = tqdm(total=len(to_fetch), desc="Fetching trades", unit="ticker", disable=not sys.stdout.isatty())
    with ThreadPoolExecutor(max_workers=MAX_TRADE_WORKERS) as ex:
        futures = {ex.submit(fetch_ticker, t): t for t in to_fetch}
        try:
            for fut in as_completed(futures):
                ticker = futures[fut]
                try:
                    data = fut.result()
                    if data:
                        buffer.extend(data)
                        for d in data:
                            existing_ids.add(d["trade_id"])
                    while len(buffer) >= TRADE_BATCH_SIZE:
                        save_batch(buffer[:TRADE_BATCH_SIZE])
                        buffer = buffer[TRADE_BATCH_SIZE:]
                except Exception as e:
                    tqdm.write(f"Error {ticker}: {e}")
                pbar.update(1)
                pbar.set_postfix(buf=len(buffer), saved=total_saved)
        except KeyboardInterrupt:
            # Cancel not-yet-started fetches so Ctrl+C actually stops promptly
            # instead of waiting for every queued ticker to finish (default
            # ThreadPoolExecutor.shutdown behavior), and flush what we have.
            ex.shutdown(wait=False, cancel_futures=True)
            pbar.close()
            if buffer:
                save_batch(buffer)
            print(f"\nInterrupted — {total_saved:,} new trades saved before stopping")
            raise
    pbar.close()

    if buffer:
        save_batch(buffer)

    print(f"\nPhase 2 done: {total_saved:,} new trades saved")
    return total_saved


# ── SUMMARY ───────────────────────────────────────────────────────────────────

def print_model_coverage():
    try:
        df = duckdb.sql(f"""
            WITH resolved AS (
                SELECT ticker, result FROM '{MARKETS_DIR}/markets_*.parquet'
                WHERE status='finalized' AND result IN ('yes','no')
            ),
            tp AS (
                SELECT CASE WHEN t.taker_side='yes' THEN t.yes_price ELSE t.no_price END AS price,
                       CASE WHEN t.taker_side=m.result THEN 1 ELSE 0 END AS won
                FROM '{TRADES_DIR}/trades_*.parquet' t
                INNER JOIN resolved m ON t.ticker=m.ticker
            )
            SELECT price, COUNT(*) as n, ROUND(100.0*SUM(won)/COUNT(*),1) as wr
            FROM tp WHERE price BETWEEN 1 AND 99
            GROUP BY price HAVING COUNT(*)>=50
            ORDER BY price
        """).df()
        print(f"\nWin-rate model: {len(df)} price points (>=50 trades each)")
        if len(df):
            print(f"  Price range covered: {int(df.price.min())}¢ – {int(df.price.max())}¢")
            print(df.to_string(index=False))
    except Exception as e:
        print(f"Model coverage check failed: {e}")


if __name__ == "__main__":
    t0 = time.time()
    fetch_historical_markets()
    fetch_missing_trades()
    print_model_coverage()
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")

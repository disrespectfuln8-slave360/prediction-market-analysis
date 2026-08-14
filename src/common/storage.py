from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import duckdb
import pandas as pd


class ParquetStorage:
    CHUNK_SIZE = 10000

    def __init__(self, data_dir: Union[Path, str] = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._existing_tickers: set[str] | None = None

    def _get_market_chunks(self) -> list[Path]:
        """Get all market chunk files sorted by start index."""
        chunks = list(self.data_dir.glob("markets_*_*.parquet"))
        chunks.sort(key=lambda p: int(p.stem.split("_")[1]))
        return chunks

    def _chunk_path(self, start: int, end: int) -> Path:
        return self.data_dir / f"markets_{start}_{end}.parquet"

    def _load_existing_tickers(self) -> set[str]:
        """Load all existing tickers for deduplication."""
        if self._existing_tickers is not None:
            return self._existing_tickers
        self._existing_tickers = set()
        chunks = self._get_market_chunks()
        if chunks:
            result = duckdb.sql(f"SELECT DISTINCT ticker FROM '{self.data_dir}/markets_*.parquet'").fetchall()
            self._existing_tickers = {row[0] for row in result}
        return self._existing_tickers

    def append_markets(self, markets: list, dedupe: bool = True) -> int:
        """Append market records. Returns the number of records actually
        written by THIS call (post-dedup) — not a running total. A caller
        printing this as per-batch progress alongside a cumulative counter
        (see KalshiMarketsIndexer.run()) needs the two to mean different
        things, or the log becomes unreadable the moment the two diverge
        (a multi-million-ticker dataset makes that immediate, not eventual).

        dedupe=True (default): permanent per-ticker dedup — once a ticker has
        been written, it's skipped forever. Correct for a one-shot historical
        backfill (don't re-fetch/re-store an already-captured finalized
        market), which is the only thing that has ever called this method.

        dedupe=False: always append every record, with its own fresh
        _fetched_at. Needed for building a genuine time series of an OPEN
        market's state (bid/ask/volume/open_interest) across repeated polls
        — with dedupe=True, the second poll of the same still-open ticker
        would be silently dropped forever, which is exactly why no
        point-in-time liquidity data exists yet (see KalshiMarketSnapshotIndexer).
        """
        fetched_at = datetime.now(timezone.utc)
        existing = self._load_existing_tickers() if dedupe else None

        records = []
        for market in markets:
            if dedupe and market.ticker in existing:
                continue
            record = asdict(market)
            record["_fetched_at"] = fetched_at
            records.append(record)
            if dedupe:
                existing.add(market.ticker)

        if not records:
            return 0

        new_df = pd.DataFrame(records)
        chunks = self._get_market_chunks()

        if not chunks:
            chunk_path = self._chunk_path(0, self.CHUNK_SIZE)
            new_df.to_parquet(chunk_path)
            return len(records)

        last_chunk = chunks[-1]
        last_df = pd.read_parquet(last_chunk)
        combined = pd.concat([last_df, new_df], ignore_index=True)

        start = int(last_chunk.stem.split("_")[1])
        if len(combined) <= self.CHUNK_SIZE:
            combined.to_parquet(last_chunk)
        else:
            first_part = combined.iloc[: self.CHUNK_SIZE]
            first_part.to_parquet(last_chunk)
            remaining = combined.iloc[self.CHUNK_SIZE :]
            new_start = start + self.CHUNK_SIZE
            new_chunk_path = self._chunk_path(new_start, new_start + self.CHUNK_SIZE)
            remaining.to_parquet(new_chunk_path)

        return len(records)

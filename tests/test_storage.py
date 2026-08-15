"""Tests for ParquetStorage's dedup and return-value contract.

Regression coverage for a bug found live on 2026-08-14: append_markets()
returned len(existing) -- the running total of every ticker ever seen --
while its caller logged that value labeled "stored" (meant to be this
batch's newly-added count). Once the dataset reached multi-million rows
mid-backfill, the two numbers diverged so far the progress log became
actively misleading.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import duckdb
import pytest

from src.common.storage import ParquetStorage


@dataclass
class FakeMarket:
    ticker: str
    status: str = "finalized"
    result: Optional[str] = None


def test_append_markets_returns_the_count_added_this_call(tmp_path):
    storage = ParquetStorage(data_dir=tmp_path)

    first = storage.append_markets([FakeMarket("A"), FakeMarket("B"), FakeMarket("C")])
    assert first == 3

    # A fourth, genuinely new ticker alongside two already-seen ones -- the
    # return value must reflect only what THIS call added, not the
    # dataset's running total (which would be 4 here, not 1).
    second = storage.append_markets([FakeMarket("A"), FakeMarket("B"), FakeMarket("D")])
    assert second == 1


def test_append_markets_dedupes_permanently_across_calls(tmp_path):
    storage = ParquetStorage(data_dir=tmp_path)
    storage.append_markets([FakeMarket("A")])
    stored_again = storage.append_markets([FakeMarket("A")])

    assert stored_again == 0
    con = duckdb.connect()
    rows = con.execute(f"SELECT COUNT(*) FROM '{tmp_path}/markets_*.parquet' WHERE ticker = 'A'").fetchone()
    assert rows[0] == 1


def test_append_markets_dedupe_false_always_appends(tmp_path):
    """The point-in-time liquidity use case (see the method's own
    docstring) depends on repeated polls of the same still-open ticker
    NOT being silently dropped."""
    storage = ParquetStorage(data_dir=tmp_path)
    storage.append_markets([FakeMarket("A")], dedupe=False)
    storage.append_markets([FakeMarket("A")], dedupe=False)

    con = duckdb.connect()
    rows = con.execute(f"SELECT COUNT(*) FROM '{tmp_path}/markets_*.parquet' WHERE ticker = 'A'").fetchone()
    assert rows[0] == 2


def test_append_markets_returns_zero_when_nothing_new(tmp_path):
    storage = ParquetStorage(data_dir=tmp_path)
    storage.append_markets([FakeMarket("A")])
    assert storage.append_markets([FakeMarket("A")]) == 0

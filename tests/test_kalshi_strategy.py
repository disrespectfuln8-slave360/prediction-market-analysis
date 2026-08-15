"""Tests for MispricingStrategy's win-rate loading, focused on the shared
global-rates cache.

Regression coverage for a live incident (2026-08-15): run_all.py runs
several MispricingStrategy instances (crypto/macro/mlb_totals) as threads
in one process, all reading the same data directory. A category with too
little of its own data falls back to an unfiltered, full-table query on
every reload -- and every instance recomputing that same expensive query
independently was enough to OOM-crash the process once the underlying
dataset grew large. The fix shares one cached result across every
instance pointed at the same directory.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.trading.kalshi import strategy as strategy_module
from src.trading.kalshi.strategy import MispricingStrategy


@pytest.fixture(autouse=True)
def _clear_global_cache():
    """The cache is process-wide by design (that's the whole point) --
    tests must not leak state into each other."""
    strategy_module._global_rates_cache.clear()
    yield
    strategy_module._global_rates_cache.clear()


def _seed_data(tmp_path, ticker="OTHER-1", n=40):
    """30+ resolved trades on a ticker NOT in either strategy's
    ticker_prefixes below, so each instance's own category-scoped query
    finds nothing and both fall back to the global one -- which this data
    populates."""
    markets_dir = tmp_path / "markets"
    trades_dir = tmp_path / "trades"
    markets_dir.mkdir()
    trades_dir.mkdir()
    pd.DataFrame([{"ticker": ticker, "status": "finalized", "result": "yes"}]).to_parquet(
        markets_dir / "markets_0_1.parquet"
    )
    pd.DataFrame([
        {"ticker": ticker, "taker_side": "yes", "yes_price": 50, "no_price": 50}
        for _ in range(n)
    ]).to_parquet(trades_dir / "trades_0_1.parquet")
    return trades_dir, markets_dir


def test_two_instances_on_the_same_directory_share_one_global_query(tmp_path, monkeypatch):
    trades_dir, markets_dir = _seed_data(tmp_path)

    call_count = 0
    real_query = MispricingStrategy._query_win_rates

    def counting_query(self, prefixes):
        nonlocal call_count
        if prefixes is None:
            call_count += 1
        return real_query(self, prefixes)

    monkeypatch.setattr(MispricingStrategy, "_query_win_rates", counting_query)

    strat_a = MispricingStrategy(
        trades_dir=trades_dir, markets_dir=markets_dir,
        ticker_prefixes=["CRYPTO-ONLY"], min_historical_trades=30,
    )
    strat_b = MispricingStrategy(
        trades_dir=trades_dir, markets_dir=markets_dir,
        ticker_prefixes=["MACRO-ONLY"], min_historical_trades=30,
    )

    rates_a = strat_a._load_win_rates()
    rates_b = strat_b._load_win_rates()

    assert call_count == 1, "second instance should reuse the cached global query, not recompute it"
    assert rates_a["yes"] == rates_b["yes"]
    assert 50 in rates_a["yes"]  # confirms the fallback actually found the seeded data


def test_different_directories_are_not_cross_contaminated(tmp_path, monkeypatch):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    trades_a, markets_a = _seed_data(dir_a, ticker="OTHER-1")
    trades_b, markets_b = _seed_data(dir_b, ticker="OTHER-2")

    call_count = 0
    real_query = MispricingStrategy._query_win_rates

    def counting_query(self, prefixes):
        nonlocal call_count
        if prefixes is None:
            call_count += 1
        return real_query(self, prefixes)

    monkeypatch.setattr(MispricingStrategy, "_query_win_rates", counting_query)

    MispricingStrategy(
        trades_dir=trades_a, markets_dir=markets_a,
        ticker_prefixes=["X"], min_historical_trades=30,
    )._load_win_rates()
    MispricingStrategy(
        trades_dir=trades_b, markets_dir=markets_b,
        ticker_prefixes=["Y"], min_historical_trades=30,
    )._load_win_rates()

    assert call_count == 2, "distinct data directories must each get their own global query"

"""Tests for the widened same-market lock scan (run_all.py's
GLOBAL_LOCK_SERIES / _run_global_lock_scanner).

find_locks()/_place_locks() have been part of MispricingStrategy/TradingBot
since before this session, wired into every bot's per-cycle scan, but had
no test coverage at all -- confirmed live 2026-08-15 it has never once
fired in this account's history. Widening its market coverage (this
change) is the first real behavior change to this code path, so this adds
the coverage that was missing before touching it.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.indexers.kalshi.models import Market
from src.trading.kalshi.bot import TradingBot
from src.trading.kalshi.strategy import MispricingStrategy
from src.trading.kalshi.trading_client import Order, Position


def make_market(ticker, yes_ask, no_ask, status="open", close_in_secs=1800):
    return Market(
        ticker=ticker, event_ticker=ticker, market_type="binary", title=ticker,
        yes_sub_title="", no_sub_title="", status=status,
        yes_bid=max(1, yes_ask - 2), yes_ask=yes_ask,
        no_bid=max(1, no_ask - 2), no_ask=no_ask,
        last_price=None, volume=100, volume_24h=100, open_interest=0, result="",
        created_time=None, open_time=None,
        close_time=datetime.now(timezone.utc) + timedelta(seconds=close_in_secs),
    )


# ── run_all.py's widened series composition ─────────────────────────────────

def test_global_lock_series_never_overlaps_the_arb_scanners_own_territory():
    """Regression: the arb scanner owns ARB_SERIES exclusively (same
    reasoning MACRO_SERIES already documents) -- two threads placing
    competing orders on the same ticker is exactly the race this repo's
    TickerReservation exists to prevent between threads that DO share a
    market, but these two scanners were never meant to share one at all."""
    import run_all
    assert set(run_all.GLOBAL_LOCK_SERIES).isdisjoint(set(run_all.ARB_SERIES))


def test_global_lock_series_covers_every_bots_own_series_plus_weather():
    import run_all
    for series in run_all.CRYPTO_SERIES + run_all.MACRO_SERIES + run_all.MLB_TOTALS_SERIES:
        assert series in run_all.GLOBAL_LOCK_SERIES
    from src.trading.kalshi.weather_model import CITY_STATIONS
    for series in CITY_STATIONS:
        assert series in run_all.GLOBAL_LOCK_SERIES


# ── find_locks() ──────────────────────────────────────────────────────────────

def test_find_locks_detects_a_real_crossed_book():
    strategy = MispricingStrategy(min_close_seconds=60, max_close_seconds=3600)
    market = make_market("KXHIGHNY-26AUG15-B85.5", yes_ask=48, no_ask=49)  # 97 total -> 3c/contract guaranteed
    locks = strategy.find_locks([market], held_tickers=set())
    assert len(locks) == 1
    assert locks[0].profit_cents == 3


def test_find_locks_ignores_a_normal_book():
    strategy = MispricingStrategy(min_close_seconds=60, max_close_seconds=3600)
    market = make_market("KXHIGHNY-26AUG15-B85.5", yes_ask=52, no_ask=50)  # 102 total -- no free money
    assert strategy.find_locks([market], held_tickers=set()) == []


def test_find_locks_skips_a_held_ticker():
    strategy = MispricingStrategy(min_close_seconds=60, max_close_seconds=3600)
    market = make_market("KXHIGHNY-26AUG15-B85.5", yes_ask=48, no_ask=49)
    assert strategy.find_locks([market], held_tickers={market.ticker}) == []


def test_find_locks_skips_markets_closing_too_soon_or_too_far_out():
    strategy = MispricingStrategy(min_close_seconds=300, max_close_seconds=3600)
    too_soon = make_market("A", yes_ask=48, no_ask=49, close_in_secs=60)
    too_far = make_market("B", yes_ask=48, no_ask=49, close_in_secs=7200)
    assert strategy.find_locks([too_soon, too_far], held_tickers=set()) == []


# ── _place_locks() ──────────────────────────────────────────────────────────

class FakeLockClient:
    def __init__(self, markets, fail_no_leg_on=None):
        self._markets = {m.ticker: m for m in markets}
        self.orders: list[tuple] = []
        self.cancelled: list[str] = []
        self._fail_no_leg_on = fail_no_leg_on or set()

    def get_market(self, ticker):
        return self._markets[ticker]

    def place_order(self, ticker, side, count, yes_price):
        if side == "no" and ticker in self._fail_no_leg_on:
            raise RuntimeError("simulated NO-leg rejection")
        self.orders.append((ticker, side, count, yes_price))
        return Order(order_id=f"o-{ticker}-{side}", ticker=ticker, side=side, action="buy",
                      count=count, yes_price=yes_price, status="filled")

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        return True

    def get_positions(self):
        return []

    def get_open_orders(self):
        return []


def test_place_locks_buys_both_legs_for_a_real_lock():
    strategy = MispricingStrategy(min_close_seconds=60, max_close_seconds=3600)
    market = make_market("KXHIGHNY-26AUG15-B85.5", yes_ask=48, no_ask=49)
    client = FakeLockClient([market])
    locks = strategy.find_locks([market], held_tickers=set())

    bot = TradingBot(client=client, strategy=strategy)
    bot._place_locks(locks, held=set(), budget_cents=1000)

    assert len(client.orders) == 2
    sides = {o[1] for o in client.orders}
    assert sides == {"yes", "no"}


def test_place_locks_cancels_the_yes_leg_if_the_no_leg_fails():
    """Never leave a naked directional position from a "risk-free" lock --
    same paranoia already proven necessary elsewhere in this codebase
    (the ladder bot's own live incident history)."""
    strategy = MispricingStrategy(min_close_seconds=60, max_close_seconds=3600)
    market = make_market("KXHIGHNY-26AUG15-B85.5", yes_ask=48, no_ask=49)
    client = FakeLockClient([market], fail_no_leg_on={market.ticker})
    locks = strategy.find_locks([market], held_tickers=set())

    bot = TradingBot(client=client, strategy=strategy)
    bot._place_locks(locks, held=set(), budget_cents=1000)

    assert len(client.orders) == 1
    assert client.orders[0][1] == "yes"
    assert client.cancelled == ["o-KXHIGHNY-26AUG15-B85.5-yes"]

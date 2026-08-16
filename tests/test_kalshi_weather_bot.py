"""Tests for the Kalshi weather trading bot's scan/decide/execute loop.

Mirrors polymarket_us/apex's own bot test style: a fake client exposing the
same surface as KalshiTradingClient, and grounding seeded directly via
_forecast_cache/_calibration (bypassing the real NOAA/history I/O) the same
way Apex's own tests bypass it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.indexers.kalshi.models import Market
from src.trading.kalshi.learning import PerformanceTracker
from src.trading.kalshi.trading_client import Order, Position
from src.trading.kalshi.weather_bot import MAX_SPEND_CENTS, KalshiWeatherBot
from src.trading.kalshi.weather_model import KalshiWeatherGroundingEngine


def make_market(ticker, title, yes_bid, yes_ask, no_bid=None, no_ask=None, close_time=None):
    # Kalshi's no_bid/no_ask are independent real quotes fetched from the
    # API, NOT derived from yes_bid/yes_ask the way polymarket_us's are
    # (confirmed live 2026-08-15) -- defaulting to the complementary values
    # here is only a test convenience, and callers exercising the NO side
    # for real must pass genuinely independent values, or this helper would
    # hide the exact class of bug (a formula assuming no_bid==100-yes_ask)
    # already found and fixed in weather_bot.py.
    if no_bid is None:
        no_bid = (100 - yes_ask) if yes_ask is not None else None
    if no_ask is None:
        no_ask = (100 - yes_bid) if yes_bid is not None else None
    return Market(
        ticker=ticker, event_ticker=ticker.rsplit("-", 1)[0], market_type="binary",
        title=title, yes_sub_title="", no_sub_title="", status="open",
        yes_bid=yes_bid, yes_ask=yes_ask, no_bid=no_bid, no_ask=no_ask,
        last_price=None, volume=0, volume_24h=0, open_interest=0, result="",
        created_time=None, open_time=None, close_time=close_time,
    )


def make_grounding_engine(tmp_path, forecasts=None, calibration=None, stale=False):
    """_scan_once() always calls reload_calibration() before checking
    staleness (same ordering lesson Apex's own gate already learned) -- a
    real reload against tmp_path's nonexistent file would wipe out
    hand-seeded state, so it's stubbed to a no-op here and the "reload"
    state is set directly instead, the same way the bot-level tests focus
    on the bot's own logic, not the engine's already-separately-tested
    reload mechanism."""
    engine = KalshiWeatherGroundingEngine(history_path=tmp_path / "missing.parquet")
    engine._forecast_cache = forecasts or {}
    engine._calibration = calibration or {}
    engine._history_max_date = None if stale else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    engine.reload_calibration = lambda: None
    return engine


class FakeClient:
    def __init__(self, markets, fills=None, balance=100_000):
        self._markets = markets
        self._fills = fills or {}
        self._balance = balance
        self.positions: dict[str, int] = {}
        self.orders: list[tuple] = []
        self._cancelled: list[str] = []

    def get_balance(self):
        return self._balance

    def get_positions(self):
        return [Position(ticker=t, position=p) for t, p in self.positions.items() if p != 0]

    def get_open_orders(self):
        return []

    def list_liquid_open_markets(self, series_list=None):
        return list(self._markets)

    def place_order(self, ticker, side, count, yes_price):
        self.orders.append((ticker, side, count, yes_price))
        filled = min(count, self._fills.get(ticker, count))
        signed = filled if side == "yes" else -filled
        self.positions[ticker] = self.positions.get(ticker, 0) + signed
        status = "filled" if filled >= count else "resting"
        return Order(order_id="o-" + ticker, ticker=ticker, side=side, action="buy",
                      count=count, yes_price=yes_price, status=status)

    def cancel_order(self, order_id):
        self._cancelled.append(order_id)
        return True


TICKER = "KXHIGHNY-26AUG15-B93.5"
TITLE = "Will the maximum temperature be 93-94° on Aug 15, 2026?"


def make_bot(client, grounding, tmp_path, **kw):
    return KalshiWeatherBot(
        client=client, grounding=grounding,
        tracker=PerformanceTracker(path=tmp_path / "learning.json"),
        poll_interval=1, pending_fills_path=tmp_path / "pending.json",
        **kw,
    )


def test_scan_buys_yes_when_calibration_shows_a_trustworthy_edge(tmp_path):
    market = make_market(TICKER, TITLE, yes_bid=38, yes_ask=40)
    grounding = make_grounding_engine(
        tmp_path,
        forecasts={("KXHIGHNY", "2026-08-15"): {"forecast_mu": 93.5, "forecast_sigma": 0.01, "forecast_cycle": "c", "lead_hours": 15}},
        calibration={("KXHIGHNY", 90): {"count": 40, "actual_rate": 92.0, "mean_predicted": 99.0}},
    )
    client = FakeClient([market], fills={TICKER: 99})
    bot = make_bot(client, grounding, tmp_path)

    bot._scan_once()

    assert len(client.orders) == 1
    ticker, side, count, yes_price = client.orders[0]
    assert (ticker, side) == (TICKER, "yes")
    assert yes_price == (38 + 40) // 2


def test_scan_buys_no_using_the_real_no_bid_even_with_no_yes_side_quote(tmp_path):
    """Regression: the NO-side execution path used to price the order off
    100-yes_ask instead of the real market.no_bid it was scored with in
    _best_candidate_for_bucket_group -- inconsistent with itself, and would
    crash outright on a book like this one (no_bid/no_ask quoted, yes_ask
    genuinely absent), since calibration_trustworthy_no's own None-checks
    never require yes_ask to be present at all."""
    market = make_market(TICKER, TITLE, yes_bid=None, yes_ask=None, no_bid=58, no_ask=62)
    grounding = make_grounding_engine(
        tmp_path,
        # mu far below the bucket -> model_prob(YES) ~0 -> 1-model_prob ~1.0 -> decile 90
        forecasts={("KXHIGHNY", "2026-08-15"): {"forecast_mu": 50.0, "forecast_sigma": 0.01, "forecast_cycle": "c", "lead_hours": 15}},
        calibration={("KXHIGHNY", 90): {"count": 40, "actual_rate": 92.0, "mean_predicted": 99.0}},
    )
    client = FakeClient([market], fills={TICKER: 99})
    bot = make_bot(client, grounding, tmp_path)

    bot._scan_once()  # must not raise

    assert len(client.orders) == 1
    ticker, side, count, yes_price = client.orders[0]
    assert (ticker, side) == (TICKER, "no")
    assert yes_price == 100 - ((58 + 62) // 2)  # priced off the real no_bid/no_ask midpoint


def test_scan_passes_when_calibration_is_untrustworthy(tmp_path):
    market = make_market(TICKER, TITLE, yes_bid=38, yes_ask=40)
    grounding = make_grounding_engine(
        tmp_path,
        forecasts={("KXHIGHNY", "2026-08-15"): {"forecast_mu": 75.0, "forecast_sigma": 0.01, "forecast_cycle": "c", "lead_hours": 15}},
        calibration={},  # nothing trustworthy anywhere
    )
    client = FakeClient([market], fills={TICKER: 99})
    bot = make_bot(client, grounding, tmp_path)

    bot._scan_once()

    assert client.orders == []


def test_scan_never_stacks_a_second_position_in_the_same_city_day(tmp_path):
    other_ticker = "KXHIGHNY-26AUG15-B95.5"
    other_title = "Will the maximum temperature be 95-96° on Aug 15, 2026?"
    market = make_market(TICKER, TITLE, yes_bid=38, yes_ask=40)
    other_market = make_market(other_ticker, other_title, yes_bid=38, yes_ask=40)
    grounding = make_grounding_engine(
        tmp_path,
        forecasts={("KXHIGHNY", "2026-08-15"): {"forecast_mu": 93.5, "forecast_sigma": 0.01, "forecast_cycle": "c", "lead_hours": 15}},
        calibration={("KXHIGHNY", 90): {"count": 40, "actual_rate": 92.0, "mean_predicted": 99.0}},
    )
    client = FakeClient([market, other_market], fills={TICKER: 99, other_ticker: 99})
    client.positions[TICKER] = 5  # already held -- same (series, date) group as other_ticker
    bot = make_bot(client, grounding, tmp_path)

    bot._scan_once()

    assert client.orders == []  # other_ticker is the SAME city-day as the held TICKER


def test_stale_calibration_blocks_trading(tmp_path):
    market = make_market(TICKER, TITLE, yes_bid=38, yes_ask=40)
    grounding = make_grounding_engine(tmp_path, stale=True)
    client = FakeClient([market], fills={TICKER: 99})
    bot = make_bot(client, grounding, tmp_path)

    bot._scan_once()

    assert client.orders == []


def test_pending_fill_confirmed_records_a_bet(tmp_path):
    market = make_market(TICKER, TITLE, yes_bid=38, yes_ask=40)
    grounding = make_grounding_engine(
        tmp_path,
        forecasts={("KXHIGHNY", "2026-08-15"): {"forecast_mu": 93.5, "forecast_sigma": 0.01, "forecast_cycle": "c", "lead_hours": 15}},
        calibration={("KXHIGHNY", 90): {"count": 40, "actual_rate": 92.0, "mean_predicted": 99.0}},
    )
    # Fills fewer than requested on the first pass -- order rests.
    client = FakeClient([market], fills={TICKER: 1})
    bot = make_bot(client, grounding, tmp_path)
    bot._scan_once()
    assert len(bot._pending_fills) == 1

    # Next scan: order no longer resting (client.get_open_orders() is
    # always [] in this fake), position reflects the fill -- reconcile.
    bot._reconcile_pending_fills(client.get_positions(), client.get_open_orders())
    assert bot._pending_fills == {}
    assert len(bot.tracker._bets) == 1
    assert bot.tracker._bets[0].ticker == TICKER

"""Tests for the Kalshi bot's scan loop: stale-order handling and the halt path.

The stale-order tests exist because of a defect these caught in paper mode:
`_cancel_stale` used to decide by matching resting orders against the scan's
opportunity list, but `find_opportunities` skips every held ticker and a
ticker with a resting order is held by definition — so the match could never
succeed and EVERY resting order was cancelled one poll interval after being
placed, whatever the price had done. That silently defeated the maker
strategy (`_limit_price` posts at the mid specifically to capture spread).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from src.indexers.kalshi.models import Market
from src.trading.kalshi.bot import TradingBot
from src.trading.kalshi.risk import RiskLedger
from src.trading.kalshi.strategy import MispricingStrategy
from src.trading.kalshi.trading_client import Order, Position

TICKER = "KXBTC-26AUG3114-B63250"


def make_market(ticker=TICKER, yes_bid=53, yes_ask=55, no_ask=45, status="open"):
    return Market(
        ticker=ticker,
        event_ticker=ticker.rsplit("-", 1)[0],
        market_type="binary",
        title="",
        yes_sub_title="",
        no_sub_title="",
        status=status,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=100 - yes_ask,
        no_ask=no_ask,
        last_price=yes_ask,
        volume=5000,
        volume_24h=5000,
        open_interest=500,
        result="",
        created_time=None,
        open_time=None,
        close_time=datetime.now(timezone.utc) + timedelta(minutes=30),
    )


class FakeClient:
    """Records cancels and orders instead of sending them anywhere."""

    def __init__(self, markets=None, positions=None, open_orders=None, balance=100_000):
        self.markets = markets if markets is not None else [make_market()]
        self.positions = positions or []
        self.open_orders = open_orders or []
        self.balance = balance
        self.cancelled: list[str] = []
        self.placed: list[dict] = []

    def get_balance(self):
        return self.balance

    def get_positions(self):
        return self.positions

    def get_open_orders(self):
        return self.open_orders

    def list_liquid_open_markets(self, min_volume_24h=0, series_list=None):
        return self.markets

    def get_market(self, ticker):
        for m in self.markets:
            if m.ticker == ticker:
                return m
        raise KeyError(ticker)

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        return True

    def place_order(self, ticker, side, count, yes_price):
        self.placed.append({"ticker": ticker, "side": side, "count": count, "yes_price": yes_price})
        return Order(
            order_id=f"o{len(self.placed)}",
            ticker=ticker,
            side=side,
            action="buy",
            count=count,
            yes_price=yes_price,
            status="resting",
        )


def make_bot(client, win_rates=None, risk_ledger=None):
    strategy = MispricingStrategy(ticker_prefixes=["KXBTC"])
    strategy._win_rates = win_rates or {"yes": {30: 45.0, 55: 68.0}, "no": {}}
    bot = TradingBot(
        client=client,
        strategy=strategy,
        series_list=["KXBTC"],
        risk_ledger=risk_ledger,
    )
    # _scan_once refreshes the calibration table from disk on its first pass.
    # There is no dataset under test, so that would correctly blank the
    # injected table and make every scan a no-op. Mark it freshly loaded.
    bot._last_reload = time.monotonic()
    return bot


def resting_order(order_id="o1", side="yes", yes_price=54, ticker=TICKER):
    return Order(
        order_id=order_id,
        ticker=ticker,
        side=side,
        action="buy",
        count=10,
        yes_price=yes_price,
        status="resting",
    )


# ── Stale-order handling ──────────────────────────────────────────────────


def test_resting_order_that_still_has_edge_is_kept():
    """The regression this file exists for: an order with edge must survive
    the next scan rather than being churned."""
    client = FakeClient()
    bot = make_bot(client)
    # 54c against a ~67% interpolated win rate is comfortably above the floor.
    bot._cancel_stale([resting_order(yes_price=54)], client.markets)
    assert client.cancelled == []


def test_resting_order_whose_edge_is_gone_is_cancelled():
    client = FakeClient()
    # A win rate barely above the price leaves nothing after fees.
    bot = make_bot(client, win_rates={"yes": {30: 30.0, 55: 55.0}, "no": {}})
    bot._cancel_stale([resting_order(yes_price=54)], client.markets)
    assert client.cancelled == ["o1"]


def test_resting_order_is_cancelled_when_its_market_leaves_the_book():
    client = FakeClient(markets=[])
    bot = make_bot(client)
    bot._cancel_stale([resting_order()], markets=[])
    assert client.cancelled == ["o1"]


def test_resting_order_is_cancelled_when_no_calibration_data_covers_its_price():
    """Can't evaluate it, so don't keep sitting on it."""
    client = FakeClient()
    bot = make_bot(client, win_rates={"yes": {90: 95.0}, "no": {}})
    bot._cancel_stale([resting_order(yes_price=54)], client.markets)
    assert client.cancelled == ["o1"]


def test_cancel_stale_evaluates_the_orders_own_limit_price_not_the_touch():
    """An order rests at its own price; that price decides whether it keeps
    its edge. Here the touch (55c) has edge but the order's price (31c) sits
    in a bucket that does not."""
    client = FakeClient()
    bot = make_bot(client, win_rates={"yes": {30: 31.0, 55: 68.0}, "no": {}})
    bot._cancel_stale([resting_order(yes_price=31)], client.markets)
    assert client.cancelled == ["o1"]


def test_cancel_all_pulls_every_order_regardless_of_edge():
    client = FakeClient()
    bot = make_bot(client)
    orders = [resting_order("o1"), resting_order("o2", yes_price=50)]
    bot._cancel_all(orders, "trading halted")
    assert client.cancelled == ["o1", "o2"]


# ── Halt path ─────────────────────────────────────────────────────────────


def test_halted_scan_places_no_orders():
    client = FakeClient()
    ledger = RiskLedger(starting_bankroll_cents=100_000)
    ledger.update_bankroll(70_000)  # -30%, hard halt
    bot = make_bot(client, risk_ledger=ledger)

    bot._scan_once()
    assert client.placed == []


def test_halted_scan_pulls_resting_orders():
    """A halt that left orders resting on the book would not be a halt —
    they are still live exposure."""
    order = resting_order()
    client = FakeClient(open_orders=[order])
    ledger = RiskLedger(starting_bankroll_cents=100_000)
    ledger.update_bankroll(70_000)
    bot = make_bot(client, risk_ledger=ledger)

    bot._scan_once()
    assert client.cancelled == ["o1"]
    assert client.placed == []


def test_unhalted_scan_does_place_orders():
    """Control for the halt tests: the same setup trades when not halted."""
    client = FakeClient()
    ledger = RiskLedger(starting_bankroll_cents=100_000)
    bot = make_bot(client, risk_ledger=ledger)

    bot._scan_once()
    assert len(client.placed) == 1
    assert client.placed[0]["ticker"] == TICKER


def test_bankroll_marks_open_positions_conservatively():
    """Positions are marked at 1c/contract so the drawdown breaker cannot be
    flattered by a stale or thin quote."""
    client = FakeClient(balance=50_000)
    bot = make_bot(client)
    bankroll = bot._bankroll_cents(50_000, [Position(ticker=TICKER, position=80)])
    assert bankroll == 50_080


def test_short_positions_count_toward_bankroll_by_magnitude():
    client = FakeClient()
    bot = make_bot(client)
    assert bot._bankroll_cents(1_000, [Position(ticker=TICKER, position=-40)]) == 1_040

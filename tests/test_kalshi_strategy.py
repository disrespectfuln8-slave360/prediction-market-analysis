"""Tests for the Kalshi mispricing/lock strategy.

Focused on the rules that came out of real incidents (see
`docs/kalshi-bot-operating-manual.md` §4) and on the manual's §3 list of
things that are known false — those are the ones that cost money when they
regress, and none of them are obvious from reading the code alone.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.indexers.kalshi.models import Market
from src.trading.kalshi import risk
from src.trading.kalshi.strategy import (
    MispricingStrategy,
    event_key,
    normalize_series,
)

BANKROLL = 100_000
CASH = 100_000


def make_market(
    ticker="KXBTC-26JUL0918-B63250",
    yes_bid=48,
    yes_ask=50,
    no_ask=50,
    status="open",
    close_in_seconds=1800,
    close_time=None,
):
    # Brackets of one real-world event share an exact close time on the
    # exchange, and the dedup key depends on that. Tests that build several
    # brackets of one event must therefore pass the SAME close_time — two
    # `datetime.now()` calls microseconds apart would look like two distinct
    # events and silently defeat the very dedup under test.
    close = close_time or (datetime.now(timezone.utc) + timedelta(seconds=close_in_seconds))
    return Market(
        ticker=ticker,
        event_ticker="KXBTC-26JUL0918",
        market_type="binary",
        title="",
        yes_sub_title="",
        no_sub_title="",
        status=status,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=100 - yes_ask if yes_ask else None,
        no_ask=no_ask,
        last_price=yes_ask,
        volume=1000,
        volume_24h=1000,
        open_interest=100,
        result="",
        created_time=None,
        open_time=None,
        close_time=close,
    )


def make_strategy(win_rates=None, **kwargs):
    strat = MispricingStrategy(**kwargs)
    # Inject the calibration table directly — these tests are about what the
    # strategy DOES with a table, not about the duckdb query that builds one.
    strat._win_rates = win_rates if win_rates is not None else {"yes": {}, "no": {}}
    return strat


# ── event_key: incident #3 ────────────────────────────────────────────────


def test_event_key_groups_brackets_of_one_close_hour():
    """Adjacent BTC brackets of the same close hour are mutually exclusive
    and must share one exposure key."""
    a = event_key("KXBTC-26JUL0918-B63250")
    b = event_key("KXBTC-26JUL0918-B63350")
    assert a == b == "KXBTC-26JUL0918"


def test_event_key_does_not_collapse_distinct_games():
    """A 2-segment ticker has no bracket suffix to drop — dropping its only
    hyphen would merge every game in the series into one key."""
    ticker = "KXMLBRFI-26MAY311335MINPIT"
    assert event_key(ticker) == ticker


def test_normalize_series_collapses_sibling_underlyings():
    """Hourly, 15-minute and daily BTC resolve off the same number, so they
    must share one exposure key."""
    assert normalize_series("KXBTCD") == "KXBTC"
    assert normalize_series("KXBTC15M") == "KXBTC"
    assert normalize_series("KXETH15M") == "KXETH"


# ── No-longshot rule (manual §3) ──────────────────────────────────────────


def test_longshot_prices_are_never_traded_however_large_the_edge():
    """A 40pp edge on a 10c contract is exactly the shape of a noise-driven
    'opportunity' the manual bans."""
    strat = make_strategy(win_rates={"yes": {10: 50.0}, "no": {}})
    market = make_market(yes_bid=8, yes_ask=10, no_ask=90)
    opps = strat.find_opportunities([market], set(), BANKROLL, CASH)
    assert opps == []


def test_price_band_defaults_come_from_risk_module():
    """One definition of the band, not two that can drift apart."""
    strat = MispricingStrategy()
    assert strat.min_entry_price == risk.MIN_TRADEABLE_PRICE_CENTS
    assert strat.max_entry_price == risk.MAX_TRADEABLE_PRICE_CENTS


def test_sample_floor_defaults_to_the_manuals_value():
    """Manual §5: 500 rows. The old default of 30 is what made the
    thinnest-sampled buckets look like the best opportunities."""
    assert MispricingStrategy().min_historical_trades == 500


# ── Opportunity discovery ─────────────────────────────────────────────────


def test_finds_a_genuine_mispricing_inside_the_band():
    strat = make_strategy(win_rates={"yes": {50: 62.0}, "no": {}})
    market = make_market()
    opps = strat.find_opportunities([market], set(), BANKROLL, CASH)
    assert len(opps) == 1
    assert opps[0].side == "yes"
    assert opps[0].count > 0
    assert opps[0].stake_cents > 0


def test_edge_recorded_on_the_opportunity_is_after_fees():
    strat = make_strategy(win_rates={"yes": {50: 62.0}, "no": {}})
    opps = strat.find_opportunities([make_market()], set(), BANKROLL, CASH)
    # Raw edge would be 12pp; after the fee it must be strictly less.
    assert 0 < opps[0].edge < 12.0


def test_no_side_uses_its_own_table_not_the_inverted_yes_table():
    """Manual §3, 'known false': inverting the YES table for NO mispredicted
    NO bets by 2-3x against the live record. With a YES table that would
    imply a big NO edge and an empty NO table, nothing may be found."""
    strat = make_strategy(win_rates={"yes": {50: 95.0}, "no": {}})
    market = make_market(yes_bid=48, yes_ask=50, no_ask=50)
    opps = strat.find_opportunities([market], set(), BANKROLL, CASH)
    assert all(o.side != "no" for o in opps)


def test_held_tickers_are_skipped():
    strat = make_strategy(win_rates={"yes": {50: 62.0}, "no": {}})
    market = make_market()
    assert strat.find_opportunities([market], {market.ticker}, BANKROLL, CASH) == []


def test_sibling_bracket_of_a_held_event_is_blocked():
    """Incident #3: nothing stopped a later scan adding an adjacent bracket
    of an event already held."""
    strat = make_strategy(win_rates={"yes": {50: 62.0}, "no": {}})
    held_bracket = make_market(ticker="KXBTC-26JUL0918-B63250")
    other_bracket = make_market(ticker="KXBTC-26JUL0918-B63350")
    opps = strat.find_opportunities(
        [held_bracket, other_bracket],
        {held_bracket.ticker},
        BANKROLL,
        CASH,
        event_blocked_tickers={held_bracket.ticker},
    )
    assert opps == []


def test_only_the_best_bracket_per_event_survives_dedup():
    """Mutually exclusive brackets of one close hour: betting several means
    guaranteeing all but one lose. Only the highest-edge one may survive."""
    strat = make_strategy(win_rates={"yes": {50: 62.0, 40: 60.0}, "no": {}})
    close = datetime.now(timezone.utc) + timedelta(seconds=1800)
    a = make_market(ticker="KXBTC-26JUL0918-B63250", yes_bid=48, yes_ask=50, close_time=close)
    b = make_market(ticker="KXBTC-26JUL0918-B63350", yes_bid=38, yes_ask=40, close_time=close)
    opps = strat.find_opportunities([a, b], set(), BANKROLL, CASH)
    assert len(opps) == 1
    # 40c at a 60% win rate is a ~18pp after-fee edge; 50c at 62% is only
    # ~10pp. The survivor is the larger edge, not the higher price.
    assert opps[0].entry_price == 40


def test_closed_markets_are_skipped():
    strat = make_strategy(win_rates={"yes": {50: 62.0}, "no": {}})
    market = make_market(status="finalized")
    assert strat.find_opportunities([market], set(), BANKROLL, CASH) == []


def test_markets_closing_too_soon_are_skipped():
    strat = make_strategy(win_rates={"yes": {50: 62.0}, "no": {}})
    market = make_market(close_in_seconds=60)
    assert strat.find_opportunities([market], set(), BANKROLL, CASH) == []


def test_wide_spread_markets_are_skipped():
    strat = make_strategy(win_rates={"yes": {50: 62.0}, "no": {}})
    market = make_market(yes_bid=20, yes_ask=50)  # 30c spread
    assert strat.find_opportunities([market], set(), BANKROLL, CASH) == []


def test_one_sided_book_is_skipped():
    strat = make_strategy(win_rates={"yes": {50: 62.0}, "no": {}})
    market = make_market(yes_bid=0, yes_ask=50)
    assert strat.find_opportunities([market], set(), BANKROLL, CASH) == []


def test_empty_calibration_table_finds_nothing():
    """Manual §6.1: with no data the bot must find zero opportunities. That
    is correct behaviour, not a bug to route around."""
    strat = make_strategy(win_rates={"yes": {}, "no": {}})
    assert strat.find_opportunities([make_market()], set(), BANKROLL, CASH) == []


def test_no_bankroll_finds_nothing():
    strat = make_strategy(win_rates={"yes": {50: 62.0}, "no": {}})
    assert strat.find_opportunities([make_market()], set(), 0, 0) == []


# ── Locks ─────────────────────────────────────────────────────────────────


def test_lock_that_does_not_clear_fees_is_rejected():
    """A 99c pair is 1c gross — several cents short after both legs' fees.
    Booking it as an arb is a guaranteed loss recorded as guaranteed profit.
    """
    strat = make_strategy()
    market = make_market(yes_bid=48, yes_ask=50, no_ask=49)  # total 99
    assert strat.find_locks([market], set()) == []


def test_lock_with_a_wide_enough_spread_is_found():
    strat = make_strategy()
    market = make_market(yes_bid=38, yes_ask=40, no_ask=50)  # total 90
    locks = strat.find_locks([market], set())
    assert len(locks) == 1
    assert locks[0].count > 0


def test_lock_profit_is_reported_net_of_fees():
    strat = make_strategy()
    market = make_market(yes_bid=38, yes_ask=40, no_ask=50)  # total 90, 10c gross
    lock = strat.find_locks([market], set())[0]
    assert 0 < lock.profit_cents < 10


def test_lock_sizing_uses_the_pair_cost_not_one_leg():
    """Incident #5: sizing off max(leg) let a lock spend ~2x its budget."""
    strat = make_strategy(max_spend_cents=100)
    market = make_market(yes_bid=38, yes_ask=40, no_ask=50)  # pair costs 90
    lock = strat.find_locks([market], set())[0]
    assert (lock.yes_ask + lock.no_ask) * lock.count <= 100


def test_held_tickers_are_skipped_for_locks():
    strat = make_strategy()
    market = make_market(yes_bid=38, yes_ask=40, no_ask=50)
    assert strat.find_locks([market], {market.ticker}) == []

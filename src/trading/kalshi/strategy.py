from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import duckdb

from src.indexers.kalshi.models import Market

# Shared cache for the global (unfiltered) win-rate fallback query, keyed by
# (trades_dir, markets_dir) so isolated test paths never collide with the
# real one. run_all.py runs multiple MispricingStrategy instances (crypto,
# macro, mlb_totals, ...) as threads in one process, all pointed at the same
# data directory -- each one independently falling back to this same
# full-table scan is what caused a live OOM crash (2026-08-15) once the
# underlying dataset grew past ~85M combined rows: a category with too
# little of its own data (real macro, split out of MACRO_SERIES that same
# day) falls back on every reload, and three threads each recomputing the
# identical global aggregation needed more memory than the machine had free
# at once. Caching it once per process, shared across every instance, turns
# 3 full scans into 1 -- but only with a lock: the crash happened on cold
# startup, when all three bot threads reload within moments of each other
# and would otherwise all race an empty cache and run the expensive query
# concurrently anyway. The lock makes the second and third caller wait for
# the first's result instead of duplicating its work.
_GLOBAL_RATES_CACHE_TTL = 300.0  # seconds
_global_rates_cache: dict[tuple, tuple[float, dict]] = {}
_global_rates_lock = threading.Lock()

if TYPE_CHECKING:
    from src.trading.kalshi.learning import PerformanceTracker

logger = logging.getLogger(__name__)


@dataclass
class Opportunity:
    market: Market
    side: str        # "yes" or "no"
    entry_price: int  # cents you pay per contract
    win_rate: float  # historical win rate at this price (0-100)
    edge: float      # win_rate - entry_price in percentage points
    count: int       # contracts to buy


@dataclass
class Lock:
    market: Market
    yes_ask: int      # cents to buy YES leg
    no_ask: int       # cents to buy NO leg
    count: int        # contracts of each leg
    profit_cents: int # guaranteed profit per contract pair (100 - yes_ask - no_ask)


def event_key(ticker: str) -> str:
    """Group ticker variants (price brackets, strikes, game sides) that share
    the same underlying real-world event.

    Kalshi tickers are "SERIES-DATETIME[PLAYERS]-SUFFIX" — dropping the last
    hyphen-separated segment groups the different brackets/strikes of one
    market instance together (e.g. every BTC price bracket for the same
    close hour: "KXBTC-26JUL0918-B63250" and "KXBTC-26JUL0918-B63350" both
    key to "KXBTC-26JUL0918"). Only applies when the ticker has 3+ segments
    (2+ hyphens) — a 2-segment ticker like "KXMLBRFI-26MAY311335MINPIT" has
    no separate strike/bracket suffix to drop; dropping its only hyphen
    would collapse every distinct game in that series into one key, which
    is wrong (verified against real data: this exact series does this if
    the last-segment drop is applied unconditionally). For a 2-segment
    ticker the whole ticker is its own event_key instead.

    Known limitations, not correctness bugs — this is a heuristic, not an
    exhaustive per-series parser:
      - Doesn't group across different SERIES for the same real-world event
        (e.g. KXMLBGAME and KXMLBSPREAD tickers for the same game get
        different keys).
      - For tickers with 4+ segments where more than one trailing segment
        varies per distinct market (e.g. multi-candidate primary tickers
        like "KXCAGOVPRIMARYTOP3-26JUN02-BEC-BIA-HIL" vs
        "...-BEC-BIA-STE" — different permutations of the same election),
        dropping only the last segment does NOT merge them even though
        they're the same event. Do not add any such multi-candidate/
        permutation series to a bot's series_list without revisiting this
        function first.
    """
    n_hyphens = ticker.count("-")
    if n_hyphens < 2:
        return ticker
    parts = ticker.rsplit("-", 1)
    return parts[0]


# Series that are different wrappers around the SAME underlying — e.g. hourly
# brackets, 15-minute brackets, and daily thresholds on the BTC price. Bets
# across these series with the same close time resolve off the same number,
# so they must share one dedup/exposure key. (When ticker_prefixes contains
# e.g. "KXBTC", _ticker_series already collapses KXBTCD/KXBTC15M into it by
# prefix match; this map makes the grouping explicit and config-independent.)
SERIES_UNDERLYING = {
    "KXBTCD": "KXBTC",
    "KXBTC15M": "KXBTC",
    "KXETHD": "KXETH",
    "KXETH15M": "KXETH",
    "KXSOLD": "KXSOL",
    "KXSOL15M": "KXSOL",
}


def normalize_series(series: str) -> str:
    """Collapse sibling series onto their shared underlying."""
    return SERIES_UNDERLYING.get(series, series)


class MispricingStrategy:
    """Find open markets where current ask price is below the historical win rate.

    Loads taker win rates by price from the local parquet dataset, optionally
    restricted to a category of tickers (e.g. crypto or macro), and compares
    them against live market ask prices. Falls back to global win rates if the
    category-specific dataset is too sparse.
    """

    def __init__(
        self,
        trades_dir: Optional[Path] = None,
        markets_dir: Optional[Path] = None,
        min_edge_pp: float = 5.0,
        max_spend_cents: int = 100,
        min_historical_trades: int = 30,
        # Category filtering — list of series prefixes like ["KXBTC", "KXETH"].
        # Win rates are computed only from trades on matching tickers.
        # Falls back to global rates when fewer than 5 price points are found.
        ticker_prefixes: Optional[list[str]] = None,
        # Market quality filters
        min_entry_price: int = 1,    # full 1-99¢ range available
        max_entry_price: int = 99,
        max_spread: int = 20,        # skip if yes_ask - yes_bid > 20¢ (illiquid)
        min_close_seconds: int = 300,   # skip markets closing within 5 min
        max_close_seconds: int = 3600,  # skip markets closing more than 1hr away
        tracker: Optional["PerformanceTracker"] = None,
    ):
        base = Path(__file__).parent.parent.parent.parent
        self.trades_dir = trades_dir or base / "data" / "kalshi" / "trades"
        self.markets_dir = markets_dir or base / "data" / "kalshi" / "markets"
        self.min_edge_pp = min_edge_pp
        self.max_spend_cents = max_spend_cents
        self.min_historical_trades = min_historical_trades
        self.ticker_prefixes = ticker_prefixes
        self.min_entry_price = min_entry_price
        self.max_entry_price = max_entry_price
        self.max_spread = max_spread
        self.min_close_seconds = min_close_seconds
        self.max_close_seconds = max_close_seconds
        self.tracker = tracker
        self._win_rates: Optional[dict[str, dict[int, float]]] = None

    # ── Win rate loading ──────────────────────────────────────────────────────

    def _query_win_rates(self, prefixes: Optional[list[str]]) -> dict[str, dict[int, float]]:
        """Run the win-rate SQL query, split by taker side, optionally scoped
        to ticker prefixes.

        Split by side rather than pooled: confirmed against the live paper-
        trading record that a pooled table systematically mispredicts NO
        bets specifically (e.g. 10-29c NO bets were winning 9-12% of the
        time against a pooled-model prediction of 22-29% — the model was
        off by roughly 2-3x). A pooled table doesn't just blend two similar
        distributions, it silently assumes YES-takers and NO-takers at the
        same price have the same win rate, which the data shows is false.
        """
        prefix_clause = ""
        if prefixes:
            conditions = " OR ".join(f"ticker LIKE '{p}%'" for p in prefixes)
            prefix_clause = f"AND ({conditions})"

        with duckdb.connect() as con:
            df = con.execute(f"""
                WITH resolved AS (
                    SELECT ticker, result
                    FROM '{self.markets_dir}/*.parquet'
                    WHERE status = 'finalized' AND result IN ('yes', 'no')
                    {prefix_clause}
                ),
                taker_positions AS (
                    SELECT
                        t.taker_side,
                        CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
                        CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END AS won
                    FROM '{self.trades_dir}/*.parquet' t
                    INNER JOIN resolved m ON t.ticker = m.ticker
                )
                SELECT
                    taker_side,
                    price,
                    COUNT(*) AS n,
                    100.0 * SUM(won) / COUNT(*) AS win_rate
                FROM taker_positions
                WHERE price BETWEEN 1 AND 99
                GROUP BY taker_side, price
                HAVING COUNT(*) >= {self.min_historical_trades}
                ORDER BY taker_side, price
            """).df()

        rates: dict[str, dict[int, float]] = {"yes": {}, "no": {}}
        for _, row in df.iterrows():
            rates[row.taker_side][int(row.price)] = row.win_rate
        return rates

    def _cached_global_win_rates(self) -> dict[str, dict[int, float]]:
        """_query_win_rates(None) result, shared across every strategy
        instance pointed at the same data directory within this process.
        See the module-level cache comment for why this exists and why it's
        locked, not just memoized."""
        key = (self.trades_dir, self.markets_dir)
        with _global_rates_lock:
            cached = _global_rates_cache.get(key)
            now = time.monotonic()
            if cached is not None and now - cached[0] < _GLOBAL_RATES_CACHE_TTL:
                return cached[1]
            rates = self._query_win_rates(None)
            _global_rates_cache[key] = (now, rates)
            return rates

    def _load_win_rates(self) -> dict[str, dict[int, float]]:
        if not list(self.trades_dir.glob("*.parquet")):
            return {"yes": {}, "no": {}}

        rates = self._query_win_rates(self.ticker_prefixes)
        global_rates: Optional[dict[str, dict[int, float]]] = None

        # Fall back to global rates when category-specific data is too sparse
        # -- independently per side, since one side can be sparse while the
        # other has plenty (e.g. a series traded mostly one direction).
        for side in ("yes", "no"):
            if self.ticker_prefixes and len(rates[side]) < 5:
                if global_rates is None:
                    global_rates = self._cached_global_win_rates()
                logger.info(
                    "Only %d category-specific %s win rate points — falling back to global rates",
                    len(rates[side]), side,
                )
                rates[side] = global_rates[side]

        logger.info(
            "Win rates loaded: %d YES price points, %d NO price points",
            len(rates["yes"]), len(rates["no"]),
        )
        return rates

    @property
    def win_rates(self) -> dict[str, dict[int, float]]:
        if self._win_rates is None:
            self._win_rates = self._load_win_rates()
        return self._win_rates

    def reload_win_rates(self):
        self._win_rates = self._load_win_rates()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _ticker_series(self, ticker: str) -> str:
        """Extract the series prefix from a ticker, e.g. 'KXBTC-...' → 'KXBTC'."""
        if self.ticker_prefixes:
            for p in self.ticker_prefixes:
                if ticker.startswith(p):
                    return p
        return ticker.split("-")[0]

    def _contracts_for_spend(self, price_cents: int, series: Optional[str] = None) -> int:
        if price_cents <= 0:
            return 0
        spend = self.max_spend_cents
        if series is not None and self.tracker:
            # Scale size by this category's realized track record — a proven
            # loser gets sized down (never below 75%), a proven winner gets a
            # modest, capped boost. Never overrides the min_edge_pp gate, only
            # nudges how much to bet once a bet has already cleared it.
            spend = int(spend * self.tracker.confidence_multiplier(series))
        return spend // price_cents

    def _interpolate_win_rate(self, price: int, side: str) -> Optional[float]:
        """Return win rate at price for this side, interpolating between
        nearest known points on that side's own table."""
        table = self.win_rates.get(side, {})
        if price in table:
            return table[price]
        keys = sorted(table.keys())
        lo = [k for k in keys if k < price]
        hi = [k for k in keys if k > price]
        if not lo or not hi:
            return None
        p0, p1 = lo[-1], hi[0]
        w0, w1 = table[p0], table[p1]
        return w0 + (w1 - w0) * (price - p0) / (p1 - p0)

    # ── Signal generation ─────────────────────────────────────────────────────

    def find_opportunities(
        self,
        markets: list[Market],
        held_tickers: set[str],
        event_blocked_tickers: Optional[set[str]] = None,
    ) -> list[Opportunity]:
        """Return mispriced markets sorted by edge descending.

        held_tickers: exact tickers to never bet on (positions, open orders,
        dead/410 tickers).

        event_blocked_tickers: tickers whose whole EVENT is off-limits —
        real held exposure only, not dead tickers (a 410'd bracket must not
        block its siblings). Defaults to held_tickers. This closes the
        cross-scan leak the end-of-scan dedup can't see: dedup picks one
        bracket per event per scan, but nothing stopped scan N+1 from adding
        an adjacent (mutually exclusive) bracket of the same event once
        prices drifted — the live ledger accumulated 4 adjacent YES brackets
        on one BTC close hour, and contradictory YES>64199.99 + NO>63999.99
        threshold bets on the same close, each pair guaranteed to lose at
        least one leg.
        """
        if not self.win_rates.get("yes") and not self.win_rates.get("no"):
            return []

        blocked = event_blocked_tickers if event_blocked_tickers is not None else held_tickers
        held_events = {event_key(t) for t in blocked}
        # (normalized series, close_time) of held exposure — catches sibling
        # series on the same underlying/close (KXBTC vs KXBTCD vs KXBTC15M)
        # that event_key's string grouping can't. Only resolvable for held
        # tickers still present in this scan's market list.
        by_ticker = {m.ticker: m for m in markets}
        held_series_close: set[tuple] = set()
        for t in blocked:
            m = by_ticker.get(t)
            if m is not None and m.close_time:
                held_series_close.add(
                    (normalize_series(self._ticker_series(t)), m.close_time)
                )

        opps: list[Opportunity] = []
        for market in markets:
            if market.ticker in held_tickers:
                continue
            if market.status not in ("open", "active"):
                continue

            # Skip markets with no close time — can't enforce time window without it
            if not market.close_time:
                continue

            # Skip markets closing too soon or too far away
            now = datetime.now(tz=market.close_time.tzinfo or timezone.utc)
            secs = (market.close_time - now).total_seconds()
            if secs < self.min_close_seconds or secs > self.max_close_seconds:
                continue

            # Skip markets resolving on a future calendar day
            if market.close_time.date() > now.date():
                continue

            # Require two-sided book — one-sided means near-certain outcome
            if not (market.yes_bid and market.yes_ask
                    and market.yes_bid > 0 and market.yes_ask > 0):
                continue

            # Skip illiquid markets with wide bid-ask spread
            if (market.yes_ask - market.yes_bid) > self.max_spread:
                continue

            series = self._ticker_series(market.ticker)

            # One position per real-world event at a time (see docstring) —
            # never add a second bracket/strike/side of an event we already
            # have exposure on.
            if event_key(market.ticker) in held_events:
                continue
            if (normalize_series(series), market.close_time) in held_series_close:
                continue

            # Circuit breaker — skip this whole category if it's proven a
            # loser across all sides/prices, not just underperforming in one
            # price bucket (which get_edge_adjustment below already handles).
            if self.tracker:
                block_reason = self.tracker.is_category_blocked(series)
                if block_reason:
                    logger.debug("Skipping %s: %s", market.ticker, block_reason)
                    continue

            # YES side: pay yes_ask, win if YES resolves
            if self.min_entry_price <= market.yes_ask <= self.max_entry_price:
                wr_yes = self._interpolate_win_rate(market.yes_ask, "yes")
                if wr_yes is not None:
                    edge = wr_yes - market.yes_ask
                    # Apply learned adjustment — raises required edge if this
                    # (series, side, price bucket) has been underperforming.
                    learned_adj = (
                        self.tracker.get_edge_adjustment(series, "yes", market.yes_ask)
                        if self.tracker else 0.0
                    )
                    if edge >= self.min_edge_pp + learned_adj:
                        count = self._contracts_for_spend(market.yes_ask, series)
                        if count > 0:
                            opps.append(Opportunity(
                                market=market,
                                side="yes",
                                entry_price=market.yes_ask,
                                win_rate=wr_yes,
                                edge=edge,
                                count=count,
                            ))

            # NO side: pay no_ask, win if NO resolves. Looked up directly on
            # the NO-takers' own table now (see _query_win_rates docstring)
            # — previously this inverted the YES table via 100-no_ask, which
            # implicitly assumed YES-takers and NO-takers at complementary
            # prices win at complementary rates. Real data shows they don't.
            if market.no_ask and self.min_entry_price <= market.no_ask <= self.max_entry_price:
                no_wr = self._interpolate_win_rate(market.no_ask, "no")
                if no_wr is not None:
                    edge = no_wr - market.no_ask
                    learned_adj = (
                        self.tracker.get_edge_adjustment(series, "no", market.no_ask)
                        if self.tracker else 0.0
                    )
                    if edge >= self.min_edge_pp + learned_adj:
                        count = self._contracts_for_spend(market.no_ask, series)
                        if count > 0:
                            opps.append(Opportunity(
                                market=market,
                                side="no",
                                entry_price=market.no_ask,
                                win_rate=no_wr,
                                edge=edge,
                                count=count,
                            ))

        # Dedup: for each (underlying, close_time) keep only the highest-edge
        # opportunity. Price brackets for the same expiry are mutually exclusive —
        # BTC can only close in one bracket, so betting on multiple adjacent
        # brackets means you're guaranteed to lose all but one. Same applies to
        # both sides of a sports game. Series are normalized so sibling series
        # on the same underlying (hourly/15-min/daily BTC) dedup together.
        best_per_event: dict[tuple, Opportunity] = {}
        for opp in opps:
            series = normalize_series(self._ticker_series(opp.market.ticker))
            close = opp.market.close_time
            key = (series, close)
            if key not in best_per_event or opp.edge > best_per_event[key].edge:
                best_per_event[key] = opp

        return sorted(best_per_event.values(), key=lambda o: o.edge, reverse=True)

    def find_locks(self, markets: list[Market], held_tickers: set[str]) -> list[Lock]:
        """Find markets where yes_ask + no_ask < 99 — guaranteed arbitrage profit."""
        locks = []
        for market in markets:
            if market.ticker in held_tickers:
                continue
            if market.status not in ("open", "active"):
                continue
            if not market.close_time:
                continue
            now = datetime.now(tz=market.close_time.tzinfo or timezone.utc)
            secs = (market.close_time - now).total_seconds()
            if secs < self.min_close_seconds or secs > self.max_close_seconds:
                continue
            if market.close_time.date() > now.date():
                continue
            if not (market.yes_ask and market.no_ask and market.yes_ask > 0 and market.no_ask > 0):
                continue
            total = market.yes_ask + market.no_ask
            if total < 100:  # guaranteed ≥1¢ profit per contract pair
                count = self._contracts_for_spend(max(market.yes_ask, market.no_ask))
                if count > 0:
                    locks.append(Lock(
                        market=market,
                        yes_ask=market.yes_ask,
                        no_ask=market.no_ask,
                        count=count,
                        profit_cents=100 - total,
                    ))
        return sorted(locks, key=lambda l: l.profit_cents, reverse=True)

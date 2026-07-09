from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import duckdb

from src.indexers.kalshi.models import Market

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
        self._win_rates: Optional[dict[int, float]] = None

    # ── Win rate loading ──────────────────────────────────────────────────────

    def _query_win_rates(self, prefixes: Optional[list[str]]) -> dict[int, float]:
        """Run the win-rate SQL query, optionally scoped to ticker prefixes."""
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
                        CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
                        CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END AS won
                    FROM '{self.trades_dir}/*.parquet' t
                    INNER JOIN resolved m ON t.ticker = m.ticker
                )
                SELECT
                    price,
                    COUNT(*) AS n,
                    100.0 * SUM(won) / COUNT(*) AS win_rate
                FROM taker_positions
                WHERE price BETWEEN 1 AND 99
                GROUP BY price
                HAVING COUNT(*) >= {self.min_historical_trades}
                ORDER BY price
            """).df()

        return {int(row.price): row.win_rate for _, row in df.iterrows()}

    def _load_win_rates(self) -> dict[int, float]:
        if not list(self.trades_dir.glob("*.parquet")):
            return {}

        rates = self._query_win_rates(self.ticker_prefixes)

        # Fall back to global rates when category-specific data is too sparse
        if self.ticker_prefixes and len(rates) < 5:
            logger.info(
                "Only %d category-specific win rate points — falling back to global rates",
                len(rates),
            )
            rates = self._query_win_rates(None)

        logger.info("Win rates loaded: %d price points", len(rates))
        return rates

    @property
    def win_rates(self) -> dict[int, float]:
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

    def _contracts_for_spend(self, price_cents: int) -> int:
        if price_cents <= 0:
            return 0
        return self.max_spend_cents // price_cents

    def _interpolate_win_rate(self, price: int) -> Optional[float]:
        """Return win rate at price, interpolating between nearest known points."""
        if price in self.win_rates:
            return self.win_rates[price]
        keys = sorted(self.win_rates.keys())
        lo = [k for k in keys if k < price]
        hi = [k for k in keys if k > price]
        if not lo or not hi:
            return None
        p0, p1 = lo[-1], hi[0]
        w0, w1 = self.win_rates[p0], self.win_rates[p1]
        return w0 + (w1 - w0) * (price - p0) / (p1 - p0)

    # ── Signal generation ─────────────────────────────────────────────────────

    def find_opportunities(
        self, markets: list[Market], held_tickers: set[str]
    ) -> list[Opportunity]:
        """Return mispriced markets sorted by edge descending."""
        if not self.win_rates:
            return []

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

            # YES side: pay yes_ask, win if YES resolves
            if self.min_entry_price <= market.yes_ask <= self.max_entry_price:
                wr_yes = self._interpolate_win_rate(market.yes_ask)
                if wr_yes is not None:
                    edge = wr_yes - market.yes_ask
                    # Apply learned adjustment — raises required edge if this
                    # (series, side, price bucket) has been underperforming.
                    learned_adj = (
                        self.tracker.get_edge_adjustment(series, "yes", market.yes_ask)
                        if self.tracker else 0.0
                    )
                    if edge >= self.min_edge_pp + learned_adj:
                        count = self._contracts_for_spend(market.yes_ask)
                        if count > 0:
                            opps.append(Opportunity(
                                market=market,
                                side="yes",
                                entry_price=market.yes_ask,
                                win_rate=wr_yes,
                                edge=edge,
                                count=count,
                            ))

            # NO side: pay no_ask, win if NO resolves
            if market.no_ask and self.min_entry_price <= market.no_ask <= self.max_entry_price:
                yes_equiv = 100 - market.no_ask
                no_wr_raw = self._interpolate_win_rate(yes_equiv)
                if no_wr_raw is not None:
                    no_wr = 100.0 - no_wr_raw
                    edge = no_wr - market.no_ask
                    learned_adj = (
                        self.tracker.get_edge_adjustment(series, "no", market.no_ask)
                        if self.tracker else 0.0
                    )
                    if edge >= self.min_edge_pp + learned_adj:
                        count = self._contracts_for_spend(market.no_ask)
                        if count > 0:
                            opps.append(Opportunity(
                                market=market,
                                side="no",
                                entry_price=market.no_ask,
                                win_rate=no_wr,
                                edge=edge,
                                count=count,
                            ))

        # Dedup: for each (series, close_time) keep only the highest-edge opportunity.
        # Price brackets for the same expiry are mutually exclusive — BTC can only close
        # in one bracket, so betting on multiple adjacent brackets means you're guaranteed
        # to lose all but one. Same applies to both sides of a sports game.
        best_per_event: dict[tuple, Opportunity] = {}
        for opp in opps:
            series = self._ticker_series(opp.market.ticker)
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

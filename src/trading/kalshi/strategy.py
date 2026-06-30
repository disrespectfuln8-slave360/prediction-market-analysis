from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import duckdb

from src.indexers.kalshi.models import Market


@dataclass
class Opportunity:
    market: Market
    side: str        # "yes" or "no"
    entry_price: int  # cents you pay per contract
    win_rate: float  # historical win rate at this price (0-100)
    edge: float      # win_rate - entry_price in percentage points
    count: int       # contracts to buy


class MispricingStrategy:
    """Find open markets where current ask price is below the historical win rate.

    Loads taker win rates by price from the local parquet dataset and compares
    them against live market ask prices. Only generates signals where the edge
    exceeds min_edge_pp and there are enough historical trades to trust the data.
    """

    def __init__(
        self,
        trades_dir: Optional[Path] = None,
        markets_dir: Optional[Path] = None,
        min_edge_pp: float = 5.0,
        max_spend_cents: int = 500,
        min_historical_trades: int = 50,
        ticker_filter: Optional[str] = None,
    ):
        base = Path(__file__).parent.parent.parent.parent
        self.trades_dir = trades_dir or base / "data" / "kalshi" / "trades"
        self.markets_dir = markets_dir or base / "data" / "kalshi" / "markets"
        self.min_edge_pp = min_edge_pp
        self.max_spend_cents = max_spend_cents
        self.min_historical_trades = min_historical_trades
        self.ticker_filter = ticker_filter  # e.g. "BTC" to restrict to BTC markets
        self._win_rates: Optional[dict[int, float]] = None

    def _load_win_rates(self) -> dict[int, float]:
        parquets = list(self.trades_dir.glob("*.parquet"))
        if not parquets:
            return {}

        con = duckdb.connect()
        df = con.execute(f"""
            WITH resolved AS (
                SELECT ticker, result
                FROM '{self.markets_dir}/*.parquet'
                WHERE status = 'finalized' AND result IN ('yes', 'no')
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

    @property
    def win_rates(self) -> dict[int, float]:
        if self._win_rates is None:
            self._win_rates = self._load_win_rates()
        return self._win_rates

    def reload_win_rates(self):
        self._win_rates = self._load_win_rates()

    def _contracts_for_spend(self, price_cents: int) -> int:
        if price_cents <= 0:
            return 0
        return max(1, self.max_spend_cents // price_cents)

    def _interpolate_win_rate(self, price: int) -> Optional[float]:
        """Return win rate for price, interpolating between nearest known points."""
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

    def find_opportunities(
        self, markets: list[Market], held_tickers: set[str]
    ) -> list[Opportunity]:
        """Return mispriced markets sorted by edge descending."""
        if not self.win_rates:
            return []

        opps: list[Opportunity] = []
        for market in markets:
            if market.ticker in held_tickers or market.status != "open":
                continue
            if self.ticker_filter and self.ticker_filter.upper() not in market.ticker.upper():
                continue

            # YES side: pay yes_ask, win if YES resolves
            if market.yes_ask and market.yes_ask > 0:
                wr = self._interpolate_win_rate(market.yes_ask)
                if wr is not None:
                    edge = wr - market.yes_ask
                    if edge >= self.min_edge_pp:
                        opps.append(Opportunity(
                            market=market,
                            side="yes",
                            entry_price=market.yes_ask,
                            win_rate=wr,
                            edge=edge,
                            count=self._contracts_for_spend(market.yes_ask),
                        ))

            # NO side: pay no_ask, win if NO resolves
            if market.no_ask and market.no_ask > 0:
                yes_equiv = 100 - market.no_ask
                no_wr_raw = self._interpolate_win_rate(yes_equiv)
                if no_wr_raw is not None:
                    no_wr = 100.0 - no_wr_raw
                    edge = no_wr - market.no_ask
                    if edge >= self.min_edge_pp:
                        opps.append(Opportunity(
                            market=market,
                            side="no",
                            entry_price=market.no_ask,
                            win_rate=no_wr,
                            edge=edge,
                            count=self._contracts_for_spend(market.no_ask),
                        ))

        return sorted(opps, key=lambda o: o.edge, reverse=True)

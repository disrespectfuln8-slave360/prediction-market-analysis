from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import duckdb

from src.trading.polymarket.models import Market

if TYPE_CHECKING:
    from src.trading.polymarket.learning import PerformanceTracker

logger = logging.getLogger(__name__)


@dataclass
class TradeOpp:
    market: Market
    outcome: str      # outcome name to buy, e.g. "Yes" / "No"
    entry_price: int  # cents you pay per share
    win_rate: float    # historical win rate at this price (0-100)
    edge: float        # win_rate - entry_price in percentage points
    count: int         # shares to buy


class MispricingStrategy:
    """Find open Polymarket markets where the ask price is below the
    historical win rate observed at that price on-chain.

    Historical win rates are computed once from the local blockchain-trade
    parquet dataset as a single symmetric price→win-rate table (a taker who
    pays P¢ for either outcome — the trade's own side, not just "yes") —
    mirroring the Kalshi strategy so both outcomes of a market can reuse one
    calibration curve via the complement (100 - price).

    Polymarket's stored trade/market parquet schema (raw blockchain events)
    has no category/tag column, so `keywords` filters historical trades by
    substring match against the market question text instead — the closest
    available proxy for Kalshi's ticker-prefix category filter. Falls back
    to the global (unfiltered) win rates when too few keyword-scoped points
    are found.
    """

    def __init__(
        self,
        trades_dir: Optional[Path] = None,
        markets_dir: Optional[Path] = None,
        min_edge_pp: float = 0.5,
        max_spend_cents: int = 100,
        min_historical_trades: int = 30,
        keywords: Optional[list[str]] = None,
        min_entry_price: int = 18,
        max_entry_price: int = 55,
        max_spread: int = 20,
        min_close_seconds: int = 60,
        max_close_seconds: int = 3600,
        tracker: Optional["PerformanceTracker"] = None,
    ):
        base = Path(__file__).parent.parent.parent.parent
        self.trades_dir = trades_dir or base / "data" / "polymarket" / "trades"
        self.markets_dir = markets_dir or base / "data" / "polymarket" / "markets"
        self.min_edge_pp = min_edge_pp
        self.max_spend_cents = max_spend_cents
        self.min_historical_trades = min_historical_trades
        self.keywords = keywords
        self.min_entry_price = min_entry_price
        self.max_entry_price = max_entry_price
        self.max_spread = max_spread
        self.min_close_seconds = min_close_seconds
        self.max_close_seconds = max_close_seconds
        self.tracker = tracker
        self._win_rates: Optional[dict[int, float]] = None

    # ── Win rate loading ──────────────────────────────────────────────────────

    def _query_win_rates(self, keywords: Optional[list[str]]) -> dict[int, float]:
        """Symmetric price→win-rate table from resolved binary markets.

        For every closed market where one outcome settled to (near) 1 and the
        other to (near) 0, every on-chain trade contributes a (price, won)
        row for both the taker (buyer) and the counterparty (seller) — the
        seller "took" the opposite side at 100 - price. Mirrors the proven
        query in src/analysis/polymarket/polymarket_win_rate_by_price.py.
        """
        keyword_clause = ""
        if keywords:
            conditions = " OR ".join(f"question ILIKE '%{kw}%'" for kw in keywords)
            keyword_clause = f"AND ({conditions})"

        with duckdb.connect() as con:
            df = con.execute(f"""
                WITH candidates AS (
                    SELECT
                        json_extract_string(clob_token_ids, '$[0]') AS token0,
                        json_extract_string(clob_token_ids, '$[1]') AS token1,
                        TRY_CAST(json_extract_string(outcome_prices, '$[0]') AS DOUBLE) AS p0,
                        TRY_CAST(json_extract_string(outcome_prices, '$[1]') AS DOUBLE) AS p1
                    FROM '{self.markets_dir}/*.parquet'
                    WHERE closed = true
                    {keyword_clause}
                ),
                resolved AS (
                    SELECT token0, token1, (p0 > 0.99) AS outcome0_won
                    FROM candidates
                    WHERE p0 IS NOT NULL AND p1 IS NOT NULL
                      AND ((p0 > 0.99 AND p1 < 0.01) OR (p0 < 0.01 AND p1 > 0.99))
                ),
                token_resolution AS (
                    SELECT token0 AS token_id, outcome0_won AS won FROM resolved WHERE token0 IS NOT NULL
                    UNION ALL
                    SELECT token1 AS token_id, NOT outcome0_won AS won FROM resolved WHERE token1 IS NOT NULL
                ),
                taker_positions AS (
                    SELECT
                        CASE WHEN t.maker_asset_id = '0'
                             THEN ROUND(100.0 * t.maker_amount / t.taker_amount)
                             ELSE ROUND(100.0 * t.taker_amount / t.maker_amount)
                        END AS price,
                        tr.won AS won
                    FROM '{self.trades_dir}/*.parquet' t
                    INNER JOIN token_resolution tr
                        ON (CASE WHEN t.maker_asset_id = '0' THEN t.taker_asset_id ELSE t.maker_asset_id END) = tr.token_id
                    WHERE t.taker_amount > 0 AND t.maker_amount > 0

                    UNION ALL

                    SELECT
                        CASE WHEN t.maker_asset_id = '0'
                             THEN ROUND(100.0 - 100.0 * t.maker_amount / t.taker_amount)
                             ELSE ROUND(100.0 - 100.0 * t.taker_amount / t.maker_amount)
                        END AS price,
                        NOT tr.won AS won
                    FROM '{self.trades_dir}/*.parquet' t
                    INNER JOIN token_resolution tr
                        ON (CASE WHEN t.maker_asset_id = '0' THEN t.taker_asset_id ELSE t.maker_asset_id END) = tr.token_id
                    WHERE t.taker_amount > 0 AND t.maker_amount > 0
                )
                SELECT
                    price,
                    COUNT(*) AS n,
                    100.0 * SUM(CASE WHEN won THEN 1 ELSE 0 END) / COUNT(*) AS win_rate
                FROM taker_positions
                WHERE price BETWEEN 1 AND 99
                GROUP BY price
                HAVING COUNT(*) >= {self.min_historical_trades}
                ORDER BY price
            """).df()

        return {int(row.price): row.win_rate for _, row in df.iterrows()}

    def _load_win_rates(self) -> dict[int, float]:
        if not list(self.trades_dir.glob("*.parquet")) or not list(self.markets_dir.glob("*.parquet")):
            return {}

        rates = self._query_win_rates(self.keywords)

        if self.keywords and len(rates) < 5:
            logger.info(
                "Only %d keyword-scoped win rate points — falling back to global rates",
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

    def _market_series(self, market: Market) -> str:
        """Label used for learning/tracker bucketing — first gamma tag, if any."""
        return market.tags[0] if market.tags else "unknown"

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
        self, markets: list[Market], held_ids: set[str]
    ) -> list[TradeOpp]:
        """Return mispriced markets sorted by edge descending."""
        if not self.win_rates:
            return []

        opps: list[TradeOpp] = []
        for market in markets:
            if market.condition_id in held_ids:
                continue
            if not market.active or market.closed or market.archived:
                continue
            if len(market.outcomes) != 2:
                continue
            if not market.end_time:
                continue

            now = datetime.now(tz=market.end_time.tzinfo or timezone.utc)
            secs = (market.end_time - now).total_seconds()
            if secs < self.min_close_seconds or secs > self.max_close_seconds:
                continue
            if market.end_time.date() > now.date():
                continue

            if not (market.yes_bid and market.yes_ask
                    and market.yes_bid > 0 and market.yes_ask > 0):
                continue
            if (market.yes_ask - market.yes_bid) > self.max_spread:
                continue

            series = self._market_series(market)

            # First outcome ("Yes"-equivalent): pay yes_ask, win if it resolves.
            if self.min_entry_price <= market.yes_ask <= self.max_entry_price:
                wr = self._interpolate_win_rate(market.yes_ask)
                if wr is not None:
                    edge = wr - market.yes_ask
                    learned_adj = (
                        self.tracker.get_edge_adjustment(series, "yes", market.yes_ask)
                        if self.tracker else 0.0
                    )
                    if edge >= self.min_edge_pp + learned_adj:
                        count = self._contracts_for_spend(market.yes_ask)
                        if count > 0:
                            opps.append(TradeOpp(
                                market=market,
                                outcome=market.outcomes[0],
                                entry_price=market.yes_ask,
                                win_rate=wr,
                                edge=edge,
                                count=count,
                            ))

            # Second outcome ("No"-equivalent): pay no_ask, win if it resolves.
            # Reuses the same calibration curve via the complement price, same
            # trick the Kalshi strategy uses to pool sparse per-side data.
            if market.no_ask and self.min_entry_price <= market.no_ask <= self.max_entry_price:
                yes_equiv = 100 - market.no_ask
                wr_raw = self._interpolate_win_rate(yes_equiv)
                if wr_raw is not None:
                    wr = 100.0 - wr_raw
                    edge = wr - market.no_ask
                    learned_adj = (
                        self.tracker.get_edge_adjustment(series, "no", market.no_ask)
                        if self.tracker else 0.0
                    )
                    if edge >= self.min_edge_pp + learned_adj:
                        count = self._contracts_for_spend(market.no_ask)
                        if count > 0:
                            opps.append(TradeOpp(
                                market=market,
                                outcome=market.outcomes[1],
                                entry_price=market.no_ask,
                                win_rate=wr,
                                edge=edge,
                                count=count,
                            ))

        # Dedup: Yes/No on the same market are mutually exclusive — only the
        # higher-edge side is worth taking.
        best_per_market: dict[str, TradeOpp] = {}
        for opp in opps:
            key = opp.market.condition_id
            if key not in best_per_market or opp.edge > best_per_market[key].edge:
                best_per_market[key] = opp

        return sorted(best_per_market.values(), key=lambda o: o.edge, reverse=True)

"""Signal generation for the Kalshi bots — calibration mispricing and locks.

Governed by `docs/kalshi-bot-operating-manual.md`. Two theses live here and
they are NOT equally sound, which the manual (§3) is explicit about:

  `find_opportunities`  Calibration mispricing. Rests on the ASSUMPTION that
                        a historical win-rate-by-price table predicts future
                        outcomes. Unproven, and the reason every gate in
                        risk.py is conservative.

  `find_locks`          Structural arbitrage. The payoff arithmetic is
                        proven; whether both legs fill is not.

Sizing is not decided here. Every opportunity is passed through
`risk.evaluate`, so the manual's price band, after-fee edge floor, Kelly
caps and cash reserve are enforced on one code path rather than restated
per call site.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from src.indexers.kalshi.models import Market
from src.trading.kalshi import risk
from src.trading.kalshi.fees import taker_fee_cents

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
    side: str  # "yes" or "no"
    entry_price: int  # cents you pay per contract
    win_rate: float  # historical win rate at this price (0-100)
    edge: float  # AFTER-FEE edge in pp, over the fee-inclusive breakeven
    count: int  # contracts to buy
    stake_cents: int = 0  # total cash committed, fee included


@dataclass
class Lock:
    market: Market
    yes_ask: int  # cents to buy YES leg
    no_ask: int  # cents to buy NO leg
    count: int  # contracts of each leg
    profit_cents: int  # NET profit per contract pair, after both legs' fees


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
        trades_dir: Path | None = None,
        markets_dir: Path | None = None,
        # Extra edge required on top of risk.MIN_EDGE_PP. The after-fee floor
        # itself lives in risk.py (manual §5) — this only tightens it further
        # for a caller that wants to be pickier, never loosens it.
        min_edge_pp: float = 0.0,
        max_spend_cents: int = 100,
        # Manual §5: 500 rows per (side, price) bucket. The prior value of 30
        # is what let the thinnest-sampled buckets look like the best
        # opportunities — see the incident note in apex/risk.py.
        min_historical_trades: int = 500,
        # Category filtering — list of series prefixes like ["KXBTC", "KXETH"].
        # Win rates are computed only from trades on matching tickers.
        # Falls back to global rates when fewer than 5 price points are found.
        ticker_prefixes: list[str] | None = None,
        # Market quality filters. The band is the manual's standing no-longshot
        # rule (§3) — sourced from risk.py so there is exactly one definition.
        min_entry_price: int = risk.MIN_TRADEABLE_PRICE_CENTS,
        max_entry_price: int = risk.MAX_TRADEABLE_PRICE_CENTS,
        max_spread: int = 20,  # skip if yes_ask - yes_bid > 20¢ (illiquid)
        min_close_seconds: int = 300,  # skip markets closing within 5 min
        max_close_seconds: int = 3600,  # skip markets closing more than 1hr away
        tracker: PerformanceTracker | None = None,
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
        self._win_rates: dict[str, dict[int, float]] | None = None

    # ── Win rate loading ──────────────────────────────────────────────────────

    def _query_win_rates(self, prefixes: list[str] | None) -> dict[str, dict[int, float]]:
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
        global_rates: dict[str, dict[int, float]] | None = None

        # Fall back to global rates when category-specific data is too sparse
        # -- independently per side, since one side can be sparse while the
        # other has plenty (e.g. a series traded mostly one direction).
        for side in ("yes", "no"):
            if self.ticker_prefixes and len(rates[side]) < 5:
                if global_rates is None:
                    global_rates = self._cached_global_win_rates()
                logger.info(
                    "Only %d category-specific %s win rate points — falling back to global rates",
                    len(rates[side]),
                    side,
                )
                rates[side] = global_rates[side]

        logger.info(
            "Win rates loaded: %d YES price points, %d NO price points",
            len(rates["yes"]),
            len(rates["no"]),
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

    def _contracts_for_spend(self, price_cents: int, series: str | None = None) -> int:
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

    def _interpolate_win_rate(self, price: int, side: str) -> float | None:
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

    def _evaluate_side(
        self,
        market: Market,
        side: str,
        price: int,
        series: str,
        bankroll_cents: int,
        cash_cents: int,
    ) -> Opportunity | None:
        """Run one side of one market through the calibration table and then
        the risk gates. Returns None unless every gate passes.

        The NO side is looked up on the NO-takers' own table, never inverted
        from the YES table via 100-price: the manual (§3, "known false")
        records that pooled/inverted rates mispredicted NO bets by 2-3x
        against the live paper record.
        """
        if not (self.min_entry_price <= price <= self.max_entry_price):
            return None

        win_rate = self._interpolate_win_rate(price, side)
        if win_rate is None:
            return None

        # The learned adjustment raises the bar for a (series, side, bucket)
        # that has been underperforming. It only ever tightens.
        learned_adj = self.tracker.get_edge_adjustment(series, side, price) if self.tracker else 0.0

        decision = risk.evaluate(
            win_rate_pct=win_rate,
            price_cents=price,
            bankroll_cents=bankroll_cents,
            cash_cents=cash_cents,
            size_multiplier=(self.tracker.confidence_multiplier(series) if self.tracker else 1.0),
        )
        if decision.action != "buy":
            logger.debug("%s %s@%dc: %s", market.ticker, side.upper(), price, decision.reason)
            return None

        # Caller-supplied extra strictness on top of the risk floor.
        if decision.edge_pp < self.min_edge_pp + learned_adj:
            logger.debug(
                "%s %s@%dc: after-fee edge %.1fpp below required %.1fpp (learned +%.1f)",
                market.ticker,
                side.upper(),
                price,
                decision.edge_pp,
                self.min_edge_pp + learned_adj,
                learned_adj,
            )
            return None

        return Opportunity(
            market=market,
            side=side,
            entry_price=price,
            win_rate=win_rate,
            edge=decision.edge_pp,
            count=decision.count,
            stake_cents=decision.stake_cents,
        )

    def find_opportunities(
        self,
        markets: list[Market],
        held_tickers: set[str],
        bankroll_cents: int,
        cash_cents: int,
        event_blocked_tickers: set[str] | None = None,
    ) -> list[Opportunity]:
        """Return mispriced markets sorted by edge descending.

        bankroll_cents/cash_cents are required because sizing is Kelly-based
        against the live account (manual §5), not a fixed per-bet budget —
        a flat stake ignores both the account's size and the fact that a 4pp
        edge and a 20pp edge do not deserve the same money.

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
                held_series_close.add((normalize_series(self._ticker_series(t)), m.close_time))

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
            if not (market.yes_bid and market.yes_ask and market.yes_bid > 0 and market.yes_ask > 0):
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

            # YES side: pay yes_ask, win if YES resolves.
            yes_opp = self._evaluate_side(market, "yes", market.yes_ask, series, bankroll_cents, cash_cents)
            if yes_opp is not None:
                opps.append(yes_opp)

            # NO side: pay no_ask, win if NO resolves.
            if market.no_ask:
                no_opp = self._evaluate_side(market, "no", market.no_ask, series, bankroll_cents, cash_cents)
                if no_opp is not None:
                    opps.append(no_opp)

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
        """Find markets where both legs together cost less than the $1 they
        pay out, after fees.

        Guaranteed profit ONLY if both legs actually fill at these prices.
        A filled first leg with an unfilled second is an unhedged directional
        bet that was sized as if it were riskless — the manual (§3) flags
        this as the widest gap between "provably profitable" and "actually
        profitable" in this system, and §6.4 requires measuring the
        single-leg-fill rate on paper before any live use.
        """
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
            # Fees are charged on BOTH legs, so the gross spread has to clear
            # them before a "lock" is profitable at all. Checking `total < 100`
            # alone (what this did previously) books a 99c pair as guaranteed
            # profit when Kalshi's fee on the two legs is several cents --
            # a guaranteed *loss* recorded as an arb. The fee is computed per
            # leg at count=1 and multiplied up, matching how the exchange
            # rounds each order separately rather than netting them.
            fee_per_pair = taker_fee_cents(market.yes_ask, 1) + taker_fee_cents(market.no_ask, 1)
            net_profit = 100 - total - fee_per_pair
            if net_profit > 0:
                # Cost per contract PAIR is both legs together (total), not
                # just one leg -- sizing off max(yes_ask, no_ask) alone let
                # a lock spend up to ~2x max_spend_cents (e.g. 48+49=97:
                # max(48,49)=49 -> 100//49=2 contracts -> real cost 2*97=194c,
                # not the intended <=100c). polymarket_us's own _size_lock
                # already gets this right (buy_price + (100-sell_price));
                # this was the one place on this platform that didn't.
                count = self._contracts_for_spend(total)
                if count > 0:
                    locks.append(
                        Lock(
                            market=market,
                            yes_ask=market.yes_ask,
                            no_ask=market.no_ask,
                            count=count,
                            profit_cents=net_profit,
                        )
                    )
        return sorted(locks, key=lambda lock: lock.profit_cents, reverse=True)

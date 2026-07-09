"""Live-price convergence bot for Polymarket crypto threshold markets.

Identical design to src/trading/kalshi/convergence_bot.py:

Every `poll_interval` seconds:
  1. Fetch live BTC / ETH / SOL / DOGE spot prices (Binance primary,
     CoinGecko fallback, cached 8s to stay within rate limits).
  2. Scan crypto markets closing in the next `min_close_secs`-`max_close_secs`.
  3. Parse the threshold and direction from the market question text.
  4. Compute P(YES wins) under a zero-drift log-normal diffusion model:
       P(S_T > K | S_0=S) = N( ln(S/K) / (vol_hourly * sqrt(T_hrs)) )
  5. If P(YES) >= min_prob AND yes_ask <= max_entry_price -> BUY first outcome.
     If P(YES) <= 1 - min_prob AND no_ask <= max_entry_price -> BUY second outcome.

Polymarket questions are free text (no ticker suffix like Kalshi's "T80000"),
so coin identification and threshold/direction both come from the question
string alone.
"""
from __future__ import annotations

import logging
import math
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx

from src.trading.polymarket.models import Market
from src.trading.polymarket.trading_client import PolymarketTradingClient

logger = logging.getLogger(__name__)

# ── Volatility model ──────────────────────────────────────────────────────────
# Per-hour log-return sigma. Rounded up vs empirical to suppress borderline
# signals and partially account for fat tails / jump risk.

HOURLY_VOL: dict[str, float] = {
    "BTC":  0.018,
    "ETH":  0.025,
    "SOL":  0.035,
    "DOGE": 0.045,
}

_COIN_PATTERNS: dict[str, re.Pattern] = {
    "BTC":  re.compile(r"\b(bitcoin|btc)\b", re.IGNORECASE),
    "ETH":  re.compile(r"\b(ethereum|eth)\b", re.IGNORECASE),
    "SOL":  re.compile(r"\b(solana|sol)\b", re.IGNORECASE),
    "DOGE": re.compile(r"\b(dogecoin|doge)\b", re.IGNORECASE),
}

DEFAULT_TAG_SLUGS      = ["crypto"]
DEFAULT_MIN_PROB       = 0.93
DEFAULT_MAX_PRICE      = 92
DEFAULT_MIN_EDGE_PP    = 3.0
DEFAULT_MIN_CLOSE_SECS = 90
DEFAULT_MAX_CLOSE_SECS = 600
DEFAULT_MAX_SPEND      = 500    # cents per trade ($5 max)
DEFAULT_POLL           = 20

_P_ABOVE = re.compile(r'\babove\s+\$?([\d,]+(?:\.\d+)?)', re.IGNORECASE)
_P_BELOW = re.compile(r'\bbelow\s+\$?([\d,]+(?:\.\d+)?)', re.IGNORECASE)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erf — no scipy required."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def prob_yes(
    live:      float,
    threshold: float,
    direction: str,    # "above" | "below"
    secs:      float,
    vol_hr:    float,
) -> float:
    """P(YES resolves) under zero-drift log-normal diffusion."""
    if secs <= 0:
        return (1.0 if live > threshold else 0.0) if direction == "above" else (1.0 if live < threshold else 0.0)

    sigma = vol_hr * math.sqrt(secs / 3600.0)
    if sigma == 0:
        return prob_yes(live, threshold, direction, 0, vol_hr)

    d = math.log(live / threshold) / sigma
    return _norm_cdf(d) if direction == "above" else _norm_cdf(-d)


# ── Price fetcher ─────────────────────────────────────────────────────────────

class _PriceFetcher:
    """Fetch live spot prices for BTC/ETH/SOL/DOGE.

    Binance public API (no auth, ~1200 req/min limit) is primary.
    CoinGecko free API (no auth, ~30 req/min) is fallback.
    Results are cached for `cache_secs` seconds.
    """

    _BINANCE   = "https://api.binance.com/api/v3/ticker/price"
    _COINGECKO = "https://api.coingecko.com/api/v3/simple/price"

    _SYMBOLS: dict[str, str] = {
        "BTC": "BTCUSDT", "ETH": "ETHUSDT",
        "SOL": "SOLUSDT", "DOGE": "DOGEUSDT",
    }
    _CG_IDS: dict[str, str] = {
        "BTC": "bitcoin", "ETH": "ethereum",
        "SOL": "solana",  "DOGE": "dogecoin",
    }

    def __init__(self, cache_secs: float = 8.0):
        self._http       = httpx.Client(timeout=5.0)
        self._cache:      dict[str, float] = {}
        self._cache_at:   float = 0.0
        self._cache_secs  = cache_secs
        self._lock        = threading.Lock()
        self._binance_failed_at: float = 0.0  # timestamp of last failure
        self._binance_backoff_secs: float = 60.0  # start at 1 min, exponential backoff

    def get(self) -> dict[str, float]:
        with self._lock:
            if self._cache and (time.monotonic() - self._cache_at) < self._cache_secs:
                return dict(self._cache)

        # Try Binance unless we're in backoff period
        prices: dict[str, float] = {}
        now_time = time.monotonic()
        if now_time - self._binance_failed_at >= self._binance_backoff_secs:
            prices = self._binance()
        
        # Fall back to CoinGecko if Binance failed or timed out
        if not prices:
            prices = self._coingecko()

        with self._lock:
            if prices:
                self._cache    = prices
                self._cache_at = time.monotonic()
            return dict(self._cache)

    def _binance(self) -> dict[str, float]:
        import json as _json
        try:
            syms = _json.dumps(list(self._SYMBOLS.values()))
            r    = self._http.get(self._BINANCE, params={"symbols": syms}, timeout=4.0)
            r.raise_for_status()
            inv  = {v: k for k, v in self._SYMBOLS.items()}
            out  = {inv[d["symbol"]]: float(d["price"]) for d in r.json() if d["symbol"] in inv}
            if out:
                # Success: reset backoff
                with self._lock:
                    self._binance_backoff_secs = 60.0
            return out
        except Exception as exc:
            logger.debug("Binance fetch failed: %s", exc)
            # Exponential backoff: 1m → 2m → 4m → 8m (capped at 30m)
            with self._lock:
                self._binance_failed_at = time.monotonic()
                self._binance_backoff_secs = min(1800.0, self._binance_backoff_secs * 2.0)
            return {}

    def _coingecko(self) -> dict[str, float]:
        try:
            r = self._http.get(
                self._COINGECKO,
                params={"ids": ",".join(self._CG_IDS.values()), "vs_currencies": "usd"},
                timeout=6.0,
            )
            r.raise_for_status()
            data = r.json()
            inv  = {v: k for k, v in self._CG_IDS.items()}
            return {inv[gid]: float(data[gid]["usd"]) for gid in data if gid in inv}
        except Exception as exc:
            logger.debug("CoinGecko fetch failed: %s", exc)
            return {}

    def close(self) -> None:
        self._http.close()


# ── Opportunity dataclass ─────────────────────────────────────────────────────

@dataclass
class ConvergenceOpp:
    market:      Market
    outcome:     str        # outcome name to buy
    side:        str        # "yes" | "no" — which array index the outcome is
    entry_price: int         # cents to pay per share
    prob_win:    float
    edge_pp:     float
    coin:        str
    live_price:  float
    threshold:   float
    direction:   str         # "above" | "below"
    count:       int


# ── Main bot ──────────────────────────────────────────────────────────────────

class ConvergenceBot:
    """Scans crypto threshold markets, bets when live price clearly resolves them."""

    def __init__(
        self,
        client:          PolymarketTradingClient,
        tag_slugs:       Optional[list[str]] = None,
        min_prob:        float = DEFAULT_MIN_PROB,
        max_entry_price: int   = DEFAULT_MAX_PRICE,
        min_edge_pp:     float = DEFAULT_MIN_EDGE_PP,
        min_close_secs:  int   = DEFAULT_MIN_CLOSE_SECS,
        max_close_secs:  int   = DEFAULT_MAX_CLOSE_SECS,
        max_spend_cents: int   = DEFAULT_MAX_SPEND,
        poll_interval:   int   = DEFAULT_POLL,
    ):
        self.client          = client
        self.tag_slugs       = tag_slugs or DEFAULT_TAG_SLUGS
        self.min_prob        = min_prob
        self.max_entry_price = max_entry_price
        self.min_edge_pp     = min_edge_pp
        self.min_close_secs  = min_close_secs
        self.max_close_secs  = max_close_secs
        self.max_spend_cents = max_spend_cents
        self.poll_interval   = poll_interval
        self._fetcher        = _PriceFetcher()
        self._dead:           set[str] = set()

    # ── Question parsing ──────────────────────────────────────────────────────

    def _coin(self, question: str) -> Optional[str]:
        for coin, pattern in _COIN_PATTERNS.items():
            if pattern.search(question):
                return coin
        return None

    def _parse_threshold(self, question: str) -> Optional[tuple[float, str]]:
        m = _P_ABOVE.search(question)
        if m:
            return float(m.group(1).replace(",", "")), "above"
        m = _P_BELOW.search(question)
        if m:
            return float(m.group(1).replace(",", "")), "below"
        return None

    # ── Signal generation ─────────────────────────────────────────────────────

    def _find_opps(
        self,
        markets: list[Market],
        held:    set[str],
        prices:  dict[str, float],
    ) -> list[ConvergenceOpp]:
        opps: list[ConvergenceOpp] = []
        now = datetime.now(timezone.utc)

        for market in markets:
            if market.condition_id in held or market.condition_id in self._dead:
                continue
            if not market.active or market.closed or market.archived:
                continue
            if len(market.outcomes) != 2 or not market.end_time:
                continue

            ct = market.end_time
            if ct.tzinfo is None:
                ct = ct.replace(tzinfo=timezone.utc)
            secs = (ct - now).total_seconds()
            if not (self.min_close_secs <= secs <= self.max_close_secs):
                continue

            coin = self._coin(market.question)
            if not coin or coin not in prices:
                continue

            parsed = self._parse_threshold(market.question)
            if parsed is None:
                continue

            threshold, direction = parsed
            live = prices[coin]

            if not (0.3 * live <= threshold <= 3.0 * live):
                logger.debug(
                    "[convergence] Skipping %s — threshold %.2f looks wrong vs live %.2f",
                    market.condition_id, threshold, live,
                )
                continue

            vol   = HOURLY_VOL.get(coin, 0.030)
            p_yes = prob_yes(live, threshold, direction, secs, vol)

            def _add(side: str, price: Optional[int], p_win: float) -> None:
                if not price or not (1 <= price <= self.max_entry_price):
                    return
                edge = p_win * 100.0 - price
                if p_win < self.min_prob or edge < self.min_edge_pp:
                    return
                count = self.max_spend_cents // price
                if count <= 0:
                    return
                outcome = market.outcomes[0] if side == "yes" else market.outcomes[1]
                opps.append(ConvergenceOpp(
                    market=market, outcome=outcome, side=side,
                    entry_price=price, prob_win=p_win, edge_pp=edge,
                    coin=coin, live_price=live,
                    threshold=threshold, direction=direction,
                    count=count,
                ))

            _add("yes", market.yes_ask, p_yes)
            _add("no",  market.no_ask,  1.0 - p_yes)

        return sorted(opps, key=lambda o: o.edge_pp, reverse=True)

    # ── Main scan ─────────────────────────────────────────────────────────────

    def _scan_once(self) -> None:
        prices = self._fetcher.get()
        if not prices:
            logger.warning("[convergence] No live prices available — skipping scan")
            return

        markets = self.client.list_liquid_markets(tag_slugs=self.tag_slugs)
        if not markets:
            return

        positions   = self.client.get_positions()
        open_orders = self.client.get_open_orders()
        held = {p.condition_id for p in positions} | {o.condition_id for o in open_orders}

        opps = self._find_opps(markets, held | self._dead, prices)

        if not opps:
            price_str = "  ".join(f"{c}=${p:,.2f}" for c, p in sorted(prices.items()))
            logger.debug("[convergence] No signals | %d markets | %s", len(markets), price_str)
            return

        logger.info(
            "[convergence] %d signal(s) — %s",
            len(opps),
            "  ".join(f"{c}=${p:,.2f}" for c, p in sorted(prices.items())),
        )

        balance   = self.client.get_balance()
        remaining = balance

        for opp in opps:
            cost = opp.entry_price * opp.count
            if cost > remaining:
                logger.debug("[convergence] Skipping %s — cost %d¢ > remaining %d¢",
                             opp.market.condition_id, cost, remaining)
                continue

            close_secs = (
                opp.market.end_time - datetime.now(timezone.utc)
            ).total_seconds()
            direction_str = f"{'>' if opp.direction == 'above' else '<'} ${opp.threshold:,.2f}"

            try:
                order = self.client.place_order(
                    condition_id=opp.market.condition_id,
                    outcome=opp.outcome,
                    size=opp.count,
                    price=opp.entry_price,
                    side="BUY",
                )
                logger.info(
                    "[convergence] BET %s on %s: %sx@%d¢  "
                    "prob=%.1f%%  edge=+%.1fpp  "
                    "%s=$%.2f %s  closes in %.0fs  order=%s",
                    opp.outcome, opp.market.condition_id,
                    opp.count, opp.entry_price,
                    opp.prob_win * 100.0, opp.edge_pp,
                    opp.coin, opp.live_price, direction_str,
                    close_secs, order.order_id,
                )
                held.add(opp.market.condition_id)
                remaining -= cost
            except Exception as exc:
                msg = str(exc)
                if "not found" in msg.lower() or "404" in msg:
                    self._dead.add(opp.market.condition_id)
                elif "insufficient" in msg.lower() or "balance" in msg.lower():
                    logger.debug("[convergence] Insufficient balance, stopping")
                    break
                else:
                    logger.warning("[convergence] Order failed on %s: %s",
                                   opp.market.condition_id, exc)

    # ── Thread entry point ────────────────────────────────────────────────────

    def run(self, stop_event: threading.Event) -> None:
        logger.info(
            "[convergence] Started — min_prob=%.0f%%  edge>=%.1fpp  "
            "window=%d-%ds  max_spend=%d¢  poll=%ds",
            self.min_prob * 100, self.min_edge_pp,
            self.min_close_secs, self.max_close_secs,
            self.max_spend_cents, self.poll_interval,
        )
        while not stop_event.is_set():
            try:
                self._scan_once()
            except Exception as exc:
                logger.error("[convergence] Scan error: %s", exc, exc_info=True)
            stop_event.wait(timeout=self.poll_interval)
        self._fetcher.close()
        logger.info("[convergence] Stopped.")

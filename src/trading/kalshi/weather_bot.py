"""A NOAA-forecast-grounded trading bot for Kalshi's daily-high-temperature
markets (KXHIGH*/KXHIGHT* series).

Structurally closer to polymarket_us/apex/bot.py than to this platform's own
TradingBot/MispricingStrategy pair -- TradingBot is tightly coupled to
win-rate-table strategies (it checks strategy.win_rates directly, calls
strategy.find_locks() for arbitrage, neither of which applies here), whereas
this bot's probability estimate comes from an independent NOAA forecast the
same way Apex's does. Runs as its own process (own launcher, own balance
carve-out, own NOAA-reload cadence) rather than a thread inside run_all.py,
for the same reasons Apex is its own process.

Execution paranoia reused from TradingBot, not reinvented: GTC maker orders
posted at the bid/ask midpoint (capturing spread instead of paying it,
~32% taker loss vs ~10% maker per this account's own research), stale
orders cancelled once they're no longer the best candidate, and a bet is
only ever recorded in PerformanceTracker once a resting order is confirmed
actually filled against a real position on a later scan -- never at
placement time. Same reasoning as TradingBot's _PendingFill: an order
response admitting nothing and a maker order that never fills at all are
both real, observed failure modes on this exchange.

Portfolio construction: at most one open position per (series, date) at a
time -- multiple temperature buckets for the same city-day are draws from
one forecast, exactly the same correlation Apex guards against per
(city, date) and the sports-props bot guards against per (sport, matchup,
date).
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from src.common.event_exposure import EventExposureTracker
from src.common.reservation import TickerReservation
from src.trading.kalshi.learning import PerformanceTracker
from src.trading.kalshi.trading_client import KalshiTradingClient, Order, Position
from src.trading.kalshi.weather_model import (
    CITY_STATIONS,
    KalshiWeatherGroundingEngine,
    parse_bucket_title,
    series_of,
)

logger = logging.getLogger(__name__)

# Same fee-free-fill floor Kalshi's own strategy.py uses -- a technically
# positive but tiny edge isn't worth a real order's footprint.
MIN_EDGE_PP = 2.0
MAX_SPEND_CENTS = 100  # up to $1/trade, matching the fleet's existing tuning


@dataclass
class _PendingFill:
    """Mirrors TradingBot._PendingFill exactly -- see that class's
    docstring for why a bet is never recorded until a resting order is
    confirmed filled against a real position, not just accepted."""
    ticker: str
    series: str
    side: str
    entry_price: int
    predicted_win_rate: float
    count: int
    close_time: Optional[datetime]

    def to_json_dict(self) -> dict:
        d = asdict(self)
        d["close_time"] = self.close_time.isoformat() if self.close_time else None
        return d

    @classmethod
    def from_json_dict(cls, d: dict) -> "_PendingFill":
        close_time = d.get("close_time")
        return cls(
            ticker=d["ticker"], series=d["series"], side=d["side"],
            entry_price=d["entry_price"], predicted_win_rate=d["predicted_win_rate"],
            count=d["count"], close_time=datetime.fromisoformat(close_time) if close_time else None,
        )


def _bucket_key(ticker: str) -> Optional[tuple]:
    """(series, date) if `ticker` belongs to a known weather series and its
    title-derived date can be recovered from the ticker's own ...-{YYMMMDD}-
    segment, else None. Doesn't need the title -- the date is already
    encoded in the ticker itself (e.g. KXHIGHTATL-26AUG15-B93.5)."""
    series = series_of(ticker)
    if series not in CITY_STATIONS:
        return None
    parts = ticker.split("-")
    if len(parts) < 2:
        return None
    return series, parts[1]  # e.g. "26AUG15" -- opaque but stable per city-day


class KalshiWeatherBot:
    def __init__(
        self,
        client: KalshiTradingClient,
        grounding: Optional[KalshiWeatherGroundingEngine] = None,
        tracker: Optional[PerformanceTracker] = None,
        poll_interval: int = 60,
        max_open_positions: int = 10,
        balance_fraction: float = 0.10,
        reservation: Optional[TickerReservation] = None,
        event_tracker: Optional[EventExposureTracker] = None,
        pending_fills_path: Optional[Path] = Path("data/kalshi/weather_pending_fills.json"),
    ):
        self.client = client
        self.grounding = grounding or KalshiWeatherGroundingEngine()
        self.tracker = tracker or PerformanceTracker(path=Path("data/kalshi/weather_bot_learning.json"))
        self.poll_interval = poll_interval
        self.max_open_positions = max_open_positions
        self.balance_fraction = balance_fraction
        self.reservation = reservation
        self.event_tracker = event_tracker
        self._pending_fills: dict[str, _PendingFill] = {}
        self._pending_fills_path = pending_fills_path
        if self._pending_fills_path:
            self._load_pending_fills()

    def close(self) -> None:
        self.grounding.close()

    # ── Pending-fill persistence (same shape as TradingBot's) ────────────────

    def _load_pending_fills(self):
        path = self._pending_fills_path
        if not path or not path.exists():
            return
        try:
            raw = json.loads(path.read_text())
            self._pending_fills = {oid: _PendingFill.from_json_dict(d) for oid, d in raw.items()}
            if self._pending_fills:
                logger.info(
                    "[kalshi-weather] Loaded %d pending fill(s) from a prior run", len(self._pending_fills),
                )
        except Exception as exc:
            logger.warning("[kalshi-weather] Failed to load pending fills from %s: %s", path, exc)

    def _save_pending_fills(self):
        path = self._pending_fills_path
        if not path:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({oid: pf.to_json_dict() for oid, pf in self._pending_fills.items()}, indent=2))
            os.replace(tmp, path)
        except Exception as exc:
            logger.warning("[kalshi-weather] Failed to save pending fills to %s: %s", path, exc)

    def _reconcile_pending_fills(self, positions: list[Position], open_orders: list[Order]) -> None:
        if not self._pending_fills:
            return
        still_open_ids = {o.order_id for o in open_orders}
        net_by_ticker = {p.ticker: p.position for p in positions}
        resolved_any = False

        for order_id, pending in list(self._pending_fills.items()):
            if order_id in still_open_ids:
                continue
            del self._pending_fills[order_id]
            resolved_any = True
            net = net_by_ticker.get(pending.ticker, 0)
            filled = (pending.side == "yes" and net > 0) or (pending.side == "no" and net < 0)
            if not filled:
                logger.info(
                    "[kalshi-weather] Order %s on %s never filled — no bet recorded",
                    order_id, pending.ticker,
                )
                continue
            actual_count = min(pending.count, abs(net))
            logger.info(
                "[kalshi-weather] Confirmed fill for %s on %s: %dx (of %d requested)",
                order_id, pending.ticker, actual_count, pending.count,
            )
            try:
                self.tracker.record_bet(
                    ticker=pending.ticker, series=pending.series, side=pending.side,
                    entry_price=pending.entry_price, predicted_win_rate=pending.predicted_win_rate,
                    count=actual_count, close_time=pending.close_time,
                )
            except Exception as exc:
                logger.error(
                    "[kalshi-weather] CRITICAL: confirmed fill for %s but record_bet failed: %s",
                    order_id, exc,
                )
        if resolved_any:
            self._save_pending_fills()

    # ── Candidate selection ───────────────────────────────────────────────────

    def _best_candidate_for_bucket_group(self, tickers_and_markets: list) -> Optional[tuple]:
        """Highest-edge tradeable side across every ticker in one (series,
        date) group, or None. tickers_and_markets: list of (ticker, title,
        market)."""
        best = None  # (edge, side, ticker, market, grounding, close_time)
        for ticker, title, market in tickers_and_markets:
            parsed = parse_bucket_title(title)
            if parsed is None:
                continue
            low, high, date_str = parsed
            series = series_of(ticker)
            if series not in CITY_STATIONS:
                continue
            grounding = self.grounding.evaluate_bucket(series, date_str, low, high)
            if grounding is None:
                continue

            if grounding.calibration_trustworthy and market.yes_ask is not None and market.yes_bid is not None:
                mid = max(1, min(99, (market.yes_bid + market.yes_ask) // 2))
                edge = grounding.model_prob * 100.0 - mid
                if edge >= MIN_EDGE_PP and (best is None or edge > best[0]):
                    best = (edge, "yes", ticker, market, grounding, market.close_time)
            if grounding.calibration_trustworthy_no and market.no_ask is not None and market.no_bid is not None:
                no_mid = max(1, min(99, (market.no_bid + market.no_ask) // 2))
                edge = (1.0 - grounding.model_prob) * 100.0 - no_mid
                if edge >= MIN_EDGE_PP and (best is None or edge > best[0]):
                    best = (edge, "no", ticker, market, grounding, market.close_time)
        return best

    # ── Scan loop ────────────────────────────────────────────────────────────

    def _scan_once(self) -> None:
        # Reload before checking staleness -- same ordering lesson Apex's
        # own calibration gate already learned (see its module history).
        self.grounding.reload_calibration()
        if self.grounding.history_is_stale():
            logger.warning(
                "[kalshi-weather] Calibration history is stale or missing — is "
                "scripts/backfill_kalshi_weather_history.py --loop running? Observing only."
            )
            return

        balance_cents = self.client.get_balance()
        budget_cents = int(balance_cents * self.balance_fraction)

        positions = self.client.get_positions()
        open_orders = self.client.get_open_orders()
        self._reconcile_pending_fills(positions, open_orders)

        prefixes = tuple(CITY_STATIONS)
        own_positions = [p for p in positions if p.ticker.startswith(prefixes)]
        own_open_orders = [o for o in open_orders if o.ticker.startswith(prefixes)]
        held_tickers = {p.ticker for p in own_positions} | {o.ticker for o in own_open_orders}
        slots = max(0, self.max_open_positions - len(own_positions) - len(own_open_orders))

        try:
            markets = self.client.list_liquid_open_markets(series_list=list(prefixes))
        except Exception as exc:
            logger.warning("[kalshi-weather] market scan failed: %s", exc)
            return

        groups: dict[tuple, list] = {}
        for m in markets:
            key = _bucket_key(m.ticker)
            if key is None:
                continue
            groups.setdefault(key, []).append((m.ticker, m.title, m))

        held_groups = {_bucket_key(t) for t in held_tickers} - {None}

        logger.info(
            "[%s] balance=$%.2f (budget=$%.2f)  groups=%d  positions=%d  slots=%d",
            datetime.now().strftime("%H:%M:%S"), balance_cents / 100, budget_cents / 100,
            len(groups), len(own_positions), slots,
        )

        if slots <= 0 or budget_cents <= 0:
            return

        remaining = budget_cents
        for key, tickers_and_markets in groups.items():
            if key in held_groups:
                continue
            candidate = self._best_candidate_for_bucket_group(tickers_and_markets)
            if candidate is None:
                continue
            edge, side, ticker, market, grounding, close_time = candidate

            if side == "yes":
                mid = max(1, min(99, (market.yes_bid + market.yes_ask) // 2))
                yes_price = mid
            else:
                no_mid = max(1, min(99, (100 - market.yes_ask + market.no_ask) // 2))
                mid = no_mid
                yes_price = 100 - no_mid

            count = max(1, MAX_SPEND_CENTS // max(1, mid))
            cost = mid * count
            if cost > remaining:
                continue

            if self.reservation and not self.reservation.try_reserve(ticker):
                continue
            ekey = ticker.rsplit("-", 1)[0]
            if self.event_tracker and not self.event_tracker.try_reserve(ekey, cost):
                if self.reservation:
                    self.reservation.release(ticker)
                continue

            try:
                order = self.client.place_order(ticker=ticker, side=side, count=count, yes_price=yes_price)
            except Exception as exc:
                if self.reservation:
                    self.reservation.release(ticker)
                if self.event_tracker:
                    self.event_tracker.release(ekey, cost)
                logger.warning("[kalshi-weather] order failed on %s: %s", ticker, exc)
                continue

            remaining -= cost
            slots -= 1
            self._pending_fills[order.order_id] = _PendingFill(
                ticker=ticker, series=series_of(ticker), side=side, entry_price=mid,
                predicted_win_rate=grounding.model_prob * 100.0 if side == "yes" else (1.0 - grounding.model_prob) * 100.0,
                count=count, close_time=close_time,
            )
            self._save_pending_fills()
            logger.info(
                "[kalshi-weather] Placed %s on %s: %dx @ %d¢ (mid)  edge=%.1fpp  model=%.1f%%  %s  id=%s  status=%s",
                side.upper(), ticker, count, mid, edge, grounding.model_prob * 100.0 if side == "yes" else (1.0 - grounding.model_prob) * 100.0,
                grounding.calibration_note if side == "yes" else grounding.calibration_note_no,
                order.order_id, order.status,
            )
            if slots <= 0:
                break

        # Cancel resting orders on buckets that are no longer the best
        # candidate in their group (forecast moved, calibration flipped).
        if markets:
            live_pending_tickers = {pf.ticker for pf in self._pending_fills.values()}
            for o in own_open_orders:
                if o.ticker not in live_pending_tickers:
                    logger.info("[kalshi-weather] Cancelling stale order %s on %s", o.order_id, o.ticker)
                    self.client.cancel_order(o.order_id)

    def run_forever(self, stop_event=None) -> None:
        while True:
            try:
                self._scan_once()
            except Exception as exc:
                logger.warning("[kalshi-weather] scan error: %s", exc)
            if stop_event is not None:
                if stop_event.wait(timeout=self.poll_interval):
                    break
            else:
                time.sleep(self.poll_interval)
        self.close()

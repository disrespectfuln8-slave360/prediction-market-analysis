"""Paper-trading (dry-run) wrapper for PolymarketUSTradingClient.

Same design as src/trading/kalshi/paper_client.py: market-data calls pass
straight through to a real, authenticated PolymarketUSTradingClient
(read-only, safe); money-moving calls are simulated against a local JSON
ledger. Orders are simulated as filled immediately at the requested price
— an approximation of a maker fill that doesn't model order-book depth or
the chance a resting order never gets hit.

Positions are tracked per (market_slug, side) since Polymarket US models
each market as a single long/short instrument, same as Kalshi's per-ticker
yes/no model.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from src.trading.polymarket_us.models import Market, Order, Position, _amount_to_cents
from src.trading.polymarket_us.strategy import taker_fee_cents
from src.trading.polymarket_us.trading_client import PolymarketUSTradingClient

logger = logging.getLogger(__name__)

DEFAULT_LEDGER_PATH = Path("data/polymarket_us/paper_ledger.json")
DEFAULT_STARTING_BALANCE_CENTS = 100_000  # $1,000 paper money
RECONCILE_INTERVAL = 20.0  # seconds between settlement checks of open paper positions
STUCK_LOG_INTERVAL = 600.0  # seconds between INFO logs about positions stuck awaiting settlement


def _settlement_to_cents(payload: Optional[dict]) -> Optional[int]:
    """Extract the settlement price in cents from a /settlement response.

    Observed live shape: {"slug": ..., "settlement": 1} — a bare number in
    dollars (1 = YES, 0 = NO), NOT an Amount dict and NOT "settlementPrice"
    (reading that field made every settlement check silently skip, so paper
    positions never closed and the bot deadlocked at its open-legs cap).
    Accepts both a bare number and an Amount dict under either key, so a
    future API change back to the SDK's documented shape keeps working.
    """
    if not payload:
        return None
    for key in ("settlement", "settlementPrice"):
        if key in payload:
            val = payload[key]
            if isinstance(val, dict):
                return _amount_to_cents(val)
            try:
                return round(float(val) * 100)
            except (TypeError, ValueError):
                return None
    return None


class PaperPolymarketUSTradingClient:
    """Drop-in replacement for PolymarketUSTradingClient that never places real orders."""

    def __init__(
        self,
        real_client: Optional[PolymarketUSTradingClient] = None,
        ledger_path: Path = DEFAULT_LEDGER_PATH,
        starting_balance_cents: int = DEFAULT_STARTING_BALANCE_CENTS,
    ):
        self._real = real_client or PolymarketUSTradingClient()
        self._owns_real = real_client is None
        self.ledger_path = ledger_path
        self._lock = threading.Lock()
        self._last_reconcile = 0.0
        self._last_stuck_log = 0.0
        self._cash_cents = starting_balance_cents
        # market_slug -> {"yes": count, "no": count}
        self._positions: dict[str, dict[str, int]] = {}
        self._trade_log: list[dict] = []
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self):
        if self.ledger_path.exists():
            try:
                raw = json.loads(self.ledger_path.read_text())
                self._cash_cents = raw.get("cash_cents", self._cash_cents)
                self._positions = raw.get("positions", {})
                self._trade_log = raw.get("trade_log", [])
                logger.info(
                    "[paper] Loaded ledger: cash=$%.2f  open_markets=%d",
                    self._cash_cents / 100, len(self._positions),
                )
            except Exception as exc:
                logger.warning("[paper] Failed to load ledger, starting fresh: %s", exc)

    def _save_locked(self):
        """Caller must hold self._lock."""
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "cash_cents": self._cash_cents,
            "positions": self._positions,
            "trade_log": self._trade_log[-500:],
        }, indent=2))
        os.replace(tmp, self.ledger_path)

    # ── Context manager / lifecycle ──────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self._owns_real:
            self._real.close()

    # ── Market data (pass-through — read-only, safe) ─────────────────────────

    def get_market(self, slug: str) -> Market:
        return self._real.get_market(slug)

    def list_markets(self, active: bool = True, closed: bool = False, limit: int = 200, **filters):
        return self._real.list_markets(active=active, closed=closed, limit=limit, **filters)

    def list_all_markets(self, active: bool = True, closed: bool = False, page_size: int = 200, **filters):
        return self._real.list_all_markets(active=active, closed=closed, page_size=page_size, **filters)

    def get_bbo(self, slug: str) -> dict:
        return self._real.get_bbo(slug)

    # ── Portfolio (simulated) ────────────────────────────────────────────────

    def get_balance(self) -> int:
        self._reconcile()
        with self._lock:
            return self._cash_cents

    def get_positions(self) -> list[Position]:
        self._reconcile()
        out = []
        with self._lock:
            for slug, legs in self._positions.items():
                net = legs.get("yes", 0) - legs.get("no", 0)
                if net != 0:
                    out.append(Position(market_slug=slug, position=net))
        return out

    def get_open_orders(self) -> list[Order]:
        # Paper fills are immediate — nothing ever rests.
        return []

    def place_order(
        self,
        market_slug: str,
        side: str,
        count: int,
        yes_price: int,
        immediate_or_cancel: bool = False,
    ) -> Order:
        # immediate_or_cancel is accepted to keep this a drop-in for the real
        # client. It changes nothing here: paper fills are already immediate
        # and total, so IOC and GTC are indistinguishable in simulation. That
        # is precisely why the live-only fill-reporting bug could not have
        # been caught in paper mode.
        price_paid = yes_price if side == "yes" else (100 - yes_price)
        price_paid = max(1, min(99, price_paid))
        # Real fills pay commission (~7% * p * (1-p) per contract, measured on
        # this account). A fee-free paper ledger books every thin-edge lock as
        # a winner that live trading would turn into a loser.
        fee = taker_fee_cents(price_paid, count)
        cost = price_paid * count + fee

        with self._lock:
            if cost > self._cash_cents:
                raise RuntimeError(f"insufficient_balance: need {cost}c, have {self._cash_cents}c")
            self._cash_cents -= cost
            leg = self._positions.setdefault(market_slug, {"yes": 0, "no": 0})
            leg[side] = leg.get(side, 0) + count
            order_id = f"paper-{uuid.uuid4().hex[:12]}"
            self._trade_log.append({
                "order_id": order_id, "market_slug": market_slug, "side": side,
                "count": count, "price_cents": price_paid, "ts": time.time(),
            })
            cash_now = self._cash_cents
            self._save_locked()

        logger.info(
            "[paper] FILLED %s %dx %s @ %d¢ (cost=%d¢, cash=$%.2f)",
            side.upper(), count, market_slug, price_paid, cost, cash_now / 100,
        )
        return Order(
            order_id=order_id, market_slug=market_slug, side=side,
            count=count, price_cents=yes_price, status="filled",
        )

    def close_long(self, market_slug: str, count: int, price_cents: int) -> tuple:
        """Sell `count` of a simulated long. Returns (order_id, sold)."""
        with self._lock:
            leg = self._positions.get(market_slug)
            held = leg.get("yes", 0) if leg else 0
            sold = min(count, held)
            if sold <= 0:
                return f"paper-{uuid.uuid4().hex[:12]}", 0
            leg["yes"] = held - sold
            if leg.get("yes", 0) == 0 and leg.get("no", 0) == 0:
                self._positions.pop(market_slug, None)
            proceeds = sold * price_cents - taker_fee_cents(price_cents, sold)
            self._cash_cents += proceeds
            order_id = f"paper-{uuid.uuid4().hex[:12]}"
            self._trade_log.append({
                "order_id": order_id, "market_slug": market_slug, "side": "sell_long",
                "count": sold, "price_cents": price_cents, "ts": time.time(),
            })
            cash_now = self._cash_cents
            self._save_locked()

        logger.info(
            "[paper] SOLD %dx %s @ %d¢ (proceeds=%d¢, cash=$%.2f)",
            sold, market_slug, price_cents, proceeds, cash_now / 100,
        )
        return order_id, sold

    def cancel_order(self, order_id: str, market_slug: str = "") -> bool:
        # Nothing rests in paper mode, so there's never anything to cancel.
        return True

    # ── Settlement reconciliation ─────────────────────────────────────────────

    def _reconcile(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_reconcile) < RECONCILE_INTERVAL:
            return
        self._last_reconcile = now

        with self._lock:
            slugs = list(self._positions.keys())
        if not slugs:
            return

        stuck = 0  # closed markets we could not settle this pass
        for slug in slugs:
            try:
                market = self._real.get_market(slug)
            except Exception as exc:
                logger.debug("[paper] settlement check failed for %s: %s", slug, exc)
                continue
            if not market.closed:
                continue
            try:
                settlement = self._real.get_settlement(slug)
            except Exception as exc:
                stuck += 1
                logger.debug("[paper] no settlement yet for %s: %s", slug, exc)
                continue
            settlement_cents = _settlement_to_cents(settlement)
            if settlement_cents is None:
                stuck += 1
                logger.warning(
                    "[paper] %s is closed but its settlement response has no usable "
                    "price (got %r) — position stays open, cannot free its slot",
                    slug, settlement,
                )
                continue
            result = "yes" if settlement_cents >= 50 else "no"

            with self._lock:
                legs = self._positions.pop(slug, None)
                if legs is None:
                    continue
                payout = legs.get(result, 0) * 100
                self._cash_cents += payout
                cash_now = self._cash_cents
                self._save_locked()

            logger.info(
                "[paper] SETTLED %s -> %s  payout=+%d¢  (held yes=%d no=%d)  cash=$%.2f",
                slug, result.upper(), payout,
                legs.get("yes", 0), legs.get("no", 0), cash_now / 100,
            )

        if stuck and (now - self._last_stuck_log) >= STUCK_LOG_INTERVAL:
            self._last_stuck_log = now
            logger.info(
                "[paper] %d closed position(s) awaiting settlement data — "
                "slots stay occupied until the exchange publishes settlement",
                stuck,
            )

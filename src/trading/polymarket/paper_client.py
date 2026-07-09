"""Paper-trading (dry-run) wrapper for PolymarketTradingClient.

Same design as src/trading/kalshi/paper_client.py: market-data calls pass
straight through to a real, authenticated PolymarketTradingClient
(read-only, safe); money-moving calls are simulated against a local JSON
ledger. Orders are simulated as filled immediately at the requested price
— an approximation of a maker fill that doesn't model order-book depth or
the chance a resting order never gets hit.

Positions are tracked per (condition_id, outcome) — Polymarket outcome
strings aren't reliably binary "yes"/"no" tickers like Kalshi, so we key
directly on whatever outcome string the bot bought and match it against
the resolution API's token outcome names (case-insensitive) at settlement.
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

from src.trading.polymarket.models import Market, Order, Position
from src.trading.polymarket.trading_client import PolymarketTradingClient

logger = logging.getLogger(__name__)

DEFAULT_LEDGER_PATH = Path("data/polymarket/paper_ledger.json")
DEFAULT_STARTING_BALANCE_CENTS = 100_000  # $1,000 paper money
RECONCILE_INTERVAL = 20.0  # seconds between settlement checks of open paper positions


class PaperPolymarketTradingClient:
    """Drop-in replacement for PolymarketTradingClient that never places real orders."""

    def __init__(
        self,
        real_client: Optional[PolymarketTradingClient] = None,
        ledger_path: Path = DEFAULT_LEDGER_PATH,
        starting_balance_cents: int = DEFAULT_STARTING_BALANCE_CENTS,
    ):
        self._real = real_client or PolymarketTradingClient()
        self._owns_real = real_client is None
        self.address = getattr(self._real, "address", "")
        self.ledger_path = ledger_path
        self._lock = threading.Lock()
        self._last_reconcile = 0.0
        self._cash_cents = starting_balance_cents
        # condition_id -> {outcome_str: {"shares": float, "cost_cents": int}}
        self._positions: dict[str, dict[str, dict]] = {}
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

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self._owns_real:
            self._real.close()

    # ── Market data (pass-through — read-only, safe) ─────────────────────────

    def get_markets(self, next_cursor: Optional[str] = None) -> list[Market]:
        return self._real.get_markets(next_cursor=next_cursor)

    def list_liquid_markets(self, tag_slugs: Optional[list[str]] = None) -> list[Market]:
        return self._real.list_liquid_markets(tag_slugs=tag_slugs)

    def get_market_resolution(self, condition_id: str) -> Optional[dict]:
        return self._real.get_market_resolution(condition_id)

    # ── Portfolio (simulated) ────────────────────────────────────────────────

    def get_balance(self) -> int:
        self._reconcile()
        with self._lock:
            return self._cash_cents

    def get_positions(self) -> list[Position]:
        self._reconcile()
        out = []
        with self._lock:
            for condition_id, legs in self._positions.items():
                for outcome, leg in legs.items():
                    shares = leg.get("shares", 0)
                    if shares <= 0:
                        continue
                    avg_price = (leg.get("cost_cents", 0) / shares / 100) if shares else 0.0
                    out.append(Position(
                        condition_id=condition_id, outcome=outcome,
                        size=shares, avg_price=avg_price,
                    ))
        return out

    def get_open_orders(self) -> list[Order]:
        # Paper fills are immediate — nothing ever rests.
        return []

    def place_order(
        self,
        condition_id: str,
        outcome: str,
        size: float,
        price: int,
        side: str = "BUY",
    ) -> Order:
        if side.upper() != "BUY":
            raise NotImplementedError("paper client only simulates BUY orders")

        price = max(1, min(99, price))
        cost = round(price * size)

        with self._lock:
            if cost > self._cash_cents:
                raise RuntimeError(f"insufficient_balance: need {cost}c, have {self._cash_cents}c")
            self._cash_cents -= cost
            market_legs = self._positions.setdefault(condition_id, {})
            leg = market_legs.setdefault(outcome, {"shares": 0.0, "cost_cents": 0})
            leg["shares"] += size
            leg["cost_cents"] += cost
            order_id = f"paper-{uuid.uuid4().hex[:12]}"
            self._trade_log.append({
                "order_id": order_id, "condition_id": condition_id, "outcome": outcome,
                "size": size, "price_cents": price, "ts": time.time(),
            })
            cash_now = self._cash_cents
            self._save_locked()

        logger.info(
            "[paper] FILLED BUY %s on %s: %sx @ %d¢ (cost=%d¢, cash=$%.2f)",
            outcome, condition_id, size, price, cost, cash_now / 100,
        )
        return Order(
            order_id=order_id, condition_id=condition_id, outcome=outcome,
            size=size, price=price, status="filled",
        )

    def cancel_order(self, order_id: str) -> bool:
        # Nothing rests in paper mode, so there's never anything to cancel.
        return True

    # ── Settlement reconciliation ─────────────────────────────────────────────

    def _reconcile(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_reconcile) < RECONCILE_INTERVAL:
            return
        self._last_reconcile = now

        with self._lock:
            condition_ids = list(self._positions.keys())
        if not condition_ids:
            return

        for condition_id in condition_ids:
            try:
                data = self._real.get_market_resolution(condition_id)
            except Exception as exc:
                logger.debug("[paper] settlement check failed for %s: %s", condition_id, exc)
                continue
            if not data or not data.get("closed"):
                continue
            tokens = data.get("tokens", []) or []
            if not any(t.get("winner") for t in tokens):
                continue  # closed but not yet resolved

            with self._lock:
                legs = self._positions.pop(condition_id, None)
                if legs is None:
                    continue
                payout = 0
                for outcome, leg in legs.items():
                    match = next(
                        (t for t in tokens if t.get("outcome", "").lower() == outcome.lower()),
                        None,
                    )
                    if match and match.get("winner"):
                        payout += round(leg.get("shares", 0) * 100)
                self._cash_cents += payout
                cash_now = self._cash_cents
                self._save_locked()

            logger.info(
                "[paper] SETTLED %s -> payout=+%d¢  cash=$%.2f",
                condition_id, payout, cash_now / 100,
            )

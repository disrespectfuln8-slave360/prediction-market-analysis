"""Paper-trading (dry-run) wrapper for KalshiTradingClient.

Every market-data call passes straight through to a real, authenticated
KalshiTradingClient — safe, since those are all read-only. Every
money-moving call (get_balance, get_positions, get_open_orders,
place_order, cancel_order) is intercepted and simulated against a local
JSON ledger instead of touching the real Kalshi account.

Fill model: an order is simulated as filled immediately at the requested
price. The bots already post at the bid/ask midpoint, so this is a
reasonable approximation of a maker fill — it does not model partial
fills, order-book depth, or the chance a resting order never gets hit, so
paper P&L is optimistic relative to live trading. Settlement is detected
by polling the real market's resolution status for every open simulated
position, at most once every RECONCILE_INTERVAL seconds.
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

from src.indexers.kalshi.models import Market
from src.trading.kalshi.trading_client import KalshiTradingClient, Order, Position

logger = logging.getLogger(__name__)

DEFAULT_LEDGER_PATH = Path("data/kalshi/paper_ledger.json")
DEFAULT_STARTING_BALANCE_CENTS = 100_000  # $1,000 paper money
RECONCILE_INTERVAL = 20.0  # seconds between settlement checks of open paper positions


class PaperKalshiTradingClient:
    """Drop-in replacement for KalshiTradingClient that never places real orders."""

    def __init__(
        self,
        real_client: Optional[KalshiTradingClient] = None,
        ledger_path: Path = DEFAULT_LEDGER_PATH,
        starting_balance_cents: int = DEFAULT_STARTING_BALANCE_CENTS,
    ):
        self._real = real_client or KalshiTradingClient()
        self._owns_real = real_client is None
        self.ledger_path = ledger_path
        self._lock = threading.Lock()
        self._last_reconcile = 0.0
        self._cash_cents = starting_balance_cents
        # ticker -> {"yes": count, "no": count}
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
                    "[paper] Loaded ledger: cash=$%.2f  open_tickers=%d",
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
            "trade_log": self._trade_log[-500:],  # bound file growth
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

    def get_market(self, ticker: str) -> Market:
        return self._real.get_market(ticker)

    def list_markets(self, limit: int = 200, status: str = "open"):
        return self._real.list_markets(limit=limit, status=status)

    def list_liquid_open_markets(self, min_volume_24h: int = 0, series_list: Optional[list[str]] = None):
        return self._real.list_liquid_open_markets(min_volume_24h=min_volume_24h, series_list=series_list)

    # ── Portfolio (simulated) ────────────────────────────────────────────────

    def get_balance(self) -> int:
        self._reconcile()
        with self._lock:
            return self._cash_cents

    def get_positions(self) -> list[Position]:
        self._reconcile()
        out = []
        with self._lock:
            for ticker, legs in self._positions.items():
                net = legs.get("yes", 0) - legs.get("no", 0)
                if net != 0:
                    out.append(Position(ticker=ticker, position=net))
        return out

    def get_open_orders(self) -> list[Order]:
        # Paper fills are immediate — nothing ever rests.
        return []

    def place_order(self, ticker: str, side: str, count: int, yes_price: int) -> Order:
        price_paid = yes_price if side == "yes" else (100 - yes_price)
        price_paid = max(1, min(99, price_paid))
        cost = price_paid * count

        with self._lock:
            if cost > self._cash_cents:
                raise RuntimeError(f"insufficient_balance: need {cost}c, have {self._cash_cents}c")
            self._cash_cents -= cost
            leg = self._positions.setdefault(ticker, {"yes": 0, "no": 0})
            leg[side] = leg.get(side, 0) + count
            order_id = f"paper-{uuid.uuid4().hex[:12]}"
            self._trade_log.append({
                "order_id": order_id, "ticker": ticker, "side": side,
                "count": count, "price_cents": price_paid, "ts": time.time(),
            })
            cash_now = self._cash_cents
            self._save_locked()

        logger.info(
            "[paper] FILLED %s %dx %s @ %d¢ (cost=%d¢, cash=$%.2f)",
            side.upper(), count, ticker, price_paid, cost, cash_now / 100,
        )
        return Order(
            order_id=order_id, ticker=ticker, side=side, action="buy",
            count=count, yes_price=yes_price, status="filled",
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
            tickers = list(self._positions.keys())
        if not tickers:
            return

        for ticker in tickers:
            try:
                market = self._real.get_market(ticker)
            except Exception as exc:
                logger.debug("[paper] settlement check failed for %s: %s", ticker, exc)
                continue
            if market.status != "finalized" or market.result not in ("yes", "no"):
                continue

            with self._lock:
                legs = self._positions.pop(ticker, None)
                if legs is None:
                    continue
                payout = legs.get(market.result, 0) * 100
                self._cash_cents += payout
                cash_now = self._cash_cents
                self._save_locked()

            logger.info(
                "[paper] SETTLED %s -> %s  payout=+%d¢  (held yes=%d no=%d)  cash=$%.2f",
                ticker, market.result.upper(), payout,
                legs.get("yes", 0), legs.get("no", 0), cash_now / 100,
            )

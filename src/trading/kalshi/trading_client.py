from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from typing import Optional

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from src.common.client import HttpClient
from src.indexers.kalshi.client import KALSHI_API_HOST
from src.indexers.kalshi.models import Market


class _KalshiSigner:
    """Signs Kalshi API requests using RSA-PSS (SHA-256)."""

    def __init__(self, key_id: str, private_key_path: str):
        self.key_id = key_id
        with open(private_key_path, "rb") as f:
            self._key = serialization.load_pem_private_key(f.read(), password=None)

    def __call__(self, method: str, path: str) -> dict:
        ts = str(int(time.time() * 1000))
        # Kalshi signs the full path including the /trade-api/v2 prefix
        clean_path = "/trade-api/v2" + path.split("?")[0]
        msg = (ts + method.upper() + clean_path).encode()
        sig = self._key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }


@dataclass
class Order:
    order_id: str
    ticker: str
    side: str
    action: str
    count: int
    yes_price: int
    status: str

    @classmethod
    def from_dict(cls, data: dict) -> "Order":
        return cls(
            order_id=data.get("order_id", ""),
            ticker=data.get("ticker", ""),
            side=data.get("side", ""),
            action=data.get("action", "buy"),
            count=int(data.get("count", 0)),
            yes_price=int(data.get("yes_price", 0)),
            status=data.get("status", ""),
        )


@dataclass
class Position:
    ticker: str
    position: int  # positive = net YES, negative = net NO

    @classmethod
    def from_dict(cls, data: dict) -> "Position":
        return cls(
            ticker=data.get("ticker", ""),
            position=int(data.get("position", 0)),
        )


class KalshiTradingClient:
    """Authenticated Kalshi client for portfolio and order management.

    Requires two env vars (or explicit args):
      KALSHI_API_KEY          — your key ID (UUID from Kalshi dashboard)
      KALSHI_PRIVATE_KEY_PATH — path to your RSA private key PEM file
    """

    def __init__(
        self,
        key_id: Optional[str] = None,
        private_key_path: Optional[str] = None,
        host: str = KALSHI_API_HOST,
    ):
        key_id = key_id or os.environ["KALSHI_API_KEY"]
        private_key_path = private_key_path or os.environ["KALSHI_PRIVATE_KEY_PATH"]
        signer = _KalshiSigner(key_id, private_key_path)
        self.http = HttpClient(base_url=host, rate_limit=5, auth_hook=signer)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.http.close()

    def close(self):
        self.http.close()

    # ── Market data ─────────────────────────────────────────────────────────

    def get_market(self, ticker: str) -> Market:
        data = self.http.get(f"/markets/{ticker}")
        return Market.from_dict(data["market"])

    def list_markets(self, limit: int = 200, status: str = "open") -> list[Market]:
        params = {"limit": limit, "status": status}
        data = self.http.get("/markets", params=params)
        return [Market.from_dict(m) for m in data.get("markets", [])]

    def list_all_open_markets(self) -> list[Market]:
        """Paginate through all open markets."""
        all_markets: list[Market] = []
        cursor = None
        while True:
            params: dict = {"limit": 200, "status": "open"}
            if cursor:
                params["cursor"] = cursor
            data = self.http.get("/markets", params=params)
            all_markets.extend(Market.from_dict(m) for m in data.get("markets", []))
            cursor = data.get("cursor")
            if not cursor:
                break
        return all_markets

    # ── Portfolio ────────────────────────────────────────────────────────────

    def get_balance(self) -> int:
        """Available balance in cents."""
        data = self.http.get("/portfolio/balance")
        return int(data.get("balance", 0))

    def get_positions(self) -> list[Position]:
        data = self.http.get("/portfolio/positions")
        return [
            Position.from_dict(p)
            for p in data.get("market_positions", [])
            if p.get("position", 0) != 0
        ]

    def get_open_orders(self) -> list[Order]:
        data = self.http.get("/portfolio/orders", params={"status": "resting"})
        return [Order.from_dict(o) for o in data.get("orders", [])]

    # ── Orders ───────────────────────────────────────────────────────────────

    def place_order(
        self,
        ticker: str,
        side: str,
        count: int,
        yes_price: int,
        action: str = "buy",
        order_type: str = "limit",
    ) -> Order:
        """Place a limit order.

        Args:
            ticker: Market ticker symbol.
            side: "yes" or "no".
            count: Number of contracts.
            yes_price: YES-side price in cents (1-99). For NO orders pass
                       100 - no_price so Kalshi receives the YES-equivalent.
            action: "buy" or "sell".
            order_type: "limit" or "market".
        """
        payload = {
            "ticker": ticker,
            "action": action,
            "type": order_type,
            "side": side,
            "count": count,
            "yes_price": yes_price,
        }
        data = self.http.post("/portfolio/orders", json=payload)
        return Order.from_dict(data.get("order", data))

    def cancel_order(self, order_id: str) -> bool:
        try:
            self.http.delete(f"/portfolio/orders/{order_id}")
            return True
        except Exception:
            return False

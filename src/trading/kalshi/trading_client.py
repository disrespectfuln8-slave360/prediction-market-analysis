from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from src.common.client import HttpClient
from src.indexers.kalshi.client import KALSHI_API_HOST
from src.indexers.kalshi.models import Market

logger = logging.getLogger(__name__)


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
    def from_dict(cls, data: dict) -> Order:
        # V2 API returns "bid" (buy YES) or "ask" (buy NO) — normalize to "yes"/"no"
        # so _cancel_stale comparisons work against Opportunity.side.
        raw_side = data.get("side", "")
        side = "yes" if raw_side in ("bid", "yes") else "no"
        return cls(
            order_id=data.get("order_id", ""),
            ticker=data.get("ticker", ""),
            side=side,
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
    def from_dict(cls, data: dict) -> Position:
        raw = data.get("position_fp", data.get("position", 0))
        return cls(
            ticker=data.get("ticker", ""),
            position=int(float(raw)),
        )


class KalshiTradingClient:
    """Authenticated Kalshi client for portfolio and order management.

    Requires two env vars (or explicit args):
      KALSHI_API_KEY          — your key ID (UUID from Kalshi dashboard)
      KALSHI_PRIVATE_KEY_PATH — path to your RSA private key PEM file
    """

    def __init__(
        self,
        key_id: str | None = None,
        private_key_path: str | None = None,
        host: str = KALSHI_API_HOST,
    ):
        # Raise something actionable rather than a bare KeyError traceback.
        # This is the first thing a new operator hits, and it fires even in
        # paper mode (the paper client wraps a real one for read-only market
        # data), so "KeyError: 'KALSHI_API_KEY'" reads like a bug in the bot
        # rather than a missing local credential.
        key_id = key_id or os.environ.get("KALSHI_API_KEY")
        private_key_path = private_key_path or os.environ.get("KALSHI_PRIVATE_KEY_PATH")
        missing = [
            name
            for name, value in (
                ("KALSHI_API_KEY", key_id),
                ("KALSHI_PRIVATE_KEY_PATH", private_key_path),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                f"Missing Kalshi credential(s): {', '.join(missing)}. "
                "Copy .env.example to .env and fill them in (KALSHI_API_KEY is the "
                "key ID from your Kalshi dashboard; KALSHI_PRIVATE_KEY_PATH points at "
                "your RSA private key PEM). These are needed even in paper mode, "
                "because paper trading still reads real market data — it just never "
                "sends an order."
            )

        if not Path(private_key_path).exists():
            raise RuntimeError(
                f"KALSHI_PRIVATE_KEY_PATH points at '{private_key_path}', which does "
                "not exist. It must be the path to your RSA private key PEM file."
            )

        signer = _KalshiSigner(key_id, private_key_path)
        self.http = HttpClient(base_url=host, rate_limit=5, auth_hook=signer)
        # Order placement must never be silently retried by the shared retry
        # machinery: a retried POST after a timeout or 5xx that actually
        # succeeded server-side would place a second, real, duplicate order
        # with no way for the caller to detect it. A single attempt that
        # raises on failure is safer than a hidden duplicate order -- same
        # reasoning as src/trading/polymarket_us/trading_client.py's
        # order_http.
        self.order_http = HttpClient(base_url=host, rate_limit=5, auth_hook=signer, max_retries=1)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.http.close()
        self.order_http.close()

    # ── Market data ─────────────────────────────────────────────────────────

    def get_market(self, ticker: str) -> Market:
        data = self.http.get(f"/markets/{ticker}")
        return Market.from_dict(data["market"])

    def list_markets(self, limit: int = 200, status: str = "open") -> list[Market]:
        params = {"limit": limit, "status": status}
        data = self.http.get("/markets", params=params)
        return [Market.from_dict(m) for m in data.get("markets", [])]

    def list_liquid_open_markets(
        self,
        min_volume_24h: int = 0,
        series_list: list[str] | None = None,
    ) -> list[Market]:
        """Fetch only liquid open markets by querying high-volume series directly.

        Instead of paginating through all 50k+ markets (most illiquid esports),
        this queries specific series known to be active and filters by volume.
        Pass series_list to restrict to a specific subset (e.g. crypto-only).
        """
        series = series_list or [
            "KXBTC",
            "KXETH",
            "KXSOL",
            "KXDOGE",  # crypto prices (hourly)
            "KXBTC15M",
            "KXETH15M",
            "KXSOL15M",  # 15-min crypto brackets
            "KXBTCD",  # daily BTC brackets
            "KXINX",
            "KXNDX",
            "KXSPY",
            "KXDJI",  # stock indices
            "INX",
            "NDX",
            "NASDAQ",  # more indices
            "FED",
            "FOMC",
            "CPI",
            "JOBS",
            "GDP",
            "UNRATE",  # macro
            "PRES",
            "CONGRESS",
            "SENATE",
            "HOUSE",  # politics
        ]
        all_markets: list[Market] = []
        seen: set[str] = set()

        for prefix in series:
            cursor = None
            while True:
                params: dict = {"limit": 200, "status": "open", "series_ticker": prefix}
                if cursor:
                    params["cursor"] = cursor
                try:
                    data = self.http.get("/markets", params=params)
                except Exception:
                    break
                for m in data.get("markets", []):
                    market = Market.from_dict(m)
                    if market.ticker not in seen and market.volume_24h >= min_volume_24h:
                        all_markets.append(market)
                        seen.add(market.ticker)
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
            if float(p.get("position_fp", p.get("position", 0))) != 0
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
    ) -> Order:
        """Place a GTC limit order using the Kalshi V2 order API.

        Args:
            ticker: Market ticker symbol.
            side: "yes" (buy YES) or "no" (buy NO).
            count: Number of contracts.
            yes_price: YES-side price in cents (1-99). For NO orders pass
                       the YES-equivalent (100 - no_price).
        """
        # V2 API: side is "bid" (buy YES) or "ask" (sell YES = buy NO).
        # Price is a fixed-point dollar string like "0.14". count is a string.
        v2_side = "bid" if side == "yes" else "ask"
        payload = {
            "ticker": ticker,
            "side": v2_side,
            "count": str(count),
            "price": f"{yes_price / 100:.4f}",
            "time_in_force": "good_till_canceled",
            "self_trade_prevention_type": "taker_at_cross",
        }
        data = self.order_http.post("/portfolio/events/orders", json=payload)
        # V2 response is flat (no "order" wrapper): {order_id, fill_count, remaining_count, ...}
        remaining = float(data.get("remaining_count", count))
        return Order(
            order_id=data.get("order_id", ""),
            ticker=ticker,
            side=side,
            action="buy",
            count=count,
            yes_price=yes_price,
            status="resting" if remaining > 0 else "filled",
        )

    def cancel_order(self, order_id: str) -> bool:
        # V2 endpoint — the old "/portfolio/orders/{id}" path returns a
        # 410 "deprecated_v1_order_endpoint" error (confirmed against the
        # live API: it was silently swallowed here for the entirety of the
        # first live-trading session, so every "stale order" cancellation
        # was actually a no-op — the order stayed resting on the real book
        # instead of being pulled, exactly the exposure this method exists
        # to prevent). Matches the same /portfolio/events/orders/ family
        # already used correctly by place_order.
        try:
            self.http.delete(f"/portfolio/events/orders/{order_id}")
            return True
        except Exception as exc:
            body = getattr(getattr(exc, "response", None), "text", "")
            logger.warning("cancel_order failed for %s: %s  body=%s", order_id, exc, body[:200])
            return False

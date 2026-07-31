from __future__ import annotations

import base64
import logging
import os
import time
from typing import Optional

from nacl.signing import SigningKey

from src.common.client import HttpClient
from src.trading.polymarket_us.models import Market, Order, Position

logger = logging.getLogger(__name__)

GATEWAY_BASE_URL = "https://gateway.polymarket.us"  # public market data, unauthenticated
API_BASE_URL = "https://api.polymarket.us"  # orders/portfolio/account, requires signed auth


class _PolymarketUSSigner:
    """Signs Polymarket US API requests using Ed25519.

    Scheme confirmed by reading the official polymarket-us SDK's auth.py
    directly (that package requires Python>=3.10 so it can't be depended on
    here): message = f"{timestamp_ms}{METHOD}{path}" (no separators, path
    excludes query string), signed with the account's Ed25519 secret key.
    """

    def __init__(self, key_id: str, secret_key: str):
        self.key_id = key_id
        secret_bytes = base64.b64decode(secret_key)
        if len(secret_bytes) == 64:
            secret_bytes = secret_bytes[:32]  # some tools export seed+pubkey; we only need the seed
        self._signing_key = SigningKey(secret_bytes)

    def __call__(self, method: str, path: str) -> dict:
        clean_path = path.split("?")[0]
        timestamp = str(int(time.time() * 1000))
        message = f"{timestamp}{method.upper()}{clean_path}".encode()
        signed = self._signing_key.sign(message)
        return {
            "X-PM-Access-Key": self.key_id,
            "X-PM-Timestamp": timestamp,
            "X-PM-Signature": base64.b64encode(signed.signature).decode(),
        }


class PolymarketUSTradingClient:
    """Authenticated client for the Polymarket US (polymarket.us) regulated
    exchange API — a single long/short instrument per market, closer to
    Kalshi's model than the international Polymarket CLOB's dual-token model.

    Requires two env vars (or explicit args):
      POLYMARKET_US_KEY_ID     — API key ID (UUID) from the Developer API page
      POLYMARKET_US_SECRET_KEY — base64-encoded Ed25519 secret key
    """

    def __init__(
        self,
        key_id: Optional[str] = None,
        secret_key: Optional[str] = None,
        gateway_base_url: str = GATEWAY_BASE_URL,
        api_base_url: str = API_BASE_URL,
    ):
        key_id = key_id or os.environ["POLYMARKET_US_KEY_ID"]
        secret_key = secret_key or os.environ["POLYMARKET_US_SECRET_KEY"]
        signer = _PolymarketUSSigner(key_id, secret_key)
        # Market data lives on a separate unauthenticated gateway host; orders/
        # portfolio/account live on the authenticated host and need signing.
        self.market_http = HttpClient(base_url=gateway_base_url, rate_limit=5)
        self.http = HttpClient(base_url=api_base_url, rate_limit=5, auth_hook=signer)
        # Order creation must never be silently retried by the shared retry
        # machinery: this API has no client-order-id/idempotency field (checked
        # against the real CreateOrderParams shape), so a retried POST after a
        # timeout or 5xx that actually succeeded server-side would place a
        # second, real, duplicate order with no way for the caller to detect
        # it. A single attempt that raises on failure is safer than a hidden
        # duplicate order.
        self.order_http = HttpClient(base_url=api_base_url, rate_limit=5, auth_hook=signer, max_retries=1)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.market_http.close()
        self.http.close()
        self.order_http.close()

    # ── Market data ─────────────────────────────────────────────────────────

    def get_market(self, slug: str) -> Market:
        data = self.market_http.get(f"/v1/market/slug/{slug}")
        bbo = None
        try:
            bbo = self.market_http.get(f"/v1/markets/{slug}/bbo")
        except Exception:
            pass
        return Market.from_dict(data["market"], bbo)

    def list_markets(
        self,
        active: bool = True,
        closed: bool = False,
        limit: int = 200,
        **filters,
    ) -> list[Market]:
        """One page of markets. Use list_all_markets() to page through everything."""
        params: dict = {"active": active, "closed": closed, "limit": limit}
        params.update(filters)
        data = self.market_http.get("/v1/markets", params=params)
        return [Market.from_dict(m) for m in data.get("markets", [])]

    def list_all_markets(
        self,
        active: bool = True,
        closed: bool = False,
        page_size: int = 200,
        **filters,
    ) -> list[Market]:
        all_markets: list[Market] = []
        offset = 0
        while True:
            params: dict = {
                "active": active,
                "closed": closed,
                "limit": page_size,
                "offset": offset,
            }
            params.update(filters)
            data = self.market_http.get("/v1/markets", params=params)
            page = data.get("markets", [])
            if not page:
                break
            all_markets.extend(Market.from_dict(m) for m in page)
            if len(page) < page_size:
                break
            offset += page_size
        return all_markets

    def get_bbo(self, slug: str) -> dict:
        return self.market_http.get(f"/v1/markets/{slug}/bbo")

    def get_settlement(self, slug: str) -> dict:
        return self.market_http.get(f"/v1/markets/{slug}/settlement")

    # ── Portfolio ────────────────────────────────────────────────────────────

    def get_balance(self) -> int:
        """Available buying power in cents."""
        data = self.http.get("/v1/account/balances")
        balances = data.get("balances", [])
        if not balances:
            return 0
        return round(float(balances[0].get("buyingPower", 0)) * 100)

    def get_positions(self) -> list[Position]:
        data = self.http.get("/v1/portfolio/positions")
        positions = data.get("positions", {}) or {}
        # Filter on the same "*Decimal" field Position.from_dict prefers —
        # confirmed live that "netPosition" truncates to an integer string
        # ("0") for real fractional holdings (netPositionDecimal="0.1600"),
        # which would otherwise silently drop a genuine non-zero position
        # from this list entirely, not just misreport its size.
        return [
            Position.from_dict(slug, p)
            for slug, p in positions.items()
            if float(p.get("netPositionDecimal", p.get("netPosition", 0)) or 0) != 0
        ]

    def get_open_orders(self) -> list[Order]:
        data = self.http.get("/v1/orders/open")
        return [Order.from_dict(o) for o in data.get("orders", [])]

    # ── Orders ───────────────────────────────────────────────────────────────

    def place_order(
        self,
        market_slug: str,
        side: str,
        count: int,
        yes_price: int,
        immediate_or_cancel: bool = False,
    ) -> Order:
        """Place a limit order.

        Args:
            market_slug: Market slug.
            side: "yes" (buy long) or "no" (buy short).
            count: Number of contracts.
            yes_price: YES-equivalent price in cents (1-99). For "no" orders
                pass the YES-equivalent (100 - no_price), matching the
                Kalshi/international-Polymarket client convention used
                elsewhere in this codebase.
            immediate_or_cancel: Fill what crosses right now and let the
                exchange kill the rest. Strongly preferred for hedged
                strategies — a GTC remainder can fill at any later moment,
                including in the window where the caller is cancelling it.

        The returned status is "filled" only when the response actually says
        so. It is "unknown" — never "resting" — when the response carries no
        execution information, because this API has been observed returning
        an empty executions array for orders that filled completely (both
        legs of the 2026-07-30 incident, and both flattening sells after it).
        Treating that silence as "did not fill" is what stranded real
        positions the bot believed it had cancelled. Callers that care must
        reconcile against get_positions(), not this field.
        """
        intent = "ORDER_INTENT_BUY_LONG" if side == "yes" else "ORDER_INTENT_BUY_SHORT"
        price_cents = yes_price if side == "yes" else (100 - yes_price)
        payload = {
            "marketSlug": market_slug,
            "intent": intent,
            "type": "ORDER_TYPE_LIMIT",
            "price": {"value": f"{price_cents / 100:.2f}", "currency": "USD"},
            "quantity": count,
            "tif": (
                "TIME_IN_FORCE_IMMEDIATE_OR_CANCEL" if immediate_or_cancel
                else "TIME_IN_FORCE_GOOD_TILL_CANCEL"
            ),
        }
        data = self.order_http.post("/v1/orders", json=payload)

        filled = data.get("cumQuantity")
        if filled is None:
            executions = data.get("executions", []) or []
            filled = (
                sum(int(float(e.get("lastShares", 0) or 0)) for e in executions)
                if executions else None
            )

        if filled is None:
            status = "unknown"
        elif int(float(filled)) >= count:
            status = "filled"
        else:
            status = "resting"

        return Order(
            order_id=data.get("id", ""),
            market_slug=market_slug,
            side=side,
            count=count,
            price_cents=yes_price,
            status=status,
        )

    def close_long(self, market_slug: str, count: int, price_cents: int) -> tuple[str, int]:
        """Sell an existing long position. Returns (order_id, contracts_sold).

        Immediate-or-cancel, not good-till-cancel, and that choice is the whole
        point: a resting sell order on an illiquid book can partially fill at
        any moment, and a cancel racing that fill is what left this account
        holding unhedged legs the bot had already written off. IOC either
        crosses now or is dead — there is no remainder to reconcile later.

        Fill count comes from cumQuantity (the order's own running total),
        falling back to summing executions. The bot's place_order reads only
        the executions array, which reports nothing when a fill lands after
        the response is serialized — that silent zero is what made a real
        position look like a no-op.
        """
        payload = {
            "marketSlug": market_slug,
            "intent": "ORDER_INTENT_SELL_LONG",
            "type": "ORDER_TYPE_LIMIT",
            "price": {"value": f"{price_cents / 100:.2f}", "currency": "USD"},
            "quantity": count,
            "tif": "TIME_IN_FORCE_IMMEDIATE_OR_CANCEL",
        }
        data = self.order_http.post("/v1/orders", json=payload)
        sold = data.get("cumQuantity")
        if sold is None:
            executions = data.get("executions", []) or []
            sold = sum(int(float(e.get("lastShares", 0) or 0)) for e in executions)
        return data.get("id", ""), int(float(sold or 0))

    def cancel_order(self, order_id: str, market_slug: str) -> bool:
        try:
            self.http.post(f"/v1/order/{order_id}/cancel", json={"marketSlug": market_slug})
            return True
        except Exception as exc:
            body = getattr(getattr(exc, "response", None), "text", "")
            logger.warning("cancel_order failed for %s: %s  body=%s", order_id, exc, body[:200])
            return False

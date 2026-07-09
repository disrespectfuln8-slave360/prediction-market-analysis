"""Authenticated Polymarket CLOB trading client.

Auth model
----------
Polymarket CLOB auth has three layers, all handled by py-clob-client:
  L1 — EIP-712 signature of your EOA private key, used once to derive
       (or create) L2 API credentials (key/secret/passphrase).
  L2 — HMAC signing of every authenticated REST call using those creds.
  Order signing — each order is itself an EIP-712-signed struct (maker,
       taker, token id, amounts, salt, nonce, fee, signature type).

We use the official py-clob-client library for all of the above rather than
hand-rolling EIP-712/HMAC signing ourselves — order-signing bugs are easy to
introduce and expensive to discover in a live-money trading path.

Market discovery uses the public gamma-api (which already reports live
best bid/ask and volume) rather than paginating the raw CLOB /markets
endpoint, which has no server-side tag or liquidity filter and would mean
scanning tens of thousands of markets every poll. Positions are read from
the public Polymarket data-api, since the CLOB itself is order-book only
and does not track portfolio positions.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    AssetType,
    BalanceAllowanceParams,
    OpenOrderParams,
    OrderArgs,
    OrderType,
)
from py_clob_client.order_builder.constants import BUY, SELL

from src.common.client import HttpClient
from src.trading.polymarket.models import Market, Order, Position

logger = logging.getLogger(__name__)

CLOB_URL = "https://clob.polymarket.com"
GAMMA_URL = "https://gamma-api.polymarket.com"
DATA_API_URL = "https://data-api.polymarket.com"
CHAIN_ID = 137  # Polygon mainnet


class PolymarketTradingClient:
    """Authenticated Polymarket CLOB client for portfolio and order management.

    Requires POLY_PRIVATE_KEY in the environment (or passed explicitly) — the
    private key of the wallet that funds and signs orders.

    Optional env vars:
      POLY_FUNDER_ADDRESS — proxy/Safe wallet address if trading through one
                             (Polymarket's hosted browser wallets use this).
                             Defaults to the signer's own address (direct EOA).
      POLY_SIGNATURE_TYPE  — 0 = EOA (default), 1 = POLY_PROXY, 2 = POLY_GNOSIS_SAFE.
    """

    def __init__(
        self,
        private_key: Optional[str] = None,
        funder: Optional[str] = None,
        signature_type: Optional[int] = None,
        host: str = CLOB_URL,
    ):
        private_key = private_key or os.environ["POLY_PRIVATE_KEY"]
        funder = funder or os.environ.get("POLY_FUNDER_ADDRESS") or None
        if signature_type is None:
            signature_type = int(os.environ.get("POLY_SIGNATURE_TYPE", "0"))

        self._clob = ClobClient(
            host,
            chain_id=CHAIN_ID,
            key=private_key,
            signature_type=signature_type,
            funder=funder,
        )
        creds = self._clob.create_or_derive_api_creds()
        self._clob.set_api_creds(creds)

        self.address = funder or self._clob.get_address()
        self._gamma_http = HttpClient(base_url=GAMMA_URL, rate_limit=5)
        self._data_http = HttpClient(base_url=DATA_API_URL, rate_limit=5)
        self._market_cache: dict[str, dict] = {}  # condition_id -> raw CLOB market

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self._gamma_http.close()
        self._data_http.close()

    # ── Market data ─────────────────────────────────────────────────────────

    def get_markets(self, next_cursor: Optional[str] = None) -> list[Market]:
        """One page of the raw CLOB market list (no live bid/ask)."""
        resp = self._clob.get_markets(next_cursor=next_cursor or "MA==")
        return [Market.from_clob(m) for m in resp.get("data", [])]

    def list_liquid_markets(self, tag_slugs: Optional[list[str]] = None) -> list[Market]:
        """Fetch active, open markets with live bid/ask via the gamma API.

        Pass tag_slugs to restrict to specific categories (e.g. ["crypto"]).
        Without tag_slugs, returns all active/open markets (bounded by limit).

        Note: the gamma /markets endpoint silently ignores a tag_slug filter
        (confirmed empirically — it returns the same unfiltered page regardless
        of the value passed). /events?tag_slug=... does filter correctly, so
        tag-scoped fetches go through events and flatten each event's nested
        "markets" list instead.
        """
        seen: dict[str, Market] = {}

        if tag_slugs:
            for slug in tag_slugs:
                params = {"active": "true", "closed": "false", "limit": 100, "tag_slug": slug}
                try:
                    events = self._gamma_http.get("/events", params=params)
                except Exception as exc:
                    logger.warning("Gamma events fetch failed (tag_slug=%s): %s", slug, exc)
                    continue
                for event in events if isinstance(events, list) else []:
                    # Nested markets don't carry their own "tags" — only the
                    # parent event does — so propagate it down before parsing.
                    event_tags = [t.get("slug") for t in event.get("tags", []) or []]
                    for m in event.get("markets", []) or []:
                        cid = m.get("conditionId")
                        if not cid or cid in seen:
                            continue
                        m = {**m, "tags": m.get("tags") or event_tags}
                        try:
                            seen[cid] = Market.from_gamma(m)
                        except Exception:
                            pass
        else:
            try:
                raw = self._gamma_http.get(
                    "/markets", params={"active": "true", "closed": "false", "limit": 200}
                )
            except Exception as exc:
                logger.warning("Gamma markets fetch failed: %s", exc)
                raw = []
            for m in raw if isinstance(raw, list) else []:
                cid = m.get("conditionId")
                if not cid or cid in seen:
                    continue
                try:
                    seen[cid] = Market.from_gamma(m)
                except Exception:
                    pass

        return list(seen.values())

    def _raw_market(self, condition_id: str) -> Optional[dict]:
        if condition_id in self._market_cache:
            return self._market_cache[condition_id]
        try:
            data = self._clob.get_market(condition_id)
        except Exception as exc:
            logger.debug("get_market failed for %s: %s", condition_id, exc)
            return None
        self._market_cache[condition_id] = data
        return data

    def _resolve_token_id(self, condition_id: str, outcome: str) -> Optional[str]:
        market = self._raw_market(condition_id)
        if not market:
            return None
        tokens = market.get("tokens", []) or []
        for t in tokens:
            if t.get("outcome", "").lower() == outcome.lower():
                return t.get("token_id")
        # Fallback: Yes/No binary convention — index 0 = Yes, index 1 = No.
        if outcome.lower() == "yes" and tokens:
            return tokens[0].get("token_id")
        if outcome.lower() == "no" and len(tokens) > 1:
            return tokens[1].get("token_id")
        return None

    def get_market_resolution(self, condition_id: str) -> Optional[dict]:
        """Fresh (uncached) raw CLOB market record, used to check settlement.

        Includes "closed" and, once resolved, each token's "winner" flag —
        deliberately bypasses _market_cache since that cache is meant for
        stable token-id lookups, not resolution status that changes over time.
        """
        try:
            return self._clob.get_market(condition_id)
        except Exception as exc:
            logger.debug("get_market_resolution failed for %s: %s", condition_id, exc)
            return None

    def _resolve_outcome(self, condition_id: str, token_id: str) -> str:
        market = self._raw_market(condition_id)
        if not market:
            return ""
        for t in market.get("tokens", []) or []:
            if t.get("token_id") == token_id:
                return t.get("outcome", "")
        return ""

    # ── Portfolio ────────────────────────────────────────────────────────────

    def get_balance(self) -> int:
        """Available USDC balance in cents."""
        resp = self._clob.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
        raw_units = int(resp.get("balance", 0))  # USDC has 6 decimals
        return raw_units // 10000  # → cents

    def get_positions(self) -> list[Position]:
        data = self._data_http.get("/positions", params={"user": self.address})
        rows = data if isinstance(data, list) else []
        return [Position.from_dict(p) for p in rows if float(p.get("size", 0) or 0) != 0]

    def get_open_orders(self) -> list[Order]:
        raw_orders = self._clob.get_orders(OpenOrderParams())
        orders = []
        for o in raw_orders:
            condition_id = o.get("market", "")
            outcome = self._resolve_outcome(condition_id, o.get("asset_id", ""))
            orders.append(Order.from_dict(o, outcome=outcome))
        return orders

    # ── Orders ───────────────────────────────────────────────────────────────

    def place_order(
        self,
        condition_id: str,
        outcome: str,
        size: float,
        price: int,
        side: str = "BUY",
    ) -> Order:
        """Place a GTC limit order.

        Args:
            condition_id: Market condition ID.
            outcome: Outcome name to trade (e.g. "Yes" / "No").
            size: Number of shares.
            price: Limit price in cents (1-99).
            side: "BUY" or "SELL".
        """
        token_id = self._resolve_token_id(condition_id, outcome)
        if not token_id:
            raise ValueError(f"Could not resolve token_id for {condition_id} outcome={outcome}")

        order_args = OrderArgs(
            token_id=token_id,
            price=price / 100.0,
            size=float(size),
            side=BUY if side.upper() == "BUY" else SELL,
        )
        signed_order = self._clob.create_order(order_args)
        resp = self._clob.post_order(signed_order, OrderType.GTC)

        if resp.get("success") is False:
            raise RuntimeError(resp.get("errorMsg") or "order placement failed")

        return Order(
            order_id=resp.get("orderID", resp.get("order_id", "")),
            condition_id=condition_id,
            outcome=outcome,
            size=size,
            price=price,
            status=resp.get("status", "live"),
        )

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._clob.cancel(order_id)
            return True
        except Exception:
            return False

"""Polymarket price client.

Reads live bid/ask prices from Polymarket's gamma API (no auth required).
Prices come in as fractions (0.0–1.0), converted to cents (0–100) here.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL  = "https://clob.polymarket.com"


@dataclass
class PolyMarket:
    question: str
    end_date: str          # ISO date "YYYY-MM-DD"
    slug: str              # market slug
    condition_id: str      # needed to place orders via PolymarketTradingClient
    yes_token_id: str
    no_token_id: str
    yes_bid: Optional[int]  # cents (0-100), None if no liquidity
    yes_ask: Optional[int]
    no_bid: Optional[int]
    no_ask: Optional[int]
    volume: float           # USD


class PolymarketClient:
    """Fetch current Polymarket prices using the public gamma API."""

    def __init__(self, timeout: int = 15):
        self._http = httpx.Client(timeout=timeout)

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def get_active_markets(
        self,
        end_date_min: Optional[str] = None,
        end_date_max: Optional[str] = None,
        limit: int = 200,
    ) -> list[PolyMarket]:
        """Return active Polymarket markets with bid/ask prices.

        end_date_min/max: ISO date strings "YYYY-MM-DD" to filter by close date.
        """
        params: dict = {"active": "true", "closed": "false", "limit": limit}
        if end_date_min:
            params["end_date_min"] = end_date_min
        if end_date_max:
            params["end_date_max"] = end_date_max

        try:
            r = self._http.get(f"{GAMMA_URL}/markets", params=params)
            r.raise_for_status()
            raw = r.json()
        except Exception as exc:
            logger.warning("Polymarket gamma API error: %s", exc)
            return []

        markets = raw if isinstance(raw, list) else []
        out = []
        for m in markets:
            try:
                out.append(self._parse(m))
            except Exception:
                pass
        return out

    def get_markets_by_keyword(self, keyword: str, limit: int = 100) -> list[PolyMarket]:
        """Search Polymarket for markets whose question contains keyword."""
        try:
            r = self._http.get(
                f"{GAMMA_URL}/events",
                params={"active": "true", "closed": "false", "limit": limit, "q": keyword},
            )
            r.raise_for_status()
            events = r.json() if isinstance(r.json(), list) else []
        except Exception as exc:
            logger.warning("Polymarket search error for '%s': %s", keyword, exc)
            return []

        out = []
        for event in events:
            for m in event.get("markets", []):
                try:
                    out.append(self._parse(m))
                except Exception:
                    pass
        return out

    def get_price(self, yes_token_id: str) -> Optional[tuple[int, int]]:
        """Fetch (best_bid, best_ask) in cents for a YES token from the CLOB.

        Falls back to None if the market has no liquidity.
        """
        try:
            r = self._http.get(f"{CLOB_URL}/book", params={"token_id": yes_token_id})
            r.raise_for_status()
            book = r.json()
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            bid = int(round(float(bids[0]["price"]) * 100)) if bids else None
            ask = int(round(float(asks[0]["price"]) * 100)) if asks else None
            return (bid, ask)
        except Exception as exc:
            logger.debug("CLOB price fetch failed for %s: %s", yes_token_id[:20], exc)
            return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse(m: dict) -> PolyMarket:
        raw_cids = m.get("clobTokenIds", "[]")
        cids = json.loads(raw_cids) if isinstance(raw_cids, str) else (raw_cids or [])

        raw_prices = m.get("outcomePrices", "[]")
        prices = json.loads(raw_prices) if isinstance(raw_prices, str) else (raw_prices or [])
        yes_price = float(prices[0]) if prices else None
        no_price  = float(prices[1]) if len(prices) > 1 else None

        best_bid = m.get("bestBid")
        best_ask = m.get("bestAsk")

        def _c(v) -> Optional[int]:
            if v is None:
                return None
            return int(round(float(v) * 100))

        # If market has bestBid/bestAsk (YES side), derive NO prices from mid
        yes_bid = _c(best_bid)
        yes_ask_book = _c(best_ask)
        yes_ask = yes_ask_book if yes_ask_book is not None else _c(yes_price)

        no_ask = (100 - yes_bid) if yes_bid is not None else _c(no_price)
        no_bid = (100 - yes_ask) if yes_ask is not None else None

        end = (m.get("endDateIso") or m.get("endDate") or "")[:10]

        return PolyMarket(
            question=m.get("question", ""),
            end_date=end,
            slug=m.get("slug", ""),
            condition_id=m.get("conditionId", ""),
            yes_token_id=cids[0] if cids else "",
            no_token_id=cids[1] if len(cids) > 1 else "",
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            volume=float(m.get("volume", 0) or 0),
        )

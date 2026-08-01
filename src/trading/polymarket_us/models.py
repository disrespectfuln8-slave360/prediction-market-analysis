"""Data models for the Polymarket US (polymarket.us) regulated exchange API.

Field shapes reverse-engineered from the official (Python 3.10+-only, hence
not directly usable in this 3.9 project) `polymarket-us` SDK source —
https://github.com/Polymarket/polymarket-us-python — not from guesswork.

Polymarket US models each market as a single long/short instrument (one
market = one specific outcome, e.g. "Bitcoin above $100k by Dec 31 - Yes"),
unlike the international Polymarket CLOB's two-separate-outcome-token model.
This is architecturally much closer to Kalshi's single-ticker YES/NO model,
so "side" here means "yes" (long — betting this outcome happens) or "no"
(short — betting it doesn't), mirroring the same convention used throughout
src/trading/kalshi/.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


def _amount_to_cents(amount: Optional[dict]) -> Optional[int]:
    """Amount objects are {"value": "0.5500", "currency": "USD"} — dollars as
    a string, not cents. Convert once here so the rest of the codebase can
    keep using integer cents like the Kalshi/international-Polymarket clients."""
    if not amount:
        return None
    try:
        return round(float(amount["value"]) * 100)
    except (KeyError, ValueError, TypeError):
        return None


def _parse_time(val: Optional[str]) -> Optional[datetime]:
    if not val:
        return None
    try:
        return datetime.fromisoformat(val.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _best_quotes(data: dict, bbo: Optional[dict]):
    """Return (yes_bid_cents, yes_ask_cents) from whichever source is freshest.

    Confirmed against the live API (the official SDK's TypedDicts don't
    match this): /v1/markets and /v1/market/slug/{slug} both embed
    bestBidQuote/bestAskQuote directly on the market record. A separately
    fetched /v1/markets/{slug}/bbo response instead nests bestBid/bestAsk
    one level down under a "marketData" key, and is preferred when passed
    since it's a fresher quote than the market record's snapshot.
    """
    if bbo:
        md = bbo.get("marketData", bbo)
        bid, ask = md.get("bestBid"), md.get("bestAsk")
        if bid or ask:
            return _amount_to_cents(bid), _amount_to_cents(ask)
    return _amount_to_cents(data.get("bestBidQuote")), _amount_to_cents(data.get("bestAskQuote"))


def _best_depths(bbo: Optional[dict]):
    """Return (bid_depth, ask_depth), or (None, None) when unknown.

    Only the /bbo endpoint carries these — the bulk /v1/markets records have
    no depth field at all, which is why a bulk-scan candidate can never be
    sized safely on its own.

    The unit is not documented (the official SDK types say only `int`) and
    could be contracts, resting orders, or price levels. Callers should read
    it as a contract count, because that is the conservative reading: if it
    actually means orders or levels the order is merely smaller than it
    needed to be, whereas the opposite error oversizes past the book. That
    oversizing is what drained the live account on 2026-07-31 — 4-contract
    orders against 3-deep books filled the first leg, failed the hedge, and
    paid the spread to unwind, roughly twenty times over.
    """
    if not bbo:
        return None, None
    md = bbo.get("marketData", bbo)
    def _as_int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None
    return _as_int(md.get("bidDepth")), _as_int(md.get("askDepth"))


@dataclass
class Market:
    slug: str
    id: str
    title: str
    category: str  # e.g. "sports" — there is no dual-outcome-token "outcome" field on this API
    tags: list
    active: bool
    closed: bool
    close_time: Optional[datetime]  # from endDate
    yes_bid: Optional[int]  # cents
    yes_ask: Optional[int]
    no_bid: Optional[int]
    no_ask: Optional[int]
    volume: float
    volume_24h: float
    bid_depth: Optional[int] = None  # only present when built with a /bbo record
    ask_depth: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict, bbo: Optional[dict] = None) -> "Market":
        """Build from a market record (list or single-slug fetch), optionally
        refined with a separately-fetched MarketBBO record."""
        yes_bid, yes_ask = _best_quotes(data, bbo)
        bid_depth, ask_depth = _best_depths(bbo)
        no_bid = (100 - yes_ask) if yes_ask is not None else None
        no_ask = (100 - yes_bid) if yes_bid is not None else None
        return cls(
            slug=data.get("slug", ""),
            id=str(data.get("id", "")),
            title=data.get("title", ""),
            category=data.get("category", ""),
            tags=data.get("tags", []) or [],
            active=bool(data.get("active", False)),
            closed=bool(data.get("closed", False)),
            close_time=_parse_time(data.get("endDate")),
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            volume=float(data.get("volume", 0) or 0),
            volume_24h=float(data.get("volume24hr", 0) or 0),
            bid_depth=bid_depth,
            ask_depth=ask_depth,
        )


@dataclass
class Position:
    market_slug: str
    position: float  # positive = net long (yes), negative = net short (no); shares/contracts

    @classmethod
    def from_dict(cls, market_slug: str, data: dict) -> "Position":
        # Confirmed against a live account: "netPosition" is an integer-
        # truncated string ("0", "2") while "netPositionDecimal" ("0.1600",
        # "2.4400") is the true fractional quantity — they silently diverge
        # for any non-whole-share holding. Using netPosition alone can
        # understate (or entirely hide, when it rounds to "0") a real
        # position. Prefer the decimal field; fall back only if it's absent.
        net = data.get("netPositionDecimal", data.get("netPosition", "0"))
        try:
            net_val = float(net)
        except (TypeError, ValueError):
            net_val = 0.0
        return cls(market_slug=market_slug, position=net_val)


@dataclass
class Order:
    order_id: str
    market_slug: str
    side: str  # "yes" (long) or "no" (short)
    count: float
    price_cents: int
    status: str  # our normalized: "resting" or "filled" or the raw ORDER_STATE_* string

    @classmethod
    def from_dict(cls, data: dict) -> "Order":
        intent = data.get("intent", "")
        side = "yes" if "LONG" in intent else "no"
        state = data.get("state", "")
        # Positions have both an integer-truncated field and a "*Decimal"
        # field that carries the true fractional value (confirmed live —
        # see Position.from_dict); no live order has been observed yet to
        # confirm whether Orders follow the same pattern, but prefer the
        # decimal variant defensively if the API ever includes one.
        leaves = data.get("leavesQuantityDecimal", data.get("leavesQuantity"))
        if state == "ORDER_STATE_FILLED" or (leaves is not None and float(leaves) == 0):
            status = "filled"
        elif state in ("ORDER_STATE_CANCELED", "ORDER_STATE_EXPIRED", "ORDER_STATE_REJECTED"):
            status = state
        else:
            status = "resting"
        return cls(
            order_id=data.get("id", ""),
            market_slug=data.get("marketSlug", ""),
            side=side,
            count=float(data.get("quantityDecimal", data.get("quantity", 0)) or 0),
            price_cents=_amount_to_cents(data.get("price")) or 0,
            status=status,
        )

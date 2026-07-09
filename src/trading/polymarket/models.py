from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


def _parse_time(val: Optional[str]) -> Optional[datetime]:
    if not val:
        return None
    try:
        return datetime.fromisoformat(val.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _to_cents(val) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(round(float(val) * 100))
    except (ValueError, TypeError):
        return None


@dataclass
class Market:
    condition_id: str
    question_id: str
    question: str
    description: str
    end_time: Optional[datetime]
    outcomes: list[str]
    outcome_prices: list[float]
    volume: float
    active: bool
    closed: bool
    archived: bool
    yes_ask: Optional[int]  # cents
    no_ask: Optional[int]
    yes_bid: Optional[int]
    no_bid: Optional[int]
    tags: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    @classmethod
    def from_gamma(cls, data: dict) -> "Market":
        """Build from a gamma-api.polymarket.com /markets record (camelCase, live bid/ask)."""
        raw_outcomes = data.get("outcomes", "[]")
        outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else (raw_outcomes or [])

        raw_prices = data.get("outcomePrices", "[]")
        prices = json.loads(raw_prices) if isinstance(raw_prices, str) else (raw_prices or [])
        prices = [float(p) for p in prices]

        best_bid = data.get("bestBid")
        best_ask = data.get("bestAsk")
        yes_bid = _to_cents(best_bid)
        yes_ask_book = _to_cents(best_ask)
        yes_ask = yes_ask_book if yes_ask_book is not None else (_to_cents(prices[0]) if prices else None)
        no_ask = (100 - yes_bid) if yes_bid is not None else None
        no_bid = (100 - yes_ask) if yes_ask is not None else None

        return cls(
            condition_id=data.get("conditionId", ""),
            question_id=data.get("questionID", ""),
            question=data.get("question", ""),
            description=data.get("description", ""),
            end_time=_parse_time(data.get("endDate")),
            outcomes=outcomes,
            outcome_prices=prices,
            volume=float(data.get("volume", 0) or 0),
            active=bool(data.get("active", False)),
            closed=bool(data.get("closed", False)),
            archived=bool(data.get("archived", False)),
            yes_ask=yes_ask,
            no_ask=no_ask,
            yes_bid=yes_bid,
            no_bid=no_bid,
            tags=list(data.get("tags") or []),
        )

    @classmethod
    def from_clob(cls, data: dict) -> "Market":
        """Build from a clob.polymarket.com /markets record (snake_case, no live bid/ask)."""
        tokens = data.get("tokens", []) or []
        outcomes = [t.get("outcome", "") for t in tokens]
        prices = [float(t.get("price", 0) or 0) for t in tokens]

        return cls(
            condition_id=data.get("condition_id", ""),
            question_id=data.get("question_id", ""),
            question=data.get("question", ""),
            description=data.get("description", ""),
            end_time=_parse_time(data.get("end_date_iso")),
            outcomes=outcomes,
            outcome_prices=prices,
            volume=0.0,
            active=bool(data.get("active", False)),
            closed=bool(data.get("closed", False)),
            archived=bool(data.get("archived", False)),
            yes_ask=None,
            no_ask=None,
            yes_bid=None,
            no_bid=None,
            tags=list(data.get("tags") or []),
        )


@dataclass
class Position:
    condition_id: str
    outcome: str
    size: float
    avg_price: float  # dollars (0-1)

    @classmethod
    def from_dict(cls, data: dict) -> "Position":
        return cls(
            condition_id=data.get("conditionId", ""),
            outcome=data.get("outcome", ""),
            size=float(data.get("size", 0) or 0),
            avg_price=float(data.get("avgPrice", 0) or 0),
        )


@dataclass
class Order:
    order_id: str
    condition_id: str
    outcome: str
    size: float
    price: int  # cents
    status: str

    @classmethod
    def from_dict(cls, data: dict, outcome: str = "") -> "Order":
        return cls(
            order_id=data.get("id", data.get("order_id", data.get("orderID", ""))),
            condition_id=data.get("market", data.get("condition_id", "")),
            outcome=outcome or data.get("outcome", ""),
            size=float(data.get("original_size", data.get("size", 0)) or 0),
            price=_to_cents(data.get("price")) or 0,
            status=data.get("status", ""),
        )

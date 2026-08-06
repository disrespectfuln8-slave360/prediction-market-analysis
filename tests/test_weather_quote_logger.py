"""Bounds parsing for the weather quote logger.

The bounds rules here encode settlement semantics verified from live
market descriptions on 2026-08-02 — inclusive integer-F buckets settled
on the NWS Climatological Report (Daily). Slugs are deliberately NOT
parsed: `lt76f` means <=75 while `gte76lt77f` includes 77, so slug
arithmetic mis-bounds buckets.
"""
from __future__ import annotations

from src.trading.polymarket_us.weather_quote_logger import parse_bucket_bounds


def test_between_title_is_inclusive():
    assert parse_bucket_bounds("80 to 81") == (80, 81)


def test_below_title():
    assert parse_bucket_bounds("79 or below") == (None, 79)


def test_above_title():
    assert parse_bucket_bounds("88 or above") == (88, None)


def test_unknown_title_is_rejected_not_guessed():
    assert parse_bucket_bounds("Will it rain?") is None
    assert parse_bucket_bounds("") is None

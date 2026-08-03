"""Bounds parsing and dead-bucket logic for the weather monitor.

The bounds rules here encode settlement semantics verified from live
market descriptions on 2026-08-02 — inclusive integer-F buckets settled
on the NWS Climatological Report (Daily). Slugs are deliberately NOT
parsed: `lt76f` means <=75 while `gte76lt77f` includes 77, so slug
arithmetic mis-bounds buckets.
"""
from __future__ import annotations

import math

from src.trading.polymarket_us.weather_monitor import parse_bucket_bounds


def test_between_title_is_inclusive():
    assert parse_bucket_bounds("80 to 81") == (80, 81)


def test_below_title():
    assert parse_bucket_bounds("79 or below") == (None, 79)


def test_above_title():
    assert parse_bucket_bounds("88 or above") == (88, None)


def test_unknown_title_is_rejected_not_guessed():
    assert parse_bucket_bounds("Will it rain?") is None
    assert parse_bucket_bounds("") is None


def test_dead_bucket_requires_strictly_exceeding_upper_bound():
    """An observation of 84.0F only guarantees the climate report reads
    >=84, which does NOT kill an '84 to 85' bucket — and must not kill
    '82 to 83' until the floor of the observed max is 84.
    """
    def dead(max_f, high):
        return high is not None and math.floor(max_f) > high

    assert not dead(83.9, 83)   # report could read 83
    assert dead(84.0, 83)       # report reads at least 84
    assert not dead(84.0, 84)   # bucket includes 84 — alive
    assert not dead(85.9, 85)   # still could be 85
    assert dead(86.2, 85)
    assert not dead(120.0, None)  # unbounded top bucket can't die from heat

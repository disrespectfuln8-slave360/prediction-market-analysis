"""Tests for the Kalshi weather backfill's incremental-resume logic.

_resume_start_date() re-checks a trailing window (RECHECK_DAYS) rather than
resuming strictly from the day after the latest captured date -- Kalshi's
~20 weather series don't all settle at the same moment, so a run landing
mid-way through a date's settlements would otherwise permanently skip
whichever cities settled after that run finished. Confirmed this could
actually happen: the first real backfill's cutoff date landed on a day
where the (recently-switched) settlement source hadn't posted any results
yet -- that specific case cost nothing (zero rows for that date), but a
genuinely partial date would have silently lost the rest forever under a
strict max+1 resume.
"""
from __future__ import annotations

import pandas as pd
import pytest

import scripts.backfill_kalshi_weather_history as bkwh


@pytest.fixture(autouse=True)
def _isolate_out_path(tmp_path, monkeypatch):
    monkeypatch.setattr(bkwh, "OUT_PATH", tmp_path / "weather_history.parquet")
    yield


def test_resume_start_date_falls_back_to_start_date_when_no_file_exists():
    assert bkwh._resume_start_date() == bkwh.START_DATE


def test_resume_start_date_rechecks_a_trailing_window_not_just_the_next_day():
    """Regression: a naive "day after the latest row" resume would never
    revisit 2026-08-12 once ANY row for it existed, even if only some of
    the ~20 series had settled by the time that row was captured."""
    df = pd.DataFrame([{"series": "KXHIGHNY", "date": "2026-08-12", "settlement": 1}])
    df.to_parquet(bkwh.OUT_PATH)

    resumed = bkwh._resume_start_date()

    assert resumed <= "2026-08-12"  # re-checks 2026-08-12 itself, not just after it
    assert resumed == "2026-08-10"  # exactly RECHECK_DAYS=3 back, inclusive


def test_resume_start_date_never_goes_before_start_date():
    df = pd.DataFrame([{"series": "KXHIGHNY", "date": bkwh.START_DATE, "settlement": 1}])
    df.to_parquet(bkwh.OUT_PATH)

    assert bkwh._resume_start_date() == bkwh.START_DATE


def test_resume_start_date_falls_back_when_the_file_is_unreadable():
    bkwh.OUT_PATH.write_text("not a real parquet file")
    assert bkwh._resume_start_date() == bkwh.START_DATE


def test_resume_start_date_falls_back_when_the_date_column_is_empty():
    df = pd.DataFrame({"series": pd.Series([], dtype=str), "date": pd.Series([], dtype=str)})
    df.to_parquet(bkwh.OUT_PATH)
    assert bkwh._resume_start_date() == bkwh.START_DATE

"""Tests for the weather backfill's incremental-resume logic.

_resume_start_date() is the mechanism that turned this from a one-shot
script (silently frozen at its first run's data forever -- the same bug
class found live on Kalshi's indexer the same night) into something that
can safely re-run on a schedule: get it wrong and either days get skipped
(a real gap in Apex's calibration data) or re-processed (duplicate rows
corrupting the calibration table's sample counts).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import backfill_weather_history as bwh  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_out_path(tmp_path, monkeypatch):
    monkeypatch.setattr(bwh, "OUT_PATH", tmp_path / "weather_history.parquet")
    yield


def test_resume_start_date_falls_back_to_start_date_when_no_file_exists():
    assert bwh._resume_start_date() == bwh.START_DATE


def test_resume_start_date_is_the_day_after_the_latest_row():
    df = pd.DataFrame([
        {"city": "nychigh", "date": "2026-08-10", "settlement": 1},
        {"city": "nychigh", "date": "2026-08-12", "settlement": 0},
    ])
    df.to_parquet(bwh.OUT_PATH)

    assert bwh._resume_start_date() == "2026-08-13"


def test_resume_start_date_falls_back_when_the_file_is_unreadable(tmp_path):
    bwh.OUT_PATH.write_text("not a real parquet file")

    assert bwh._resume_start_date() == bwh.START_DATE


def test_resume_start_date_falls_back_when_the_date_column_is_empty():
    df = pd.DataFrame({"city": pd.Series([], dtype=str), "date": pd.Series([], dtype=str)})
    df.to_parquet(bwh.OUT_PATH)

    assert bwh._resume_start_date() == bwh.START_DATE

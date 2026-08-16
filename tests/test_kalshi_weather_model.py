"""Tests for the per-series (city) decile-bucketed NOAA calibration engine
backing the Kalshi weather bot.

Mirrors polymarket_us/apex/weather_model.py's own test coverage shape --
same modeling approach (NOAA NBM forecast -> continuity-corrected Normal
CDF -> per-decile calibration gate), same per-city independence lesson,
just applied to Kalshi's own ticker/title format.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.trading.kalshi.weather_model import (
    MAX_CALIBRATION_ERROR_PCT,
    MIN_SAMPLE_FLOOR,
    KalshiWeatherGroundingEngine,
    _build_calibration_table,
    parse_bucket_title,
    series_of,
)


# ── Title/ticker parsing ─────────────────────────────────────────────────────

def test_parse_bucket_title_range():
    assert parse_bucket_title("Will the maximum temperature be 93-94° on Aug 15, 2026?") == (93.0, 94.0, "2026-08-15")


def test_parse_bucket_title_below():
    assert parse_bucket_title("Will the maximum temperature be <93° on Aug 15, 2026?") == (None, 93.0, "2026-08-15")


def test_parse_bucket_title_above():
    assert parse_bucket_title("Will the maximum temperature be >100° on Aug 15, 2026?") == (100.0, None, "2026-08-15")


def test_parse_bucket_title_unrecognized_shape_returns_none():
    assert parse_bucket_title("Will it rain in Miami tomorrow?") is None


def test_series_of_strips_the_date_and_bucket_suffix():
    assert series_of("KXHIGHTATL-26AUG15-B93.5") == "KXHIGHTATL"
    assert series_of("KXHIGHLAX-26AUG16-T100") == "KXHIGHLAX"


# ── Calibration table ─────────────────────────────────────────────────────────

def _row(series, mu, sigma, low, high, settlement):
    return {"series": series, "forecast_mu": mu, "forecast_sigma": sigma, "low": low, "high": high, "settlement": settlement}


def test_build_calibration_table_groups_and_scores_correctly():
    rows = [_row("KXHIGHNY", 75.0, 0.01, 75.0, 76.0, 1 if i < 27 else 0) for i in range(30)]
    table = _build_calibration_table(pd.DataFrame(rows))
    assert table[("KXHIGHNY", 90)]["count"] == 30
    assert table[("KXHIGHNY", 90)]["actual_rate"] == pytest.approx(90.0)
    assert table[("KXHIGHNY", 90)]["mean_predicted"] > 99.0


def test_build_calibration_table_scopes_each_series_independently():
    """Regression for the exact miscalibration found live 2026-08-15 when
    scoping Apex's own equivalent table: pooling cities hid a real,
    sample-backed problem in one city behind good calibration in others.
    Same rows, same decile, different series -- stats must stay separate."""
    rows = (
        [_row("KXHIGHLAX", 75.0, 0.01, 75.0, 76.0, 1 if i < 15 else 0) for i in range(30)]  # 50% actual, bad
        + [_row("KXHIGHNY", 75.0, 0.01, 75.0, 76.0, 1) for _ in range(30)]  # 100% actual, fine
    )
    table = _build_calibration_table(pd.DataFrame(rows))
    assert table[("KXHIGHLAX", 90)]["actual_rate"] == pytest.approx(50.0)
    assert table[("KXHIGHNY", 90)]["actual_rate"] == pytest.approx(100.0)


# ── Trustworthiness gate ──────────────────────────────────────────────────────

def test_trustworthiness_requires_the_sample_floor():
    engine = KalshiWeatherGroundingEngine.__new__(KalshiWeatherGroundingEngine)
    engine._calibration = {("KXHIGHNY", 90): {"count": MIN_SAMPLE_FLOOR - 1, "actual_rate": 90.0, "mean_predicted": 91.0}}
    trustworthy, _note, count = engine._trustworthiness("KXHIGHNY", 0.95)
    assert not trustworthy
    assert count == MIN_SAMPLE_FLOOR - 1


def test_trustworthiness_rejects_a_measured_miscalibration():
    engine = KalshiWeatherGroundingEngine.__new__(KalshiWeatherGroundingEngine)
    engine._calibration = {("KXHIGHNY", 90): {"count": MIN_SAMPLE_FLOOR + 10, "actual_rate": 60.0, "mean_predicted": 95.0}}
    trustworthy, note, _count = engine._trustworthiness("KXHIGHNY", 0.95)
    assert not trustworthy
    assert f"{MAX_CALIBRATION_ERROR_PCT:.0f}pp" in note


def test_trustworthiness_passes_a_well_calibrated_bucket():
    engine = KalshiWeatherGroundingEngine.__new__(KalshiWeatherGroundingEngine)
    engine._calibration = {("KXHIGHNY", 90): {"count": MIN_SAMPLE_FLOOR + 10, "actual_rate": 92.0, "mean_predicted": 94.0}}
    trustworthy, _note, _count = engine._trustworthiness("KXHIGHNY", 0.95)
    assert trustworthy


def test_trustworthiness_is_scoped_independently_per_series():
    engine = KalshiWeatherGroundingEngine.__new__(KalshiWeatherGroundingEngine)
    engine._calibration = {
        ("KXHIGHLAX", 60): {"count": MIN_SAMPLE_FLOOR + 62, "actual_rate": 46.2, "mean_predicted": 62.6},  # bad
        ("KXHIGHNY", 60): {"count": MIN_SAMPLE_FLOOR + 5, "actual_rate": 64.0, "mean_predicted": 62.5},  # fine
    }
    lax_ok, _n, _c = engine._trustworthiness("KXHIGHLAX", 0.65)
    ny_ok, _n, _c = engine._trustworthiness("KXHIGHNY", 0.65)
    assert not lax_ok
    assert ny_ok


def test_evaluate_checks_trustworthiness_independently_per_side():
    """Same per-side independence lesson already fixed in Apex's own
    calibration gate -- a bucket calibrated as trustworthy on one side must
    not silently green-light an unverified complementary side."""
    engine = KalshiWeatherGroundingEngine.__new__(KalshiWeatherGroundingEngine)
    engine._calibration = {("KXHIGHNY", 0): {"count": MIN_SAMPLE_FLOOR + 10, "actual_rate": 5.0, "mean_predicted": 5.0}}
    grounding = engine.evaluate("KXHIGHNY", "2026-08-15", 80.0, 81.0, forecast_mu=60.0, forecast_sigma=1.0, forecast_cycle="c")
    assert grounding is not None
    assert grounding.model_prob < 0.10
    assert grounding.calibration_trustworthy
    assert not grounding.calibration_trustworthy_no


# ── Staleness ────────────────────────────────────────────────────────────────

def test_no_history_is_stale(tmp_path):
    engine = KalshiWeatherGroundingEngine(history_path=tmp_path / "missing.parquet")
    assert engine.history_is_stale()


def test_recent_history_is_not_stale():
    engine = KalshiWeatherGroundingEngine.__new__(KalshiWeatherGroundingEngine)
    from datetime import datetime, timezone
    engine._history_max_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert not engine.history_is_stale()

"""Calibration analysis for the NOAA-forecast probability model on polymarket.us weather markets.

For each resolved daily-high-temperature bucket market, derives the model-implied
win probability from the NBM forecast (mean=forecast_mu, std=forecast_sigma) under
a Normal distribution with a +/-0.5F continuity correction (settlement rounds to
integer F), then groups into decile probability buckets and compares predicted vs
actual resolution rate — the same calibration-curve format as
polymarket_calibration_by_bucket.py, so the two are directly comparable.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType

# Mirrors src/trading/kalshi/learning.py's MIN_BETS_FOR_ADJUSTMENT — the sample
# floor this codebase already trusts before leaning on a bucket's stats.
MIN_SAMPLE_FLOOR = 25


class WeatherForecastCalibrationAnalysis(Analysis):
    """Calibration of NOAA-forecast-implied probabilities against polymarket.us weather settlements."""

    def __init__(self, data_dir: Path | str | None = None):
        super().__init__(
            name="weather_forecast_calibration",
            description="NOAA forecast-implied probability calibration against polymarket.us weather settlements",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.data_dir = Path(data_dir or base_dir / "data" / "polymarket_us" / "weather_history")

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        if not self.data_dir.exists() or not any(self.data_dir.glob("*.parquet")):
            return AnalysisOutput(data=pd.DataFrame(), metadata={"note": "no backfilled data yet"})

        with self.progress("Loading weather forecast history"):
            con = duckdb.connect()
            try:
                raw = con.execute(f"SELECT * FROM '{self.data_dir}/*.parquet'").df()
            except duckdb.Error:
                return AnalysisOutput(data=pd.DataFrame(), metadata={"note": "no backfilled data yet"})

        if raw.empty:
            return AnalysisOutput(data=pd.DataFrame(), metadata={"note": "no backfilled data yet"})

        with self.progress("Computing model-implied probabilities"):
            raw = self._add_model_prob(raw)

        df = self._bucket_by_decile(raw)
        metrics = self._compute_metrics(df, raw)

        fig = self._create_figure(df, metrics)
        chart = self._create_chart(df)

        return AnalysisOutput(figure=fig, data=df, chart=chart, metadata=metrics)

    def _add_model_prob(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach model_prob = P(low <= actual_high <= high) under Normal(forecast_mu, forecast_sigma).

        Bounds get a +/-0.5F continuity correction since settlement rounds to integer F.
        Null low/high mean unbounded (-inf/+inf).
        """
        df = df.copy()
        mu = df["forecast_mu"].astype(float).to_numpy()
        sigma = df["forecast_sigma"].astype(float).to_numpy()
        low = pd.to_numeric(df["low"], errors="coerce").to_numpy()
        high = pd.to_numeric(df["high"], errors="coerce").to_numpy()

        low_bound = np.where(np.isnan(low), -np.inf, low - 0.5)
        high_bound = np.where(np.isnan(high), np.inf, high + 0.5)

        # scale must be > 0 for norm.cdf; degenerate (sigma<=0) forecasts fall back
        # to a point mass at mu instead of dividing by zero.
        valid_sigma = sigma > 0
        safe_sigma = np.where(valid_sigma, sigma, 1.0)
        model_prob = norm.cdf(high_bound, loc=mu, scale=safe_sigma) - norm.cdf(low_bound, loc=mu, scale=safe_sigma)

        within_bounds = (mu >= low_bound) & (mu <= high_bound)
        model_prob = np.where(valid_sigma, model_prob, within_bounds.astype(float))

        df["model_prob"] = np.clip(model_prob, 0.0, 1.0)
        return df.dropna(subset=["model_prob", "settlement"])

    def _bucket_by_decile(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Group rows into 0-10%, 10-20%, ..., 90-100% buckets by model_prob."""
        pct = raw["model_prob"].to_numpy() * 100.0
        # clamp so an exact 100% prediction lands in the top bucket, not an 11th one
        bucket_low = np.minimum(np.floor(pct / 10.0) * 10.0, 90.0)

        tmp = pd.DataFrame(
            {
                "bucket_low": bucket_low.astype(int),
                "model_prob_pct": pct,
                "settlement": raw["settlement"].astype(int).to_numpy(),
            }
        )
        grouped = (
            tmp.groupby("bucket_low")
            .agg(count=("settlement", "size"), wins=("settlement", "sum"), mean_predicted=("model_prob_pct", "mean"))
            .reset_index()
        )
        grouped["actual_rate"] = 100.0 * grouped["wins"] / grouped["count"]
        return grouped.sort_values("bucket_low").reset_index(drop=True)

    def _compute_metrics(self, df: pd.DataFrame, raw: pd.DataFrame) -> dict:
        """Compute calibration summary metrics plus the sample-size facts the bot's learning floor cares about.

        ECE: row-count-weighted mean of |actual_rate - mean_predicted| across buckets.
        MCE: largest absolute deviation in any single bucket.
        Brier: mean((model_prob - settlement)^2) computed per-row (exact, not bucket-approximated).
        """
        total_rows = int(df["count"].sum())
        deviations = np.abs(df["actual_rate"] - df["mean_predicted"])
        ece = float(np.average(deviations, weights=df["count"])) if total_rows else 0.0
        mce = float(deviations.max()) if not df.empty else 0.0

        brier_score = float(np.mean((raw["model_prob"].to_numpy() - raw["settlement"].astype(float).to_numpy()) ** 2))

        city_days = raw[["city", "date"]].drop_duplicates()
        rows_per_city = {str(k): int(v) for k, v in raw.groupby("city").size().items()}
        buckets_meeting_floor = int((df["count"] >= MIN_SAMPLE_FLOOR).sum())

        return {
            "ece": round(ece, 2),
            "mce": round(mce, 2),
            "brier_score": round(brier_score, 4),
            "num_buckets": len(df),
            "total_resolved_city_days": int(len(city_days)),
            "total_resolved_bucket_rows": int(len(raw)),
            "rows_per_city": rows_per_city,
            "min_sample_floor": MIN_SAMPLE_FLOOR,
            "buckets_meeting_min_sample": buckets_meeting_floor,
            "buckets_below_min_sample": len(df) - buckets_meeting_floor,
        }

    def _create_figure(self, df: pd.DataFrame, metrics: dict) -> plt.Figure:
        """Create a grouped bar chart comparing predicted vs actual per bucket."""
        fig, ax = plt.subplots(figsize=(12, 7))

        bucket_labels = [f"{int(row['bucket_low'])}-{int(row['bucket_low']) + 10}%" for _, row in df.iterrows()]
        x = np.arange(len(bucket_labels))
        bar_width = 0.35

        ax.bar(
            x - bar_width / 2,
            df["mean_predicted"],
            bar_width,
            label="Mean Predicted (%)",
            color="#4C72B0",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.bar(
            x + bar_width / 2,
            df["actual_rate"],
            bar_width,
            label="Actual Settlement Rate (%)",
            color="#DD8452",
            edgecolor="white",
            linewidth=0.5,
        )

        # Perfect calibration reference line
        ax.plot(
            x,
            df["mean_predicted"],
            linestyle="--",
            color="#D65F5F",
            linewidth=1,
            alpha=0.5,
        )

        ax.set_xlabel("Model-Implied Probability Bucket")
        ax.set_ylabel("Rate (%)")
        ax.set_title("polymarket.us Weather: NOAA Forecast Calibration by Probability Bucket")
        ax.set_xticks(x)
        ax.set_xticklabels(bucket_labels, rotation=45, ha="right")
        ax.set_ylim(0, 105)
        ax.legend(loc="upper left")
        ax.grid(axis="y", alpha=0.3)

        # Row count labels above actual bars
        for i, (_, row) in enumerate(df.iterrows()):
            ax.text(
                x[i] + bar_width / 2,
                row["actual_rate"] + 1.5,
                str(int(row["count"])),
                ha="center",
                va="bottom",
                fontsize=7,
                color="#666666",
            )

        metrics_text = (
            f"ECE: {metrics['ece']:.2f}%\n"
            f"MCE: {metrics['mce']:.2f}%\n"
            f"Brier: {metrics['brier_score']:.4f}\n"
            f"Rows: {metrics['total_resolved_bucket_rows']:,}"
        )
        ax.text(
            0.98,
            0.50,
            metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="center",
            horizontalalignment="right",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        plt.tight_layout()
        return fig

    def _create_chart(self, df: pd.DataFrame) -> ChartConfig:
        """Create the chart configuration for web display."""
        chart_data = [
            {
                "bucket": f"{int(row['bucket_low'])}-{int(row['bucket_low']) + 10}%",
                "predicted": round(row["mean_predicted"], 2),
                "actual": round(row["actual_rate"], 2),
                "rows": int(row["count"]),
            }
            for _, row in df.iterrows()
        ]

        return ChartConfig(
            type=ChartType.BAR,
            data=chart_data,
            xKey="bucket",
            yKeys=["predicted", "actual"],
            title="polymarket.us Weather: NOAA Forecast Calibration by Probability Bucket",
            yUnit=UnitType.PERCENT,
            xLabel="Model-Implied Probability Bucket",
            yLabel="Rate (%)",
        )

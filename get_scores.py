"""
Performance scoring for F1 practice sessions.

Calculates composite scores from lap telemetry across multiple dimensions:
pace (best/mean/worst lap), consistency (std dev), sector times, and speed traps.
All individual scores are normalized to [0, 1] before weighting.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from constants import SCORE_WEIGHTS, SPEED_COLUMNS
from get_data import get_driver_map, get_lap_data, load_session

logger = logging.getLogger(__name__)


def normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize a series to [0, 1]."""
    range_ = series.max() - series.min()
    if range_ == 0:
        return pd.Series(0.5, index=series.index)
    return (series - series.min()) / range_


def _inverse_ratio_score(values: pd.Series) -> pd.Series:
    """Score where lower values are better (times): 1 / (value / mean)."""
    return 1 / (values / values.mean())


def _ratio_score(values: pd.Series) -> pd.Series:
    """Score where higher values are better (speeds): value / mean."""
    return values / values.mean()


def compute_session_scores(session) -> pd.DataFrame:
    """
    Compute a weighted composite score for each driver in a session.

    Scoring dimensions and their weights (from constants.SCORE_WEIGHTS):
      - Lap time scores: best (2x), mean (1x), worst (0.5x)
      - Lap consistency: std dev of lap times (0.5x)
      - Per-sector scores: best/mean/worst time + consistency (3 sectors)
      - Top speed: composite across all speed traps (1x)

    Returns a DataFrame with columns [Score, Driver] sorted descending.
    """
    lap_data = get_lap_data(session)
    if lap_data.empty:
        return pd.DataFrame(columns=["Score", "Driver"])

    w = SCORE_WEIGHTS
    total = pd.Series(0.0, index=lap_data.index)

    # --- Lap time scores ---
    total += normalize(_inverse_ratio_score(lap_data["LapTime_min"])) * w["min_lap_time"]
    total += normalize(_inverse_ratio_score(lap_data["LapTime_mean"])) * w["mean_lap_time"]
    total += normalize(_inverse_ratio_score(lap_data["LapTime_max"])) * w["max_lap_time"]
    total += normalize(_inverse_ratio_score(lap_data["LapTime_std"])) * w["lap_consistency"]

    # --- Sector scores (same pattern for each of the 3 sectors) ---
    for i in range(1, 4):
        sector = f"Sector{i}Time"
        total += normalize(_inverse_ratio_score(lap_data[f"{sector}_min"])) * w["sector_best_time"]
        total += normalize(_inverse_ratio_score(lap_data[f"{sector}_mean"])) * w["sector_mean_time"]
        total += normalize(_inverse_ratio_score(lap_data[f"{sector}_max"])) * w["sector_max_time"]
        total += normalize(_inverse_ratio_score(lap_data[f"{sector}_std"])) * w["sector_consistency"]

    # --- Top speed (composite across all speed traps) ---
    speed_score = sum(
        _ratio_score(lap_data[f"{col}_max"]) for col in SPEED_COLUMNS
    )
    total += normalize(speed_score) * w["top_speed"]

    # Build result
    driver_map = get_driver_map(session)
    result = pd.DataFrame({"Score": total}).sort_values("Score", ascending=False)
    result["Driver"] = result.index.to_series().astype(str).map(driver_map)
    return result


def get_combined_practice_scores(year: int, gp: str | int) -> pd.DataFrame:
    """Load FP1+FP2, compute scores, and return combined rankings."""
    all_scores = []
    for fp in ["FP1", "FP2"]:
        try:
            session = load_session(year, gp, fp)
            scores = compute_session_scores(session)
            all_scores.append(scores)
        except Exception:
            logger.warning("Could not load %s %s %s", year, gp, fp, exc_info=True)

    if not all_scores:
        return pd.DataFrame(columns=["Score", "Driver"])

    combined = (
        pd.concat(all_scores)
        .groupby("Driver")["Score"]
        .sum()
        .sort_values(ascending=False)
    )
    # Normalize combined scores
    combined = normalize(combined)
    return combined.to_frame("Score")


def main() -> None:
    parser = argparse.ArgumentParser(description="F1 practice session performance scores")
    parser.add_argument("--year", type=int, default=2024, help="Season year")
    parser.add_argument("--gp", required=True, help="Grand Prix name (e.g. 'Bahrain')")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = get_combined_practice_scores(args.year, args.gp)
    print(result.to_string())


if __name__ == "__main__":
    main()

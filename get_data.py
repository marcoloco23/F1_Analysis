"""
Data extraction and aggregation from fastf1 session telemetry.

Extracts per-driver lap statistics (mean/max/min/std) from a loaded session,
encoding categorical columns and converting timedeltas to seconds.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

from constants import AGGREGATIONS, COMPOUND_MAP, LAP_DATA_COLUMNS, TIME_COLUMNS

logger = logging.getLogger(__name__)

# Default cache directory (portable, inside project)
CACHE_DIR = Path(__file__).resolve().parent / "f1_cache"


def enable_cache(cache_dir: Path | str | None = None) -> None:
    """Enable the fastf1 disk cache at the given (or default) directory."""
    path = Path(cache_dir) if cache_dir else CACHE_DIR
    path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(path))


def load_session(year: int, gp: str | int, session_type: str) -> fastf1.core.Session:
    """Load a fastf1 session with caching enabled."""
    enable_cache()
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    return session


def get_lap_data(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Extract aggregated per-driver lap statistics from a session.

    Returns a DataFrame indexed by DriverNumber with columns like
    LapTime_mean, LapTime_max, Sector1Time_min, SpeedI1_std, etc.
    """
    laps = session.laps
    accurate = laps.loc[laps["IsAccurate"] == True, LAP_DATA_COLUMNS].dropna()  # noqa: E712

    if accurate.empty:
        logger.warning("No accurate laps found in session")
        return pd.DataFrame()

    # Convert timedeltas to seconds
    for col in TIME_COLUMNS:
        accurate[col] = accurate[col] / np.timedelta64(1, "s")

    # Encode tire compound as integer
    accurate["Compound"] = accurate["Compound"].map(COMPOUND_MAP).fillna(1).astype(int)
    accurate["FreshTyre"] = accurate["FreshTyre"].astype(int)

    # Aggregate per driver
    grouped = accurate.groupby("DriverNumber")
    parts = [getattr(grouped, agg)().add_suffix(f"_{agg}") for agg in AGGREGATIONS]
    result = pd.concat(parts, axis=1)

    # Drop the DriverNumber aggregation columns (it's the index)
    driver_num_cols = [c for c in result.columns if c.startswith("DriverNumber_")]
    result = result.drop(columns=driver_num_cols)

    return result


def get_driver_map(session: fastf1.core.Session) -> dict[str, str]:
    """Build a DriverNumber -> Abbreviation map from session results."""
    results = session.results
    return dict(zip(results["DriverNumber"].astype(str), results["Abbreviation"]))

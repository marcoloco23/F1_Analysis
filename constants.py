"""
F1 Analysis constants and configuration.

Driver/GP data is loaded dynamically from fastf1 for any season.
The constants below define the telemetry columns and session types used throughout.
"""

from __future__ import annotations

# Telemetry columns extracted per lap
LAP_DATA_COLUMNS: list[str] = [
    "DriverNumber",
    "LapNumber",
    "LapTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "SpeedI1",
    "SpeedI2",
    "SpeedFL",
    "SpeedST",
    "Compound",
    "TyreLife",
    "FreshTyre",
]

# Columns containing timedelta values that need conversion to seconds
TIME_COLUMNS: list[str] = ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]

# Speed trap columns
SPEED_COLUMNS: list[str] = ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]

# Free practice sessions used for training data
PRACTICE_SESSIONS: list[str] = ["FP1", "FP2", "FP3"]

# Aggregation functions applied per driver
AGGREGATIONS: list[str] = ["mean", "max", "min", "std"]

# Tire compound encoding (fastf1 uses string names)
COMPOUND_MAP: dict[str, int] = {
    "HARD": 0,
    "MEDIUM": 1,
    "SOFT": 2,
    "INTERMEDIATE": 3,
    "WET": 4,
}

# Score weights for composite session scoring
SCORE_WEIGHTS: dict[str, float] = {
    "min_lap_time": 2.0,
    "mean_lap_time": 1.0,
    "max_lap_time": 0.5,
    "lap_consistency": 0.5,
    "sector_best_time": 2.0,      # applied per sector
    "sector_mean_time": 1.0,      # applied per sector
    "sector_max_time": 0.5,       # applied per sector
    "sector_consistency": 0.5,    # applied per sector
    "top_speed": 1.0,
}

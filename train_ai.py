"""
Train an XGBoost model to predict F1 race points from practice session telemetry.

Usage:
    # Build dataset from a season and train
    python train_ai.py --year 2024 --prepare

    # Train on existing data.csv
    python train_ai.py

    # Evaluate with cross-validation
    python train_ai.py --cv
"""

from __future__ import annotations

import argparse
import logging

import fastf1
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler

from constants import PRACTICE_SESSIONS
from get_data import enable_cache, get_lap_data

logger = logging.getLogger(__name__)

MODEL_PATH = "f1_model.json"
DATA_PATH = "data.csv"


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max scale all feature columns to [0, 1]."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)
    return pd.DataFrame(scaled, columns=df.columns, index=df.index)


def prepare_dataset(year: int) -> pd.DataFrame:
    """
    Build a training dataset from an entire F1 season.

    For each Grand Prix, loads FP1/FP2/FP3 telemetry, aggregates per driver,
    normalizes features, and joins with actual race points.
    """
    enable_cache()
    schedule = fastf1.get_event_schedule(year, include_testing=False)

    data_rows = []
    for _, event in schedule.iterrows():
        gp_name = event["EventName"]
        round_num = event["RoundNumber"]
        if round_num < 1:
            continue

        logger.info("Processing %s (Round %d)", gp_name, round_num)
        try:
            # Load race results for target variable
            race = fastf1.get_session(year, round_num, "R")
            race.load()

            # Load practice sessions
            practice_dfs = []
            for fp in PRACTICE_SESSIONS:
                try:
                    session = fastf1.get_session(year, round_num, fp)
                    session.load()
                    lap_data = get_lap_data(session)
                    if not lap_data.empty:
                        practice_dfs.append(normalize_features(lap_data))
                except Exception:
                    logger.debug("Skipping %s %s %s", gp_name, fp, exc_info=True)

            if not practice_dfs:
                logger.warning("No practice data for %s, skipping", gp_name)
                continue

            # Average across practice sessions per driver
            combined = pd.concat(practice_dfs).groupby(level=0).mean()
            combined["Points"] = race.results.set_index("DriverNumber")["Points"]
            combined = combined.dropna(subset=["Points"])
            data_rows.append(combined)

        except Exception:
            logger.warning("Failed to process %s", gp_name, exc_info=True)

    if not data_rows:
        raise RuntimeError(f"No data collected for {year} season")

    dataset = pd.concat(data_rows).reset_index(drop=True).fillna(0)
    dataset.to_csv(DATA_PATH, index=False)
    logger.info("Dataset saved: %s (%d rows, %d columns)", DATA_PATH, *dataset.shape)
    return dataset


def train_model(data: pd.DataFrame, use_cv: bool = False) -> xgb.XGBRegressor:
    """
    Train an XGBoost regressor on the dataset.

    Args:
        data: DataFrame with feature columns and a 'Points' target column.
        use_cv: If True, also run 5-fold cross-validation and log results.
    """
    X = data.drop(columns=["Points"])
    y = data["Points"]

    # Train/test split
    train_mask = np.random.RandomState(42).rand(len(data)) < 0.8
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    model = xgb.XGBRegressor(
        booster="gbtree",
        objective="reg:squarederror",
        learning_rate=0.01,
        n_estimators=5000,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="rmse",
        early_stopping_rounds=50,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    logger.info("Test RMSE: %.3f  |  R²: %.3f  |  Best iteration: %d", rmse, r2, model.best_iteration)

    if use_cv:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            xgb.XGBRegressor(
                booster="gbtree", objective="reg:squarederror",
                learning_rate=0.01, n_estimators=model.best_iteration,
                max_depth=6, subsample=0.8, colsample_bytree=0.8,
            ),
            X, y, cv=kf, scoring="r2",
        )
        logger.info("5-Fold CV R²: %.3f ± %.3f", cv_scores.mean(), cv_scores.std())

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train F1 race points prediction model")
    parser.add_argument("--year", type=int, default=2024, help="Season year for dataset prep")
    parser.add_argument("--prepare", action="store_true", help="Build dataset from season data")
    parser.add_argument("--cv", action="store_true", help="Run cross-validation after training")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.prepare:
        data = prepare_dataset(args.year)
    else:
        data = pd.read_csv(DATA_PATH)
        logger.info("Loaded %s (%d rows)", DATA_PATH, len(data))

    model = train_model(data, use_cv=args.cv)
    model.save_model(MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)


if __name__ == "__main__":
    main()

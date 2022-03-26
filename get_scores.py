import fastf1
import numpy as np
import pandas as pd
import argparse
from constants import DRIVER_NUMBER_TO_DRIVER
from get_data import get_lap_data

pd.options.mode.chained_assignment = None

parser = argparse.ArgumentParser()
parser.add_argument("-gp", "--gp", help="GP String")
args = parser.parse_args()
fastf1.Cache.enable_cache("/Users/marcsperzel/Desktop/F1")


def normalize(df: pd.DataFrame):
    return (df - df.min()) / (df.max() - df.min())


def get_min_time_score(lap_data: pd.DataFrame):
    # Measure best lap relative to field
    score = 1 / (lap_data.LapTime_min / lap_data.LapTime_min.mean())
    return score


def get_mean_time_score(lap_data: pd.DataFrame):
    # Measure average lap relative to field
    score = 1 / (lap_data.LapTime_mean / lap_data.LapTime_mean.mean())
    return score


def get_max_time_score(lap_data: pd.DataFrame):
    # Measure worst lap relative to field
    score = 1 / (lap_data.LapTime_max / lap_data.LapTime_mean.max())
    return score


def get_lap_consistency_score(lap_data: pd.DataFrame):
    # Measure consistency in lap times relative to field
    score = 1 / (lap_data.LapTime_std / lap_data.LapTime_std.mean())
    return score


def get_sector_1_consistency_score(lap_data: pd.DataFrame):
    # Measure consistency in sector times relative to field
    score = 1 / (lap_data.Sector1Time_std / lap_data.Sector1Time_std.mean())
    return score


def get_sector_2_consistency_score(lap_data: pd.DataFrame):
    # Measure consistency in sector times relative to field
    score = 1 / (lap_data.Sector2Time_std / lap_data.Sector2Time_std.mean())
    return score


def get_sector_3_consistency_score(lap_data: pd.DataFrame):
    # Measure consistency in sector times relative to field
    score = 1 / (lap_data.Sector3Time_std / lap_data.Sector3Time_std.mean())
    return score


def get_sector_1_time_score(lap_data: pd.DataFrame):
    # Measure fastest sector time relative to field
    score = 1 / (lap_data.Sector1Time_min / lap_data.Sector1Time_min.mean())
    return score


def get_sector_2_time_score(lap_data: pd.DataFrame):
    # Measure fastest sector time relative to field
    score = 1 / (lap_data.Sector2Time_min / lap_data.Sector2Time_min.mean())
    return score


def get_sector_3_time_score(lap_data: pd.DataFrame):
    # Measure fastest sector time relative to field
    score = 1 / (lap_data.Sector3Time_min / lap_data.Sector3Time_min.mean())
    return score


def get_sector_1_mean_time_score(lap_data: pd.DataFrame):
    # Measure average sector time relative to field
    score = 1 / (lap_data.Sector1Time_mean / lap_data.Sector1Time_mean.mean())
    return score


def get_sector_2_mean_time_score(lap_data: pd.DataFrame):
    # Measure average sector time relative to field
    score = 1 / (lap_data.Sector2Time_mean / lap_data.Sector2Time_mean.mean())
    return score


def get_sector_3_mean_time_score(lap_data: pd.DataFrame):
    # Measure average sector time relative to field
    score = 1 / (lap_data.Sector3Time_mean / lap_data.Sector3Time_mean.mean())
    return score


def get_sector_1_max_time_score(lap_data: pd.DataFrame):
    # Measure worst sector time relative to field
    score = 1 / (lap_data.Sector1Time_max / lap_data.Sector1Time_max.mean())
    return score


def get_sector_2_max_time_score(lap_data: pd.DataFrame):
    # Measure worst sector time relative to field
    score = 1 / (lap_data.Sector2Time_max / lap_data.Sector2Time_max.mean())
    return score


def get_sector_3_max_time_score(lap_data: pd.DataFrame):
    # Measure worst sector time relative to field
    score = 1 / (lap_data.Sector3Time_max / lap_data.Sector3Time_max.mean())
    return score


def get_tire_life_score(lap_data: pd.DataFrame):
    # Measure mean tire life relative to field
    score = lap_data.TyreLife_mean / lap_data.TyreLife_mean.mean()
    return score


def get_top_speed_score(lap_data: pd.DataFrame):
    # Measure top speed relative to field
    speed_score = (
        (lap_data.SpeedI1_max / lap_data.SpeedI1_max.mean())
        + (lap_data.SpeedI2_max / lap_data.SpeedI2_max.mean())
        + (lap_data.SpeedFL_max / lap_data.SpeedFL_max.mean())
        + (lap_data.SpeedST_max / lap_data.SpeedST_max.mean())
    )
    return speed_score


def get_compound_score(lap_data: pd.DataFrame):
    # Measure mean tire life relative to field
    score = 1 / (lap_data.Compound_mean / lap_data.Compound_mean.mean())
    return score


def get_tire_freshness_score(lap_data: pd.DataFrame):
    # Measure mean tire freshness relative to field, older is better
    score = lap_data.FreshTyre_mean / lap_data.FreshTyre_mean.mean()
    return score


def get_session_scores(session):
    lap_data = get_lap_data(session)
    min_time_score = normalize(get_min_time_score(lap_data))
    mean_time_score = normalize(get_mean_time_score(lap_data))
    max_time_score = normalize(get_max_time_score(lap_data))
    lap_consistency_score = normalize(get_lap_consistency_score(lap_data))
    top_speed_score = normalize(get_top_speed_score(lap_data))
    sector_1_consistency_score = normalize(get_sector_1_consistency_score(lap_data))
    sector_2_consistency_score = normalize(get_sector_2_consistency_score(lap_data))
    sector_3_consistency_score = normalize(get_sector_3_consistency_score(lap_data))
    sector_1_time_score = normalize(get_sector_1_time_score(lap_data))
    sector_2_time_score = normalize(get_sector_2_time_score(lap_data))
    sector_3_time_score = normalize(get_sector_3_time_score(lap_data))
    sector_1_mean_time_score = normalize(get_sector_1_mean_time_score(lap_data))
    sector_2_mean_time_score = normalize(get_sector_2_mean_time_score(lap_data))
    sector_3_mean_time_score = normalize(get_sector_3_mean_time_score(lap_data))
    sector_1_max_time_score = normalize(get_sector_1_max_time_score(lap_data))
    sector_2_max_time_score = normalize(get_sector_2_max_time_score(lap_data))
    sector_3_max_time_score = normalize(get_sector_3_max_time_score(lap_data))
    total_score = (
        lap_consistency_score / 2
        + min_time_score * 2
        + max_time_score / 2
        + mean_time_score
        + sector_1_consistency_score / 2
        + sector_2_consistency_score / 2
        + sector_3_consistency_score / 2
        + sector_1_time_score * 2
        + sector_2_time_score * 2
        + sector_3_time_score * 2
        + sector_1_mean_time_score
        + sector_2_mean_time_score
        + sector_3_mean_time_score
        + sector_1_max_time_score / 2
        + sector_2_max_time_score / 2
        + sector_3_max_time_score / 2
        + top_speed_score
    )
    score_df = pd.DataFrame(
        total_score.sort_values(ascending=False), columns=["Score"],
    )
    score_df["DriverName"] = score_df.index.to_series().map(DRIVER_NUMBER_TO_DRIVER)
    return score_df


def get_combined_free_practice_scores(year: int, gp: str):
    training_sessions = ["FP1", "FP2"]
    session_scores_list = []
    for training_session in training_sessions:
        session = fastf1.get_session(year, gp, training_session)
        session.load()
        session_scores = get_session_scores(session=session)
        session_scores_list.append(session_scores)

    combined_session_scores = normalize(
        pd.concat(session_scores_list, axis=0)
        .groupby(["DriverName"])
        .sum()
        .sort_values("Score", ascending=False)
    )
    print(combined_session_scores)


if __name__ == "__main__":
    get_combined_free_practice_scores(2022, args.gp)
    # session = fastf1.get_session(2022, args.gp, "R")
    # session.load()
    # session_scores = get_session_scores(session=session)
    # print(session_scores)

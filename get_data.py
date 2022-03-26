import numpy as np
import pandas as pd
from constants import LAP_DATA_CATEGORIES, TIME_DATA


def get_lap_data(session):
    lap_df = session.laps
    accurate_laps = lap_df[lap_df["IsAccurate"] == True]
    lap_data = accurate_laps[LAP_DATA_CATEGORIES]
    lap_data = lap_data.dropna()
    lap_data[TIME_DATA] = lap_data[TIME_DATA] / np.timedelta64(1, "s")
    lap_data[["Compound", "FreshTyre"]] = lap_data[["Compound", "FreshTyre"]].apply(
        lambda col: pd.Categorical(col).codes
    )
    mean_lap_data = lap_data.groupby(["DriverNumber"]).mean().add_suffix("_mean")
    max_lap_data = lap_data.groupby(["DriverNumber"]).max().add_suffix("_max")
    min_lap_data = lap_data.groupby(["DriverNumber"]).min().add_suffix("_min")
    std_lap_data = lap_data.groupby(["DriverNumber"]).std().add_suffix("_std")
    lap_data = pd.concat(
        [mean_lap_data, max_lap_data, min_lap_data, std_lap_data], axis=1
    )
    return lap_data

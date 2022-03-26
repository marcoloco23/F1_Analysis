import fastf1
import numpy as np
import pandas as pd
import xgboost as xgb
from constants import GP_LIST, TRAINING_SESSIONS
from get_data import get_lap_data
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing

fastf1.Cache.enable_cache("/Users/marcsperzel/Desktop/F1")


def normalize(df):
    x = df.values
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    x_scaled = scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns=df.columns)
    return df


def train_ai(data: pd.DataFrame):
    training_data = data.sample(frac=0.8, random_state=25)
    testing_data = data.drop(training_data.index)
    y_train = training_data["Points"]
    x_train = training_data.drop(["Points"], axis=1)
    y_test = testing_data["Points"]
    x_test = testing_data.drop(["Points"], axis=1)
    eval_set = [(x_test, y_test)]
    params = {
        "booster": "gbtree",
        "objective": "reg:squarederror",
        "learning_rate": 0.001,
        "n_estimators": 10000,
        "eval_metric": "rmse",
    }
    model = xgb.XGBRegressor(**params)

    model.fit(
        x_train,
        y_train,
        verbose=True,
        eval_metric="rmse",
        early_stopping_rounds=20,
        eval_set=eval_set,
    )
    y_pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print("RMSE: %f\n" % (rmse))
    print("R^2: %f\n" % (r2))
    return model


def prepare_dataset():
    data_list = []
    for gp in GP_LIST:
        try:
            race_session = fastf1.get_session(2021, gp, "R")
            race_session.load()
            practice_data_list = []
            for training_session in TRAINING_SESSIONS:
                session = fastf1.get_session(2021, gp, training_session)
                session.load()
                lap_data = get_lap_data(session)
                normalized_lap_data = normalize(lap_data).set_index(lap_data.index)
                practice_data_list.append(normalized_lap_data)

            combined_gp_data = (
                pd.concat(practice_data_list, axis=0).groupby(["DriverNumber"]).mean()
            )
            combined_gp_data["Points"] = race_session.results.Points
            data_list.append(combined_gp_data)
        except Exception as e:
            print(e)

    combined_data = pd.concat(data_list).reset_index(drop=True).fillna(0)
    combined_data.to_csv("data.csv", index=False)


if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    model = train_ai(data)
    model.save_model("f1_model.json")

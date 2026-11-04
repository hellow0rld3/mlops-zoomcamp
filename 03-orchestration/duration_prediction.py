import pickle
import mlflow
import argparse
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from pathlib import Path
from sklearn.metrics import root_mean_squared_error

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

def read_parquet(year, month):

    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']
    df['duration'] = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[df.duration >= 1]
    df = df[df.duration <= 60]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    return df


def create_X(df, dv=None):
    categorical = ['PU_DO'] 
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv

def fix_numpy(X_train, X_eval):
    X_train.indices = X_train.indices.astype(np.int32)
    X_train.indptr = X_train.indptr.astype(np.int32)

    X_eval.indices = X_eval.indices.astype(np.int32)
    X_eval.indptr = X_eval.indptr.astype(np.int32)



def train_model(X_train, y_train, X_eval, y_eval, dv):
    with mlflow.start_run() as run:

        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_eval, label=y_eval)

        best_params = {
            "learning_rate" : 0.2948646839243271,
            "max_depth" : 30,
            "min_child_weight" : 1.9805346649872448,
            "objective" : "reg:squarederror",
            "reg_alpha" : 0.06275960619157453,
            "reg_lambda" : 0.22388496105060088,
            "seed" : 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, "validation")],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_eval, y_pred)
        mlflow.log_metric("RMSE", rmse)

        with open("../models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("../models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        return run.info.run_id


def run(year, month):

    df_train = read_parquet(year=year, month=month)

    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1

    df_eval = read_parquet(year=year, month=month + 1)

    X_train, dv = create_X(df=df_train)
    X_eval, _ = create_X(df=df_eval, dv=dv)

    fix_numpy(X_train=X_train, X_eval=X_eval)

    target = 'duration'
    y_train = df_train[target].values
    y_eval = df_eval[target].values

    run_id = train_model(X_train=X_train, y_train=y_train, X_eval=X_eval, y_eval=y_eval, dv=dv)
    print(f"MLflow run_id: {run_id}")
    return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model to predict taxi trip duration")
    parser.add_argument("--year", type=int, required=True, help="Year of the data to train on")
    parser.add_argument("--month", type=int, required=True, help="Month of the data to train on")
    args = parser.parse_args()

    run_id = run(year=args.year, month=args.month)
    with open("run_id.txt", "w") as f:
        f.write(run_id)
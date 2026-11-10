# Q1 and Q2 answer:
# Airflow orchestrator, version 3.4.0

import pandas as pd
import numpy as np
import mlflow

from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("nyc-taxi-experiment")

MARCH_YELLOW_TAXI_LINK = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet'

'''
df = pd.read_parquet(MARCH_YELLOW_TAXI_LINK)

print(df.shape[0])

#Q3 answer:
# 3,403,766

'''
def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df


'''
print(read_dataframe(MARCH_YELLOW_TAXI_LINK).shape[0])


#Q4 answer:
# 3,316,216
'''

def create_X(df, dv=None):
    categorical=['PULocationID','DOLocationID']
    dicts = df[categorical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv

def fix_numpy(X_train, X_eval):
    X_train.indices = X_train.indices.astype(np.int32)
    X_train.indptr = X_train.indptr.astype(np.int32)

    X_eval.indices = X_eval.indices.astype(np.int32)
    X_eval.indptr = X_eval.indptr.astype(np.int32)

def train_model(X_train, y_train, X_eval, y_eval):
    with mlflow.start_run():
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_eval)
        mlflow.log_metric("RMSE", root_mean_squared_error(y_eval, y_pred))
        mlflow.log_metric("Intercept", lr.intercept_)
        model_info = mlflow.sklearn.log_model(lr, name="model")
        mlflow.register_model(
            model_uri=model_info.model_uri,
            name="nyc-taxi-lr",
        )

def run(link):
    df = read_dataframe(link)

    X, dv = create_X(df)

    target = 'duration'
    y_train = df[target].values
    y_eval = df[target].values

    train_model(X_train=X, y_train=y_train, X_eval=X, y_eval=y_eval)

    fix_numpy(X_train=X, X_eval=X)

if __name__ == "__main__":
    run(MARCH_YELLOW_TAXI_LINK)
    
#Q5 answer: (from mlflow ui)
# 24,77

#Q6 answer: (from mlflow ui)
# 8521? WRONG
import os
from datetime import datetime, timedelta

from pymongo import MongoClient
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split

from app.backend.data_processing import DataPreprocessor
from sklearn.metrics import mean_squared_error

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from mlflow.data.pandas_dataset import PandasDataset

import mlflow

mlflow.set_tracking_uri("http://localhost:8090")
mlflow.set_experiment("rightmove-rent-prediction")


if os.environ.get("staging"):
    MONGO_URI = "mongodb://mongodb:27017/"
else:
    MONGO_URI = "mongodb://localhost:27017/"

def load_data_from_mongo(collection_name="properties", fields=None):

    client = MongoClient(MONGO_URI)  # Hosted with Docker

    db = client["rightmove"]

    collection = db[collection_name]

    query = {}

    data = collection.find(query, fields)

    df = pd.DataFrame(list(data))

    return df

def create_dataset(train_df, val_df, features):
    X_train = train_df[features]
    y_train = train_df[['price']]

    X_val = val_df[features]
    y_val = val_df[['price']]

    train_dataset: PandasDataset = mlflow.data.from_pandas(train_df[features])
    val_dataset: PandasDataset = mlflow.data.from_pandas(val_df[features])

    return X_train, y_train, X_val, y_val, train_dataset, val_dataset

def train_best_model(X_train, y_train, X_val, y_val, train_dataset, val_dataset) -> None:
    """train a model with best hyperparams and write everything out"""

    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(val_dataset, context="validation")

        best_params = {
            "learning_rate": 0.09585355369315604,
            "max_depth": 30,
            "min_child_weight": 1.060597050922164,
            "objective": "reg:linear",
            "reg_alpha": 0.018060244040060163,
            "reg_lambda": 0.011658731377413597,
            "seed": 42,
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, "validation")],
            early_stopping_rounds=20,
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

    return booster

def train_model():
    preprocessor = DataPreprocessor()

    property_fields = {"id": 1, "bedrooms": 1, "bathrooms":1, "location":1, "price":1,
         "feature_list": 1, "commercial": 1, "development": 1, "students": 1,
         "summary": 1}

    walk_score_fields = {"id": 1, "scores": 1}

    properties_df = load_data_from_mongo(collection_name="properties", fields=property_fields)
    walk_score_df = load_data_from_mongo(collection_name="walk_scores", fields=walk_score_fields)

    processed_properties = preprocessor.preprocess_properties(properties_df)
    processed_walk_scores = preprocessor.preprocess_walk_score(walk_score_df)

    full_df = processed_properties.merge(processed_walk_scores[['id', 'walk_score']], on='id')
    full_df = full_df.drop(columns=['id', '_id'])

    train_val, test_df = train_test_split(full_df, test_size=0.1, random_state=42)  # 10% for test set
    train_df, val_df = train_test_split(train_val, test_size=0.2, random_state=42)

    numerical_features_walk_score = ['bedrooms', 'bathrooms', 'longitude', 'latitude', 'walk_score']
    X_train, y_train, X_val, y_val, train_dataset, val_dataset = create_dataset(train_df, val_df, numerical_features_walk_score)

    booster = train_best_model(X_train, y_train, X_val, y_val)

    X_test = test_df[numerical_features_walk_score]
    y_test = test_df['price']

    valid = xgb.DMatrix(X_test, label=y_test)

    y_pred = booster.predict(valid)

    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f"Achieved a RMSE of {rmse} on test set")

    return None

def preprocess_data(input_data):
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess(input_data)
    return processed_data

default_args = {
    'owner': 'airflow_app',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'scrape_rightmove',
    default_args=default_args,
    description='DAG for training new model',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
    max_active_runs=1
)

start_task = DummyOperator(
    task_id='start',
    dag=dag
)

train_model_task = PythonOperator(
    task_id='train_model_task',
    python_callable=train_model,
    dag=dag
)

end_task = DummyOperator(
    task_id='end',
    dag=dag
)

start_task >> train_model_task >> end_task
import os
from datetime import datetime, timedelta

from pymongo import MongoClient
from google.cloud import storage
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from mlflow.data.pandas_dataset import PandasDataset
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator

from rightmove.data_processing.data_processor import DataPreprocessor

from sklearn.ensemble import RandomForestRegressor

from dotenv import load_dotenv

load_dotenv("/Users/alexander.girardet/Code/Personal/projects/rightmove_project/.env")

import mlflow

import logging

logging.basicConfig(level=logging.INFO)

MONGO_URI = os.environ.get("MONGO_URI")

mlflow.set_tracking_uri(
    "postgresql+psycopg2://postgres:postgres@realestate-database.czkkjkojmucd.eu-west-2.rds.amazonaws.com:5432/mlflow"
)
experiment_name = "rightmove-prediction"
mlflow.set_experiment(experiment_name)

client = storage.Client()
bucket = client.get_bucket("rightmove-artifacts-ml")

def load_data_from_mongo(collection_name="properties", fields=None):
    logging.info("Loading data from mongo")

    client = MongoClient(MONGO_URI)  # Hosted with Docker

    db = client["rightmove"]

    collection = db[collection_name]

    query = {}

    data = collection.find(query, fields)

    df = pd.DataFrame(list(data))

    if len(df) == 0:
        raise ValueError(f"No data found in collection {collection_name}")
    else:
        logging.info(f"Data loaded from collection {collection_name}")

    return df

def generate_foldername():
    now = datetime.now()
    return now.strftime("%Y-%m-%d-%H-%M-%S")


def load_df_to_gcs(df, dest_path):
    blob = bucket.blob(dest_path)
    try:
        blob.upload_from_string(df.to_csv(), "text/csv")
        logging.info(f"Data uploaded to {dest_path}")
        return True
    except Exception as e:
        print(e)


def preprocess_data(property_df, walkscore_df):
    preprocessor = DataPreprocessor(with_text=False, with_binary=False)

    property_df = preprocessor.preprocess_properties(property_df)
    walk_df = preprocessor.preprocess_walk_score(walkscore_df)

    df = property_df.merge(walk_df, on="id", how="left")

    logging.info("Data preprocessed")

    return df


def load_data_from_gcs(source_url):
    logging.info(f"Loading {source_url} from GCS")
    df = pd.read_csv(source_url, index_col=0)
    return df


def fetch_preprocess_and_upload_data():
    property_df = load_data_from_mongo(
        collection_name="properties",
        fields={
            "id": 1,
            "price.amount": 1,
            "price.frequency": 1,
            "firstVisibleDate": 1,
            "bedrooms": 1,
            "bathrooms": 1,
            "listingUpdate": 1,
            "location": 1,
        },
    )
    walkscore_df = load_data_from_mongo(
        collection_name="walk_scores", fields={"id": 1, "scores": 1}
    )

    df = preprocess_data(property_df, walkscore_df)

    df = df[["bedrooms", "bathrooms", "price", "longitude", "latitude", "walk_score"]]

    folder_name = generate_foldername()
    parent_folder = "data"

    df["price_bin"] = pd.qcut(df["price"], q=10, duplicates="drop")

    # Create train test,  validation split
    train_val, test_df = train_test_split(
        df, test_size=0.1, stratify=df["price_bin"], random_state=42
    )
    train_df, val_df = train_test_split(
        train_val, test_size=0.2, stratify=train_val["price_bin"], random_state=42
    )

    # Upload to GCS train, test, and validation data
    load_df_to_gcs(train_df, f"{parent_folder}/{folder_name}/train.csv")
    load_df_to_gcs(val_df, f"{parent_folder}/{folder_name}/val.csv")
    load_df_to_gcs(test_df, f"{parent_folder}/{folder_name}/test.csv")

    logging.info("Data uploaded to GCS")

    return folder_name


def train_model(**kwargs):
    if "ti" in kwargs:
        ti = kwargs["ti"]
        folder_name = ti.xcom_pull(task_ids="load_data")
    else:
        folder_name = kwargs["folder_name"]

    logging.info(f"Training model with data from {folder_name}")

    train_dataset_source_url = (
        f"gs://rightmove-artifacts-ml/data/{folder_name}/train.csv"
    )
    val_dataset_source_url = f"gs://rightmove-artifacts-ml/data/{folder_name}/val.csv"
    test_dataset_source_url = f"gs://rightmove-artifacts-ml/data/{folder_name}/test.csv"

    train_df = load_data_from_gcs(train_dataset_source_url)
    val_df = load_data_from_gcs(val_dataset_source_url)
    test_df = load_data_from_gcs(test_dataset_source_url)

    train_df = train_df.dropna()
    val_df = val_df.dropna()

    features = ["bedrooms", "bathrooms", "longitude", "latitude", "walk_score"]
    target = "price"

    X_train = train_df[features]
    y_train = train_df[target]

    X_val = val_df[features]
    y_val = val_df[target]

    train_dataset: PandasDataset = mlflow.data.from_pandas(
        train_df, source=train_dataset_source_url
    )
    val_dataset: PandasDataset = mlflow.data.from_pandas(
        val_df, source=val_dataset_source_url
    )
    test_dataset: PandasDataset = mlflow.data.from_pandas(
        test_df, source=val_dataset_source_url
    )

    with mlflow.start_run() as run:
        mlflow.set_tag("developer", "Alex")

        mlflow.log_param("Model type", "Random Forest")
        model = RandomForestRegressor()

        # Log the datasets
        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(val_dataset, context="validation")
        mlflow.log_input(test_dataset, context="test")

        logging.info("Fitting model")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)

        r2 = model.score(X_val, y_val)

        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, "random-forest")

        logging.info("Model trained and logged to MLflow")

    return run.info.run_id


def register_model(**kwargs):
    if "ti" in kwargs:
        ti = kwargs["ti"]
        run_id = ti.xcom_pull(task_ids="train_model")
    else:
        run_id = kwargs["run_id"]

    model_name = "Random Forest Walk Score"
    artifact_path = "random-forest"

    model_uri = f"runs:/{run_id}/{artifact_path}"

    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
    logging.info(
        f"Model registered with name: {model_name} and version: {model_details.version}"
    )

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_details.version,
        stage="Staging",
        archive_existing_versions=False,
    )
    logging.info(f"Model version {model_details.version} transitioned to Staging")


default_args = {
    "owner": "airflow_app",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "train_model",
    default_args=default_args,
    description="DAG for making scraping rightmove",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
    max_active_runs=1,
)
with dag:
    load_and_preprocess_data_task = PythonOperator(
        task_id="load_data", python_callable=fetch_preprocess_and_upload_data
    )

    train_model_task = PythonOperator(
        task_id="train_model", python_callable=train_model, provide_context=True
    )

    register_model_task = PythonOperator(
        task_id="register_model", python_callable=register_model, provide_context=True
    )

    load_and_preprocess_data_task >> train_model_task >> register_model_task

if __name__ == "__main__":
    folder_name = fetch_preprocess_and_upload_data()
    print(folder_name)
    run_id = train_model(folder_name=folder_name)
    print(run_id)
    register_model(run_id=run_id)

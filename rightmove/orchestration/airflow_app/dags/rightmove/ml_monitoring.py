mfrom airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator

from rightmove.data_processing.data_processor import DataPreprocessor
from rightmove.data_processing.metric_extraction import MetricExtraction

import re


from pymongo import MongoClient
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from google.cloud import storage
import os
import random
import requests
import logging

import mlflow

client = storage.Client()
bucket = client.get_bucket("rightmove-artifacts-ml")

MONITORING_URI_PG = os.environ.get("MONITORING_URI_PG")

mlflow.set_tracking_uri(MONITORING_URI_PG)

experiment_name = "rightmove-prediction"
mlflow.set_experiment(experiment_name)

ML_SERVING_URL = "http://fastapi_app:8000/batch-predict"

logging.basicConfig(level=logging.INFO)

# load_dotenv("/Users/alexander.girardet/Code/Personal/projects/rightmove_project/.env")
MONGO_URI = os.environ.get("MONGO_URI")

default_args = {
    "owner": "airflow_app",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def modify_uri_to_test(uri: str) -> str:
    parts = uri.split('/')
    filename = parts[-1]
    new_filename = re.sub(r'(train|val|test)\.csv', 'test.csv', filename)
    parts[-1] = new_filename
    new_uri = '/'.join(parts)
    return new_uri



def fetch_reference_df():
    response = requests.get("http://fastapi_app:8000/latest-dataset")
    latest_uri = response.json().get("uri")

    test_uri = modify_uri_to_test(latest_uri)


    reference_data = pd.read_csv(
        test_uri, index_col=0
    )
    return reference_data


def preprocess_data(property_df, walkscore_df):
    preprocessor = DataPreprocessor(with_text=False, with_binary=False)

    property_df = preprocessor.preprocess_properties(property_df)
    walk_df = preprocessor.preprocess_walk_score(walkscore_df)

    df = property_df.merge(walk_df, on="id", how="left")

    logging.info("Data preprocessed")

    return df


def load_data_from_mongo(collection_name, fields, timestamp_field):
    client = MongoClient(MONGO_URI)

    db = client["rightmove"]

    collection = db[collection_name]

    two_hours_ago = datetime.now() - timedelta(hours=12)
    two_hours_ago_unix = two_hours_ago.timestamp()

    query = {timestamp_field: {"$gt": two_hours_ago_unix}}

    data = collection.find(query, fields)

    return pd.DataFrame(list(data))


def fetch_latest_batch():
    property_fields = {
        "id": 1,
        "bedrooms": 1,
        "bathrooms": 1,
        "location": 1,
        "price": 1,
        "listingUpdate": 1,
        "firstVisibleDate": 1,
    }
    property_df = load_data_from_mongo(
        "properties", property_fields, "extraction_timestamp"
    )

    walk_score_fields = {"id": 1, "scores": 1}
    walk_score_df = load_data_from_mongo(
        "walk_scores", walk_score_fields, "processing_timestamp"
    )

    df = preprocess_data(property_df, walk_score_df)

    df = df[["bedrooms", "bathrooms", "price", "longitude", "latitude", "walk_score"]]

    return df


def load_predictions_from_gcs(folder_name):
    current_data = pd.read_csv(
        f"gs://rightmove-artifacts-ml/predictions/{folder_name}/current.csv",
        index_col=0,
    )
    reference_data = pd.read_csv(
        f"gs://rightmove-artifacts-ml/predictions/{folder_name}/reference.csv",
        index_col=0,
    )

    return current_data, reference_data


def monitor_datasets(**kwargs):
    synthetic_data = False

    if "ti" in kwargs:
        ti = kwargs["ti"]
        folder_name = ti.xcom_pull(task_ids="load_predictions_to_gcs")
    else:
        folder_name = kwargs.get("folder_name")
        synthetic_data = kwargs.get("synthetic_data")

    current_data, reference_data = load_predictions_from_gcs(folder_name)

    current_data = current_data[['bedrooms', 'bathrooms', 'longitude', 'latitude', 'walk_score', 'prediction', 'target']]
    reference_data = reference_data[['bedrooms', 'bathrooms', 'longitude', 'latitude', 'walk_score', 'prediction', 'target']]

    metric_extractor = MetricExtraction()

    metric_extractor.connect_to_postgres()

    performance_data = metric_extractor.get_performance_metrics(
        current_data, reference_data
    )

    prediction_data = metric_extractor.get_target_drift_metrics(
        current_data, reference_data
    )

    drift_data = metric_extractor.get_data_drift_metrics(current_data, reference_data)

    quality_data = metric_extractor.get_data_quality_metrics(
        current_data, reference_data
    )

    if synthetic_data:
        fake_timestamp = datetime.now() - timedelta(days=random.randint(0, 30))
    else:
        fake_timestamp = None

    metric_extractor.load_metrics_to_postgres(
        prediction_data, "prediction_drift", loading_timestamp=fake_timestamp
    )
    metric_extractor.load_metrics_to_postgres(
        performance_data, "performance", loading_timestamp=fake_timestamp
    )
    metric_extractor.load_metrics_to_postgres(
        drift_data, "drift", loading_timestamp=fake_timestamp
    )
    metric_extractor.load_metrics_to_postgres(
        quality_data, "quality", loading_timestamp=fake_timestamp
    )
    logging.info("Metrics loaded to Postgres")

    metric_extractor.close_connection()
    logging.info("Connection to Postgres closed")


def predict_properties(properties_features):
    try:
        response = requests.post(ML_SERVING_URL, json=properties_features)
        if response.status_code != 200:
            raise ValueError("Request failed")
        else:
            predictions = response.json().get("predictions")
            return predictions
    except Exception as e:
        raise e


def generate_predictions(current_data=None, reference_data=None):
    if current_data is None:
        raise ValueError("No current data set")

    # new_logged_model = 'runs:/5c5b195cf1b74219993b436489545b7a/random-forest' # Replace with latest model from API
    # new_logged_model = mlflow.pyfunc.load_model(new_logged_model)

    current_features = current_data[
        ["bedrooms", "bathrooms", "longitude", "latitude", "walk_score"]
    ].to_dict("records")
    reference_features = reference_data[
        ["bedrooms", "bathrooms", "longitude", "latitude", "walk_score"]
    ].to_dict("records")

    current_predictions = predict_properties(current_features)
    reference_predictions = predict_properties(reference_features)

    # reference_data['predictions'] = new_logged_model.predict(reference_data.drop(columns=['price']))
    #
    # current_data['predictions'] = new_logged_model.predict(current_data.drop(columns=['price']))

    current_data["predictions"] = current_predictions

    reference_data["predictions"] = reference_predictions

    logging.info("Predictions generated")

    return current_data, reference_data


def load_df_to_gcs(df, dest_path):
    blob = bucket.blob(dest_path)
    try:
        blob.upload_from_string(df.to_csv(), "text/csv")
        logging.info(f"Data uploaded to {dest_path}")
        return True
    except Exception as e:
        print(e)


def load_data_from_gcs(source_url):
    logging.info(f"Loading {source_url} from GCS")
    df = pd.read_csv(source_url, index_col=0)
    return df


def generate_foldername():
    now = datetime.now()
    return now.strftime("%Y-%m-%d-%H-%M-%S")


def load_predictions_to_gcs():
    logging.info("Fetching data")
    current_data = fetch_latest_batch()
    reference_data = fetch_reference_df()

    logging.info("Generating predictions")
    current_data, reference_data = generate_predictions(
        current_data=current_data, reference_data=reference_data
    )

    current_data = current_data.rename(
        columns={"predictions": "prediction", "price": "target"}
    )
    reference_data = reference_data.rename(
        columns={"predictions": "prediction", "price": "target"}
    )

    folder_name = generate_foldername()
    parent_folder = "predictions"

    load_df_to_gcs(current_data, f"{parent_folder}/{folder_name}/current.csv")
    load_df_to_gcs(reference_data, f"{parent_folder}/{folder_name}/reference.csv")

    logging.info("Data loaded to GCS")

    return folder_name


dag = DAG(
    "monitor_ml_performance_rightmove",
    default_args=default_args,
    description="DAG for monitoring ML performance for rightmove",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
    max_active_runs=1,
)

start_task = DummyOperator(task_id="start", dag=dag)

load_predictions_to_gcs_task = PythonOperator(
    task_id="load_predictions_to_gcs", python_callable=load_predictions_to_gcs, dag=dag
)

monitor_datasets_task = PythonOperator(
    task_id="monitor_datasets",
    python_callable=monitor_datasets,
    provide_context=True,
    dag=dag,
)

end_task = DummyOperator(task_id="end", dag=dag)

start_task >> load_predictions_to_gcs_task >> monitor_datasets_task >> end_task

if __name__ == "__main__":
    # folder_name = load_predictions_to_gcs()
    # monitor_datasets(folder_name=folder_name)

    response = requests.get("http://localhost:8000/latest-dataset")
    latest_uri = response.json().get("uri")

    test_uri = modify_uri_to_test(latest_uri)

    print(test_uri)
    #
    # import logging
    # import pandas as pd

    # def split_df_into_chunks(df, chunk_size=500):
    #     """Yield successive chunks of rows from df."""
    #     for i in range(0, df.shape[0], chunk_size):
    #         yield df.iloc[i:i + chunk_size]
    #
    #
    # def load_chunk_to_gcs(chunk, parent_folder, folder_name, base_filename, chunk_index):
    #     """Load a single chunk of DataFrame to GCS, with a unique filename."""
    #     filename = f"{base_filename}_part{chunk_index}.csv"
    #     path = f"{parent_folder}/{folder_name}/{filename}"
    #     # This function should be defined to handle the actual loading process to GCS
    #     load_df_to_gcs(chunk, path)
    #
    #
    # logging.info("Fetching data")
    # current_data = fetch_latest_batch()
    # reference_data = fetch_reference_df()
    #
    # logging.info("Generating predictions")
    # current_data, reference_data = generate_predictions(current_data=current_data, reference_data=reference_data)
    #
    # current_data = current_data.rename(columns={"predictions": "prediction", "price": "target"})
    # reference_data = reference_data.rename(columns={"predictions": "prediction", "price": "target"})
    #
    # folder_name = generate_foldername()
    # parent_folder = "predictions"
    #
    # # Split and load current_data
    # for index, chunk in enumerate(split_df_into_chunks(current_data)):
    #     monitor_datasets(chunk, reference_data)
    #     # load_chunk_to_gcs(chunk, parent_folder, folder_name, "current", index)

    logging.info("Data loaded to GCS")

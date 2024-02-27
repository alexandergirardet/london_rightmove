from datetime import datetime, timedelta
import pandas as pd
from pymongo import MongoClient
import os
from google.cloud import storage
import logging
from io import BytesIO

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from rightmove.data_processing.data_processor import DataPreprocessor

logging.basicConfig(level=logging.INFO)

client = storage.Client()
bucket = client.get_bucket("rightmove-artifacts-ml")

MONGO_URI = os.environ.get("MONGO_URI")
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


def load_df_to_gcs_parquet(df, dest_path):
    # Create an in-memory bytes buffer
    buffer = BytesIO()

    try:
        # Save the dataframe to the buffer in parquet format
        df.to_parquet(buffer, index=False)

        # Move the buffer's pointer to the beginning of the file
        buffer.seek(0)

        # Create a blob in the specified GCS bucket path
        blob = bucket.blob(dest_path)

        # Upload the buffer content as a parquet file
        blob.upload_from_file(buffer, content_type='application/octet-stream')

        logging.info(f"Data uploaded to {dest_path} in Parquet format")
        return True
    except Exception as e:
        logging.error(f"Failed to upload data to {dest_path}: {e}")
        return False

def preprocess_data(property_df, walkscore_df):
    preprocessor = DataPreprocessor(with_text=True, with_binary=False)

    property_df = preprocessor.preprocess_properties(property_df)
    walk_df = preprocessor.preprocess_walk_score(walkscore_df)

    df = property_df.merge(walk_df, on="id", how="left")

    logging.info("Data preprocessed")

    return df

def fetch_preprocess_data():
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
            "summary": 1,
            "feature_list": 1,
        },
    )
    walkscore_df = load_data_from_mongo(
        collection_name="walk_scores", fields={"id": 1, "scores": 1}
    )

    df = preprocess_data(property_df, walkscore_df)

    dest_path = f"streamlit_data/{generate_foldername()}/data.parquet"
    load_df_to_gcs_parquet(df, dest_path)

    logging.info(f"Data saved to {dest_path}")
    
    return df
def load_data():
    df = fetch_preprocess_data()
    logging.info("Data loaded")

default_args = {
    "owner": "airflow_app",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "streamlit_data_extraction",
    default_args=default_args,
    description="DAG for extracting data for Streamlit app",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
    max_active_runs=1,
)

start_task = DummyOperator(task_id="start", dag=dag)

load_data_task = PythonOperator(
    task_id="load_data", python_callable=load_data, dag=dag
)

end_task = DummyOperator(task_id="end", dag=dag)

start_task>> load_data_task >> end_task

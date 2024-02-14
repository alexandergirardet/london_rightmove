import os
from datetime import datetime, timedelta

from pymongo import MongoClient
from google.cloud import storage
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from data_processing.DataPreprocessor import DataPreprocessor
from mlflow.data.pandas_dataset import PandasDataset

import mlflow

mlflow.set_tracking_uri("postgresql+psycopg2://airflow:airflow@localhost:5432/mlflow")
mlflow.set_experiment("rightmove-rent-prediction")

client = storage.Client()
bucket = client.get_bucket('rightmove-ml-artifacts')

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

# Function to generate a filename based on date and time
def generate_filename():
    now = datetime.now()
    return now.strftime("%Y-%m-%d-%H-%M-%S")

def load_df_to_gcs(df, dest_path):
    blob = bucket.blob(dest_path)
    try:
        blob.upload_from_string(df.to_csv(), 'text/csv')
    except Exception as e:
        print(e)

def preprocess_data():
    preprocessor = DataPreprocessor(with_text=False, with_binary=False)

    property_df = load_data_from_mongo(collection_name='properties', fields={"id": 1, "price.amount": 1, "price.frequency": 1, "firstVisibleDate": 1, "bedrooms": 1, "bathrooms": 1, "listingUpdate": 1, 'location': 1})
    walkscore_df = load_data_from_mongo(collection_name='walk_scores', fields={"id": 1, "scores": 1})

    property_df = preprocessor.preprocess_properties(property_df)
    walk_df = preprocessor.preprocess_walk_score(walkscore_df)

    df = property_df.merge(walk_df, on='id', how='left')

    # features = ['bedrooms', 'bathrooms', 'longitude', 'latitude', 'walk_score']
    #
    # # train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    # #
    # # X_train, y_train, X_val, y_val, train_dataset, val_dataset = create_dataset(train_df, val_df, features)

    return df

def fetch_preprocess_and_upload_data():
    df = preprocess_data()
    filename = generate_filename()
    folder = "data"
    file_path = f"{folder}/{filename}.csv"
    load_df_to_gcs(df, file_path)


MONGO_URI = "mongodb://localhost:27017/"

default_args = {
    'owner': 'airflow_app',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# dag = DAG(
#     'train_model',
#     default_args=default_args,
#     description='DAG for making scraping rightmove',
#     schedule_interval=timedelta(days=1),
#     start_date=datetime(2023, 1, 1),
#     catchup=False,
#     max_active_runs=1
# )
#
# start_task = DummyOperator(
#     task_id='start',
#     dag=dag
# )
#
# start_spider_task = PythonOperator(
#     task_id='start_spider',
#     python_callable=start_spider,
#     dag=dag
# )
#
# periodic_requests = PythonOperator(
#     task_id='periodic_requests',
#     python_callable=repeated_requests,
#     provide_context=True,
#     dag=dag
# )
#
# cancel_spider_task = PythonOperator(
#     task_id='cancel_spider',
#     python_callable=cancel_spider,
#     provide_context=True,
#     dag=dag
# )
#
# end_task = DummyOperator(
#     task_id='end',
#     dag=dag
# )

# start_task >> start_spider_task >> periodic_requests >> cancel_spider_task >> end_task

if __name__ == "__main__":
    fetch_preprocess_and_upload_data()
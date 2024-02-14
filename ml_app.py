import os
from http.client import HTTPException
from typing import Union

from fastapi import FastAPI

from pydantic import BaseModel

# from scipy.stats import zscore

import mlflow

import pandas as pd

from pymongo import MongoClient

app = FastAPI()

import mlflow

mlflow.set_tracking_uri("postgresql+psycopg2://airflow:airflow@localhost:5432/mlflow")
experiment_name = "rightmove-rent-prediction"
mlflow.set_experiment(experiment_name)

# MONGO_DB_URL = "mongodb://mongodb:27017/"

MONGO_DB_URL = os.environ.get('MONGO_DB_URL')

model_name = "Random Forest Regression"
model_uri = f"models:/{model_name}/Staging"

# Load the model as a PyFunc model
model = mlflow.pyfunc.load_model(model_uri)

class Property(BaseModel):
    bedrooms: float
    bathrooms: float
    longitude: float
    latitude: float
    walk_score: float
@app.post("/predict")
def predict_rent(input_property: Property):
    try:
        # Make a prediction
        features_df = pd.DataFrame(input_property.dict(), index=[0])
        prediction = model.predict(features_df)
        return {"prediction": prediction[0]}

    except Exception as e:
        raise HTTPException()

def connect_to_client():
    client = MongoClient(MONGO_DB_URL)
    return client

def connect_to_db(client, db):
    db = client[db]
    return db

def connect_to_collection(db, collection):
    collection = db[collection]
    return collection

def get_df():
    client = connect_to_client()

    db = connect_to_db(client, "rightmove")

    collection = connect_to_collection(db, "properties")

    fields = {"price.amount": 1, "price.frequency": 1, "firstVisibleDate": 1, "bedrooms": 1, "bathrooms": 1, "listingUpdate": 1, 'location': 1}
    query = {}

    # Fetch data from the collection
    data = collection.find(query, fields)

    # Convert to Pandas DataFrame
    df = pd.DataFrame(list(data))

    df['latitude'] = df['location'].apply(lambda x: x['latitude'])
    df['longitude'] = df['location'].apply(lambda x: x['longitude'])

    client.close()

    return df

def get_property(collection, propery_id):
    fields = {"id": propery_id}
    query = {}
    data = collection.find(query, fields)

    return data

def convert_frequencies(x):
    frequency = x['frequency']
    price = x['amount']

    if frequency == 'monthly':
        return price * 12
    elif frequency == 'weekly':
        return (price / 7) * 365
    elif frequency == 'daily':
        return price * 365
    elif frequency == 'quarterly':
        return price * 4
    else:  # Yearly
        return price
def process_df(df):

    df['yearly_price'] = df['price'].apply(convert_frequencies)

    df = df.reset_index(drop=True)

    df['longitude'] = df['location'].apply(lambda x: x['longitude'])
    df['latitude'] = df['location'].apply(lambda x: x['latitude'])

    df['price_zscore'] = zscore(df['yearly_price'])

    df['longitude'] = df['location'].apply(lambda x: x['longitude'])
    df['latitude'] = df['location'].apply(lambda x: x['latitude'])

    df['update_reason'] = df['listingUpdate'].apply(lambda x: x['listingUpdateReason'])
    df['update_date'] = df['listingUpdate'].apply(lambda x: x['listingUpdateDate'])

    df['date_updated'] = pd.to_datetime(df['update_date'])

    threshold = 3

    df = df[df['price_zscore'].abs() <= threshold]

    return df


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/properties")
def read_root():
    df = get_df()
    df = process_df(df)
    json_df = df[['bedrooms', 'yearly_price', 'latitude', 'longitude', 'date_updated']].to_json(orient='records')
    return json_df


class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}

@app.get("/get_property/{property_id}")
def get_item(property_id: int):

    client = connect_to_client()

    db = connect_to_db(client, "rightmove")

    collection = connect_to_collection(db, "properties")

    data = get_property(collection, property_id)

    return data
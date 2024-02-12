import os
from typing import Union

from fastapi import FastAPI

from datetime import datetime, timedelta

from pydantic import BaseModel, ValidationError, validator

import json

import pandas as pd

from pymongo import MongoClient

from models.PricingCategory import PricingCategory
from models.Property import Property

from data_processing.DataPreprocessor import DataPreprocessor

app = FastAPI()
MONGO_DB_URL = "mongodb://localhost:27017/"

# MONGO_DB_URL = os.environ.get('MONGO_DB_URL')

@app.on_event("startup")
def startup_event():
    global df
    df = get_properties()
    walk_df = get_walk_scores()

    preprocessor = DataPreprocessor()
    df = preprocessor.preprocess_properties(df)
    walk_df = preprocessor.preprocess_walk_score(walk_df)

    df = preprocessor.merge_dataframes(df, walk_df)

    # df.to_csv("data.csv", index=False)

@app.post("/get_text")
def get_text(category: PricingCategory):
    category = category.category
    subset = df[df['price_category'] == category]
    combined_text = ' '.join(subset['text'].tolist())
    return {"category_text": combined_text}

from pydantic import BaseModel, ValidationError, validator

class AddedDates(BaseModel):
    category: str

    # Optional: Validator to provide a more specific error message
    @validator('category')
    def check_category(cls, v):
        if v not in ['Cheap', 'Average', 'Expensive']:
            raise ValidationError('Pricing must be "Cheap", "Average", or "Expensive"')
        return v
@app.get("/get_recents/{days}")
def get_recents(days: int):
    new_df = df[df['listingUpdateReason'] == 'new']

    today = pd.Timestamp(datetime.now(), tz='UTC')

    # Calculate the start date of the last week (7 days ago)
    date_start = today - timedelta(days=days)

    new_df['firstVisibleDate'] = pd.to_datetime(new_df['firstVisibleDate'], utc=True)

    # Filter the DataFrame for rows where the datetime is within the last week
    in_between_rows = new_df[(new_df['firstVisibleDate'] > date_start) & (df['firstVisibleDate'] <= today)]

    # Get the total number of rows
    total_rows = len(in_between_rows)
    return {"properties_added": total_rows}
@app.get("/properties")
def properties():
    property_df = df[['id', 'price', 'bedrooms', 'bathrooms', 'longitude', 'latitude', 'listingUpdateReason', 'firstVisibleDate']]
    return property_df.to_dict(orient='records')


@app.get("/")
def read_root():
    return {"Hello": "World"}

def connect_to_client():
    client = MongoClient(MONGO_DB_URL)
    return client

def connect_to_db(client, db):
    db = client[db]
    return db

def connect_to_collection(db, collection):
    collection = db[collection]
    return collection

def get_properties(number_of_records=25000):
    client = connect_to_client()

    db = connect_to_db(client, "rightmove")

    collection = connect_to_collection(db, "properties")

    fields = {"id": 1, "price.amount": 1, "price.frequency": 1, "firstVisibleDate": 1, "bedrooms": 1, "bathrooms": 1, "listingUpdate": 1, 'location': 1, 'summary': 1, "feature_list": 1}
    query = {}

    # Fetch data from the collection
    data = collection.find(query, fields, limit=number_of_records)

    # Convert to Pandas DataFrame
    df = pd.DataFrame(list(data))

    client.close()

    return df

def get_walk_scores(number_of_records=1000):
    client = connect_to_client()

    db = connect_to_db(client, "rightmove")

    collection = connect_to_collection(db, "walk_scores")

    fields = {"id": 1, "scores.walk_score": 1}
    query = {}

    # Fetch data from the collection
    data = collection.find(query, fields)

    # Convert to Pandas DataFrame
    walk_df = pd.DataFrame(list(data))

    client.close()

    return walk_df


def get_property(collection, propery_id):
    fields = {"id": propery_id}
    query = {}
    data = collection.find(query, fields)

    return data

if __name__ == "__main__":
    df = get_properties()
    walk_df = get_walk_scores()

    preprocessor = DataPreprocessor()
    df = preprocessor.preprocess_properties(df)
    walk_df = preprocessor.preprocess_walk_score(walk_df)

    df = preprocessor.merge_dataframes(df, walk_df)

    df.dropna(inplace=True)

    df.to_parquet("data.parquet", index=False)

    print(df)
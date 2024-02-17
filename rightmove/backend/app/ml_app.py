import os
from http.client import HTTPException
from typing import Union

from fastapi import FastAPI

from pydantic import BaseModel
from sklearn.neighbors import BallTree

from math import radians

import mlflow.pyfunc

import pandas as pd

import math
import numpy as np

from dotenv import load_dotenv
load_dotenv("/Users/alexander.girardet/Code/Personal/projects/rightmove_project/.env")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

print(MLFLOW_TRACKING_URI)

from pymongo import MongoClient

app = FastAPI()

# Load the model from the MLflow Model Registry
model_name = "Random Forest Walk Score"  # Replace with your model name
model_stage = "Staging"  # Load the production version
model_uri = f"models:/{model_name}/{model_stage}"
model = mlflow.pyfunc.load_model(model_uri)

from dotenv import load_dotenv
load_dotenv("/Users/alexander.girardet/Code/Personal/projects/rightmove_project/.env")

MONGO_URI = os.environ.get("MONGO_URI")

GCS_PARQUET_URL = "https://storage.googleapis.com/rightmove-resources-public/UK_pois.parquet"
WALK_SCORES_COLLECTION = "walk_scores"

print(MONGO_URI)

BATCH_SIZE = 50
class WalkScoreProcessor():

    def __init__(self):
        self.earth_radius = 6371000  # Earth radius in metres
        self.pois_df = pd.read_parquet(GCS_PARQUET_URL)
        self.ball_tree = BallTree(self.pois_df[['lon_rad', 'lat_rad']].values,
                                  metric='haversine')  # What is the ball tree doing?
        self.amenity_weights = {
            "grocery": [3],
            "restaurants": [.75, .45, .25, .25, .225, .225, .225, .225, .2, .2],
            "shopping": [.5, .45, .4, .35, .3],
            "coffee": [1.25, .75],
            "banks": [1],
            "parks": [1],
            "schools": [1],
            "books": [1],
            "entertainment": [1],
        }

    def fetch_current_ids(self):
        client = MongoClient(MONGO_URI)
        db = client["rightmove"]
        collection = db[WALK_SCORES_COLLECTION]
        query = {}
        data = collection.find(query, {"id": 1})
        return [x['id'] for x in list(data)]

    def process_results_df(self, distance_series, pois_df):
        results_df = pd.DataFrame(distance_series)

        results_df = results_df.join(pois_df['amenities'], how='left')

        results_df['distance_in_metres'] = results_df['distance'].apply(lambda x: x * self.earth_radius)

        results_df['distance_decayed'] = results_df['distance_in_metres'].apply(lambda x: float(self.distance_decay(x)))

        return results_df

    def distance_decay(self, distance):
        dist = distance / 1000
        score = math.e ** ((-5.0 * (dist / 4)) ** 5.0)
        return score

    def calculate_amenity_walk_score(self, property_distance_df, amenity, weights):
        k = len(weights)
        weight_array = np.array(weights)

        dist_array = property_distance_df[property_distance_df['amenities'] == amenity].iloc[0:k][
            'distance_decayed'].values
        dist_array_padded = np.pad(dist_array, (0, weight_array.size - dist_array.size), 'constant')

        scores_array = dist_array_padded * weight_array

        amenity_score = scores_array.sum()

        return amenity_score

    def calculuate_walk_score(self, longitude, latitude):

        radian_longitude = radians(longitude)
        radian_latitude = radians(latitude)

        k = 100  # Maximum number of amenities to return

        distances, indices = self.ball_tree.query([[radian_longitude, radian_latitude]], k=k, return_distance=True)

        dist_series = pd.Series(distances[0], index=indices[0], name='distance')

        results_df = self.process_results_df(dist_series, self.pois_df)

        scores_dict = {}

        for key, values in self.amenity_weights.items():
            amenity_score = self.calculate_amenity_walk_score(results_df, key, values)

            scores_dict[key] = amenity_score

        return scores_dict
class Property(BaseModel):
    bedrooms: float
    bathrooms: float
    longitude: float
    latitude: float
    walk_score: float

class Coordinates(BaseModel):
    longitude: float
    latitude: float
@app.post("/predict")
async def predict_rent(input_property: Property):
    try:
        # Make a prediction
        features_df = pd.DataFrame(input_property.dict(), index=[0])
        prediction = model.predict(features_df)
        return {"prediction": prediction[0]}

    except Exception as e:
        raise HTTPException()

@app.post("/walk_score")
async def predict_rent(input_coordinates: Coordinates):
    try:
        # Make a prediction
        walk_score_processor = WalkScoreProcessor()
        input_coordinates = input_coordinates.dict()
        longitude = input_coordinates['longitude']
        latitude = input_coordinates['latitude']
        scores_dict = walk_score_processor.calculuate_walk_score(longitude, latitude)
        walk_score = sum(scores_dict.values()) * 6.67
        return {"walk_score": walk_score}

    except Exception as e:
        raise HTTPException()

@app.get("/")
def read_root():
    return {"Hello": "World"}

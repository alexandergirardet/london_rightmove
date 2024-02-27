import os
from http.client import HTTPException
from typing import Union

from fastapi import FastAPI

from pydantic import BaseModel
from app.data_processing.walk_score_processing import WalkScoreProcessor
from sklearn.neighbors import BallTree

from math import radians
from typing import List
import mlflow.pyfunc

import pandas as pd

import math
import numpy as np

# from dotenv import load_dotenv
# load_dotenv("/Users/alexander.girardet/Code/Personal/projects/rightmove_project/.env")


class Property(BaseModel):
    bedrooms: float
    bathrooms: float
    longitude: float
    latitude: float
    walk_score: float


class Coordinates(BaseModel):
    longitude: float
    latitude: float


MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

app = FastAPI()

model_name = "Random Forest Walk Score"
model_stage = "Staging"
model_uri = f"models:/{model_name}/{model_stage}"
model = mlflow.pyfunc.load_model(model_uri)

MONGO_URI = os.environ.get("MONGO_URI")

GCS_PARQUET_URL = (
    "https://storage.googleapis.com/rightmove-resources-public/UK_pois.parquet"
)
WALK_SCORES_COLLECTION = "walk_scores"

BATCH_SIZE = 50


@app.post("/predict")
async def predict_rent(input_property: Property):
    try:
        features_df = pd.DataFrame(input_property.dict(), index=[0])
        prediction = model.predict(features_df)
        return {"prediction": prediction[0]}

    except Exception as e:
        raise HTTPException()


@app.post("/batch-predict")
async def batch_predict_rent(input_properties: List[Property]):
    try:
        properties_dicts = [property.dict() for property in input_properties]
        features_df = pd.DataFrame(properties_dicts)
        predictions = model.predict(features_df)

        return {"predictions": predictions.tolist()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/walk_score")
async def generate_walk_score(input_coordinates: Coordinates):
    try:
        walk_score_processor = WalkScoreProcessor()
        input_coordinates = input_coordinates.dict()
        longitude = input_coordinates["longitude"]
        latitude = input_coordinates["latitude"]
        scores_dict = walk_score_processor.calculuate_walk_score(longitude, latitude)
        walk_score = sum(scores_dict.values()) * 6.67
        return {"walk_score": walk_score}

    except Exception as e:
        raise HTTPException()


@app.get("/")
def read_root():
    return {"Hello": "World"}

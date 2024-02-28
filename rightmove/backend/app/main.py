import os
from http.client import HTTPException
from typing import Union

from fastapi import FastAPI
import psycopg2
from psycopg2.extras import RealDictCursor

from pydantic import BaseModel
from app.data_processing.walk_score_processing import WalkScoreProcessor
# from data_processing.walk_score_processing import WalkScoreProcessor
from sklearn.neighbors import BallTree
import json

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


SQL_QUERY = """
SELECT d.dataset_source
FROM datasets d
INNER JOIN (
    SELECT i.source_id AS dataset_id
    FROM inputs i
    INNER JOIN (
        SELECT run_id
        FROM model_versions
        ORDER BY version DESC
        LIMIT 1
    ) mv ON i.destination_id = mv.run_id
    WHERE i.source_type = 'DATASET'
    LIMIT 1
) subquery ON d.dataset_uuid = subquery.dataset_id;
"""
def fix_database_uri(uri: str) -> str:
    # Check if URI contains '+psycopg2' and remove it
    if "+psycopg2" in uri:
        uri = uri.replace("+psycopg2", "")
    return uri
@app.get("/latest-dataset")
def get_latest_dataset_source():
    try:
        PG_URI = fix_database_uri(MLFLOW_TRACKING_URI)

        with psycopg2.connect(PG_URI, cursor_factory=RealDictCursor) as conn:
            with conn.cursor() as cur:
                cur.execute(SQL_QUERY)

                result = cur.fetchone()
                if result:
                    dataset_source_info = json.loads(result["dataset_source"])
                    uri = dataset_source_info.get("uri", "URI not found")
                    return {"uri": uri}
                else:
                    return {"error": "No dataset source found for the latest model version."}

    except Exception as e:
        return {"error": str(e)}



@app.get("/")
def read_root():
    return {"Hello": "World"}

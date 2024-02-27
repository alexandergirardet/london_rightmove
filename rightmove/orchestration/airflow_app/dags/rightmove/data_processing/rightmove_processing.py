import os

from apache_beam.io.mongodbio import ReadFromMongoDB, WriteToMongoDB

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import logging
from sklearn.neighbors import BallTree
import pandas as pd
import math
import numpy as np
import datetime
from math import radians

from pymongo import MongoClient

# from dotenv import load_dotenv
# load_dotenv("/Users/alexander.girardet/Code/Personal/projects/rightmove_project/.env")

MONGO_URI = os.environ.get("MONGO_URI")

GCS_PARQUET_URL = "https://storage.googleapis.com/rightmove-resources-public/UK_pois.parquet"  # TODO: Make this private
WALK_SCORES_COLLECTION = "walk_scores"

BATCH_SIZE = 50


class ProcessElement(beam.DoFn):
    def fetch_current_ids(self):
        client = MongoClient(MONGO_URI)
        db = client["rightmove"]
        collection = db[WALK_SCORES_COLLECTION]
        query = {}
        data = collection.find(query, {"id": 1})
        return [x["id"] for x in list(data)]

    def process_results_df(self, distance_series, pois_df):
        results_df = pd.DataFrame(distance_series)

        results_df = results_df.join(pois_df["amenities"], how="left")

        results_df["distance_in_metres"] = results_df["distance"].apply(
            lambda x: x * self.earth_radius
        )

        results_df["distance_decayed"] = results_df["distance_in_metres"].apply(
            lambda x: float(self.distance_decay(x))
        )

        return results_df

    def distance_decay(sefl, distance):
        M = float(1)
        dist = distance / 1000
        score = math.e ** ((-5.0 * (dist / 4)) ** 5.0)
        return score

    def calculate_amenity_walk_score(self, property_distance_df, amenity, weights):
        k = len(weights)
        weight_array = np.array(weights)

        dist_array = (
            property_distance_df[property_distance_df["amenities"] == amenity]
            .iloc[0:k]["distance_decayed"]
            .values
        )
        dist_array_padded = np.pad(
            dist_array, (0, weight_array.size - dist_array.size), "constant"
        )

        scores_array = dist_array_padded * weight_array

        amenity_score = scores_array.sum()

        return amenity_score

    def calculuate_walk_score(self, property, ball_tree, amenity_weights, pois_df):
        property_id = property["id"]
        latitude = property["location"]["latitude"]
        longitude = property["location"]["longitude"]

        radian_longitude = radians(longitude)
        radian_latitude = radians(latitude)

        k = 100  # Maximum number of amenities to return

        distances, indices = ball_tree.query(
            [[radian_longitude, radian_latitude]], k=k, return_distance=True
        )

        dist_series = pd.Series(distances[0], index=indices[0], name="distance")

        results_df = self.process_results_df(dist_series, pois_df)

        # print(results_df)

        scores_dict = {}

        walk_score = 0

        for key, values in amenity_weights.items():
            amenity_score = self.calculate_amenity_walk_score(results_df, key, values)

            scores_dict[key] = amenity_score

        return scores_dict

    def setup(self):
        self.earth_radius = 6371000  # Earth radius in metres
        self.pois_df = pd.read_parquet(GCS_PARQUET_URL)
        self.ball_tree = BallTree(
            self.pois_df[["lon_rad", "lat_rad"]].values, metric="haversine"
        )  # What is the ball tree doing?
        self.amenity_weights = {
            "grocery": [3],
            "restaurants": [
                0.75,
                0.45,
                0.25,
                0.25,
                0.225,
                0.225,
                0.225,
                0.225,
                0.2,
                0.2,
            ],
            "shopping": [0.5, 0.45, 0.4, 0.35, 0.3],
            "coffee": [1.25, 0.75],
            "banks": [1],
            "parks": [1],
            "schools": [1],
            "books": [1],
            "entertainment": [1],
        }
        self.processed_ids = self.fetch_current_ids()

    def process(self, element):  # TODO: ADD ID processing to avoid duplicate processing
        logging.info(f"Processing element: {len(element)}")
        for ele in element:
            if ele["id"] not in self.processed_ids:
                property = {"id": ele["id"], "location": ele["location"]}
                logging.info(f"Processing property: {property}")
                scores_dict = self.calculuate_walk_score(
                    property, self.ball_tree, self.amenity_weights, self.pois_df
                )
                walk_score = sum(scores_dict.values()) * 6.67
                scores_dict["walk_score"] = walk_score

                property["scores"] = scores_dict

                property[
                    "processing_timestamp"
                ] = datetime.datetime.utcnow().timestamp()

                yield property
            else:
                logging.info(f"Property already processed: {ele['id']}")
                continue


def run():
    with beam.Pipeline(options=PipelineOptions()) as pipeline:
        (
            pipeline
            | "Read from Mongo"
            >> ReadFromMongoDB(
                uri=MONGO_URI, db="rightmove", coll="properties", bucket_auto=True
            )  # Only return the id and the location
            | "Batch Elements"
            >> beam.BatchElements(min_batch_size=BATCH_SIZE, max_batch_size=BATCH_SIZE)
            | "Process each element" >> beam.ParDo(ProcessElement())
            | "Write to MongoDB"
            >> WriteToMongoDB(
                uri=MONGO_URI,
                db="rightmove",
                coll=WALK_SCORES_COLLECTION,
                batch_size=10,
            )
        )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()

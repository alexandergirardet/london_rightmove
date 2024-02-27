from sklearn.neighbors import BallTree
import math

from math import radians
import pandas as pd
import numpy as np

GCS_PARQUET_URL = (
    "https://storage.googleapis.com/rightmove-resources-public/UK_pois.parquet"
)
WALK_SCORES_COLLECTION = "walk_scores"


class WalkScoreProcessor:
    def __init__(self):
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

    def distance_decay(self, distance):
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

    def calculuate_walk_score(self, longitude, latitude):
        radian_longitude = radians(longitude)
        radian_latitude = radians(latitude)

        k = 100  # Maximum number of amenities to return

        distances, indices = self.ball_tree.query(
            [[radian_longitude, radian_latitude]], k=k, return_distance=True
        )

        dist_series = pd.Series(distances[0], index=indices[0], name="distance")

        results_df = self.process_results_df(dist_series, self.pois_df)

        scores_dict = {}

        for key, values in self.amenity_weights.items():
            amenity_score = self.calculate_amenity_walk_score(results_df, key, values)

            scores_dict[key] = amenity_score

        return scores_dict

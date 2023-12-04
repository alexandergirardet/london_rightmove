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

BATCH_SIZE = 50
class ProcessElement(beam.DoFn):

    def process_results_df(self, distance_series, pois_df):
        results_df = pd.DataFrame(distance_series)

        results_df = results_df.join(pois_df['amenities'], how='left')

        results_df['distance_in_metres'] = results_df['distance'].apply(lambda x: x * self.earth_radius)

        results_df['distance_decayed'] = results_df['distance_in_metres'].apply(lambda x: float(self.distance_decay(x)))

        return results_df

    def distance_decay(sefl, distance):
        M = float(1)
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

    def calculuate_walk_score(self, property, ball_tree, amenity_weights, pois_df):

        property_id = property['id']
        latitude = property['location']['latitude']
        longitude = property['location']['longitude']

        radian_longitude = radians(longitude)
        radian_latitude = radians(latitude)

        k = 100  # Maximum number of amenities to return

        distances, indices = ball_tree.query([[radian_longitude, radian_latitude]], k=k, return_distance=True)

        dist_series = pd.Series(distances[0], index=indices[0], name='distance')

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
        self.pois_df = pd.read_csv("../resources/data/London_pois.csv", index_col=0)
        self.ball_tree = BallTree(self.pois_df[['lon_rad', 'lat_rad']].values, metric='haversine')  # What is the ball tree doing?
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
    def process(self, element):
        logging.info(f"Processing element: {len(element)}")
        for ele in element:
            property = {
                "id": ele['id'],
                "location": ele['location']
            }
            logging.info(f"Processing property: {property}")
            scores_dict = self.calculuate_walk_score(property, self.ball_tree, self.amenity_weights, self.pois_df)
            walk_score = sum(scores_dict.values()) * 6.67
            scores_dict['walk_score'] = walk_score

            property['scores'] = scores_dict

            property['processing_timestamp'] = datetime.datetime.utcnow().timestamp()

            yield property
def run():
    with beam.Pipeline(options=PipelineOptions()) as pipeline:
        (pipeline | "Read from Mongo" >> ReadFromMongoDB(uri='mongodb://localhost:27017',
                           db='rightmove',
                           coll='properties') # Only return the id and the location
        | 'Batch Elements' >> beam.BatchElements(min_batch_size=BATCH_SIZE, max_batch_size=BATCH_SIZE)
        | 'Process each element' >> beam.ParDo(ProcessElement())
        | 'Write to MongoDB' >> WriteToMongoDB(uri='mongodb://localhost:27017',
                                                             db='rightmove',
                                                             coll='walk_score',
                                                             batch_size=10)
                                              )

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
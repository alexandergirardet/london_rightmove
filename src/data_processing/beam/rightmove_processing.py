from apache_beam.io.mongodbio import ReadFromMongoDB, WriteToMongoDB

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import logging

class ProcessElement(beam.DoFn):
    def process(self, element):
        logging.info(f"Processing element: {element['id']}")
        processed_element = element
        return [processed_element]
def run():
    with beam.Pipeline(options=PipelineOptions()) as pipeline:
        (pipeline | "Read from Mongo" >> ReadFromMongoDB(uri='mongodb://localhost:27017',
                           db='rightmove',
                           coll='properties')
           | 'Process each element' >> beam.ParDo(ProcessElement())
           | 'Write to MongoDB' >> WriteToMongoDB(uri='mongodb://localhost:27017',
                                                             db='rightmove',
                                                             coll='walk_score',
                                                             batch_size=10)
                                              )

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
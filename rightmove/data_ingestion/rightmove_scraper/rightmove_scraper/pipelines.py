# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from pymongo import MongoClient
import os
import datetime

# MONGO_URL = "mongodb://mongodb:27017/"
MONGO_URI = os.environ.get("MONGO_URI")


class RightmoveScraperPipeline:
    def __init__(self):
        self.batch = []

        self.client = MongoClient(MONGO_URI)
        db = self.client["rightmove"]
        self.collection = db["properties"]

    def process_item(self, item, spider):
        """
        Sending items to MongoDB in batches to reduce I/O operations
        """

        item["extraction_timestamp"] = datetime.datetime.utcnow().timestamp()

        self.batch.append(item)

        if len(self.batch) >= 50:  # Batch size of file
            self.collection.insert_many(self.batch)
            self.batch = []

        return item

    def close_spider(self, spider):
        print("SPIDER CLOSING...")

        if len(self.batch) > 0:
            self.collection.insert_many(self.batch)  # Send remaining items

        self.client.close()

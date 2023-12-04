# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from pymongo import MongoClient
import datetime


class RightmoveScraperPipeline:

    def __init__(self):
        self.batch = []

        # self.token = self.get_access_token()

        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["rightmove"]
        # Access collection
        self.collection =self.db["properties"]


        now = datetime.datetime.utcnow()

        # The now instance is denominated in UTC 0 time for commonality over several time zones

        self.ymdhm = f"{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}"
        self.now_timestamp = int(now.timestamp())
    def process_item(self, item, spider):
        """
        Sending items to MongoDB in batches to reduce I/O operations
        """

        item['extraction_timestamp'] = datetime.datetime.utcnow().timestamp()

        self.batch.append(item)

        if len(self.batch) >= 50:  # Batch size of file

            self.collection.insert_many(self.batch)
            self.batch = []

        return len(self.batch)

    def close_spider(self, spider):
        print("SPIDER CLOSING...")

        if len(self.batch) > 0:
            self.collection.insert_many(self.batch)  # Send remaining items

        self.client.close()

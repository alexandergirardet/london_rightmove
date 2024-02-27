import scrapy
import os
import csv
import requests
import io

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from bs4 import BeautifulSoup

from pymongo import MongoClient

# MONGO_URL = "mongodb://mongodb:27017/"
MONGO_URI = os.environ.get("MONGO_URI")


class RightmoveSpider(scrapy.Spider):
    name = "rightmove"

    def __init__(self, *args, **kwargs):
        self.headers = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
            "Referer": "https://www.rightmove.co.uk/property-to-rent/find.html?locationIdentifier=REGION%5E87490&index=24&propertyTypes=&includeLetAgreed=false&mustHave=&dontShow=&furnishTypes=&keywords=",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.88 Safari/537.36",
            "sec-ch-ua": '" Not A;Brand";v="99", "Chromium";v="100", "Google Chrome";v="100"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
        }

        self.rightmove_ids = self.get_property_ids()

        print(self.rightmove_ids)

        print("Number of IDs: ", len(self.rightmove_ids))

        logger.info(f"Fetching new MongoDB data from {MONGO_URI}...")

        self.fetched_outcodes = self.get_outcodes()

    def start_requests(self):
        for codes in self.fetched_outcodes:
            rightmove_code = codes[1]
            postcode = codes[0]
            for index_jump in range(
                0, 100, 25
            ):  # Adjusting to 100 so I can have some extra values to test with
                url = f"https://www.rightmove.co.uk/api/_search?locationIdentifier=OUTCODE%5E{rightmove_code}&numberOfPropertiesPerPage=24&radius=10.0&sortType=6&index={index_jump}&includeLetAgreed=false&viewType=LIST&channel=RENT&areaSizeUnit=sqft&currencyCode=GBP&isFetching=false"

                yield scrapy.Request(
                    method="GET", url=url, headers=self.headers, callback=self.parse
                )

    def parse(self, response):
        listings = response.json()["properties"]
        for listing in listings:
            property_id = listing["id"]

            if property_id not in self.rightmove_ids:
                property_url = f"https://www.rightmove.co.uk/properties/{property_id}"

                yield scrapy.Request(
                    method="GET",
                    url=property_url,
                    headers=self.headers,
                    callback=self.parse_property,
                    meta={"item": listing},
                )
            else:
                print("Already loaded in")

    def parse_property(self, response):
        soup = BeautifulSoup(response.text, "lxml")

        item = response.meta["item"]

        # Get feature list
        try:
            uls = soup.find("ul", {"class": "_1uI3IvdF5sIuBtRIvKrreQ"})
            features = uls.find_all("li")
            feature_list = [feature.text for feature in features]
        except:
            feature_list = None

        # Get full summary
        summary = soup.find("div", {"class": "OD0O7FWw1TjbTD4sdRi1_"}).div.text

        # Assign content to item
        item["feature_list"] = feature_list
        item["summary"] = summary

        yield item

    def get_outcodes(self):
        # URL of the CSV file in the public GCS bucket
        csv_url = "https://storage.googleapis.com/rightmove-resources-public/rightmove_outcodes.csv"

        # Download the CSV file
        response = requests.get(csv_url)
        if response.status_code == 200:
            # Convert binary data to a text stream
            csv_text = io.StringIO(response.content.decode("utf-8"))

            # Read CSV data
            reader = csv.reader(csv_text)
            outcodes = list(reader)
            outcodes = outcodes[1:]  # Skip header row
            outcodes = [(outcode[1], outcode[2]) for outcode in outcodes]
            return outcodes
        else:
            print("Failed to download CSV file")
            return []

    def get_property_ids(self) -> list:
        client = MongoClient(MONGO_URI)
        # client = MongoClient("mongodb://localhost:27017/")
        db = client["rightmove"]
        # Access collection
        collection = db["properties"]

        # logging.info("Connected to MongoDB")

        rightmove_ids = collection.find({}, {"id": 1})

        # Convert the result to a list of IDs
        ids = [doc["id"] for doc in rightmove_ids]

        client.close()

        return ids

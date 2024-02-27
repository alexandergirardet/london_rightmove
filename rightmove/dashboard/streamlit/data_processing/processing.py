import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self, with_text=False, with_binary=False):
        self.with_text = with_text
        self.with_binary = with_binary

    @staticmethod
    def convert_frequencies(x):
        frequency = x["frequency"]
        price = x["amount"]

        if frequency == "monthly":
            return price * 12
        elif frequency == "weekly":
            return (price / 7) * 365
        elif frequency == "daily":
            return price * 365
        elif frequency == "quarterly":
            return price * 4
        else:  # Yearly
            return price

    @staticmethod
    def remove_anomalies(df, percentile_threshold=0.99):
        percentile_thresholds = df[["price", "bedrooms", "bathrooms"]].quantile(
            percentile_threshold
        )

        filtered_df = df[
            (df["price"] <= percentile_thresholds["price"])
            & (df["bedrooms"] <= percentile_thresholds["bedrooms"])
            & (df["bathrooms"] <= percentile_thresholds["bathrooms"])
        ]
        return filtered_df

    @staticmethod
    def merge_text(x):
        summary, feature_list = x[0], x[1]
        feature_list_joined = ", ".join(feature_list) if feature_list else ""
        return feature_list_joined + " , " + summary

    @staticmethod
    def label_price(price):
        if price < 8000:
            return "Cheap"
        elif price < 20_000:
            return "Average"
        else:
            return "Expensive"

    def preprocess_properties(self, df):
        df["longitude"] = df["location"].apply(lambda x: x["longitude"])
        df["latitude"] = df["location"].apply(lambda x: x["latitude"])
        df["price"] = df["price"].apply(self.convert_frequencies)

        if self.with_text:
            df["text"] = df[["summary", "feature_list"]].apply(self.merge_text, axis=1)
        if self.with_binary:
            df["commercial"] = df["commercial"].apply(lambda x: 1 if x else 0)
            df["development"] = df["development"].apply(lambda x: 1 if x else 0)
            df["students"] = df["students"].apply(lambda x: 1 if x else 0)

        df["price_category"] = df["price"].apply(self.label_price)
        df["listingUpdateReason"] = df["listingUpdate"].apply(
            lambda x: x["listingUpdateReason"]
        )
        df["firstVisibleDate"] = pd.to_datetime(df["firstVisibleDate"], utc=True)
        df = self.remove_anomalies(df)
        df = df.drop(columns=["location", "_id", "listingUpdate"])
        return df

    @staticmethod
    def preprocess_walk_score(df):
        df = df.drop_duplicates(subset=["id"])
        df["walk_score"] = df["scores"].apply(lambda x: int(x["walk_score"]))
        df = df.drop(columns=["_id", "scores"])
        return df
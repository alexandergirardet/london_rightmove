import streamlit as st
import requests
import pydeck as pdk
from wordcloud import WordCloud
import pandas as pd
from datetime import datetime, timedelta
import geopandas
import matplotlib.pyplot as plt
import json
import seaborn as sns
import plotly.express as px
import os
from data_processing.processing import DataPreprocessor
from pymongo import MongoClient

import logging

logging.basicConfig(level=logging.INFO)

from dotenv import load_dotenv
load_dotenv("/Users/alexander.girardet/Code/Personal/projects/rightmove_project/.env")

MONGO_URI = os.environ.get("MONGO_URI")

def load_data_from_mongo(collection_name="properties", fields=None):
    logging.info("Loading data from mongo")

    client = MongoClient(MONGO_URI)  # Hosted with Docker

    db = client["rightmove"]

    collection = db[collection_name]

    query = {}

    data = collection.find(query, fields)

    df = pd.DataFrame(list(data))

    if len(df) == 0:
        raise ValueError(f"No data found in collection {collection_name}")
    else:
        logging.info(f"Data loaded from collection {collection_name}")

    return df

def preprocess_data(property_df, walkscore_df):
    preprocessor = DataPreprocessor(with_text=True, with_binary=False)

    property_df = preprocessor.preprocess_properties(property_df)
    walk_df = preprocessor.preprocess_walk_score(walkscore_df)

    df = property_df.merge(walk_df, on="id", how="left")

    logging.info("Data preprocessed")

    return df

def fetch_preprocess_data():
    property_df = load_data_from_mongo(
        collection_name="properties",
        fields={
            "id": 1,
            "price.amount": 1,
            "price.frequency": 1,
            "firstVisibleDate": 1,
            "bedrooms": 1,
            "bathrooms": 1,
            "listingUpdate": 1,
            "location": 1,
            "summary": 1,
            "feature_list": 1,
        },
    )
    walkscore_df = load_data_from_mongo(
        collection_name="walk_scores", fields={"id": 1, "scores": 1}
    )

    df = preprocess_data(property_df, walkscore_df)

    return df

@st.cache_data
def load_data():
    df = pd.read_parquet(
        "gs://rightmove-artifacts-ml/streamlit_data/2024-02-27-12-32-07/data.parquet"
    )

    df["monthly_price"] = df["price"] / 12
    df = df.dropna()
    return df


def get_recents(subset, days):
    new_df = subset[subset["listingUpdateReason"] == "new"]

    today = pd.Timestamp(datetime.now(), tz="UTC")

    # Calculate the start date of the last week (7 days ago)
    date_start = today - timedelta(days=days)

    new_df["firstVisibleDate"] = pd.to_datetime(new_df["firstVisibleDate"], utc=True)

    # Corrected filtering to use new_df instead of df
    in_between_rows = new_df[
        (new_df["firstVisibleDate"] > date_start)
        & (new_df["firstVisibleDate"] <= today)
    ]

    # Get the total number of rows
    total_rows = len(in_between_rows)
    return total_rows


def plot_bedrooms_distribution(df):
    max_bedrooms = df["bedrooms"].max()
    fig = px.histogram(df, x="bedrooms", title="Distribution of Bedrooms")
    fig.update_layout(
        xaxis=dict(title="Number of Bedrooms", tickmode="linear", dtick=1),
        yaxis_title="Number of Properties",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_bathrooms_distribution(df):
    # Determine the maximum number of bathrooms to set appropriate bins
    max_bathrooms = df["bathrooms"].max()
    fig = px.histogram(df, x="bathrooms", title="Distribution of Bathrooms")
    fig.update_layout(
        xaxis=dict(title="Number of Bathrooms", tickmode="linear", dtick=1),
        yaxis_title="Number of Properties",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_price_density(df):
    fig = px.histogram(
        df, x="monthly_price", title="Distribution of Monthly Rental Prices"
    )
    fig.update_layout(
        xaxis_title="Rental Price",
        yaxis_title="Number of Properties",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


df = load_data()

min_price, max_price = st.sidebar.slider(
    "Select a monthly rental price range:",
    min_value=int(df["monthly_price"].min()),  # Minimum value for the slider
    max_value=int(df["monthly_price"].max()),  # Maximum value for the slider
    value=(
        int(df["monthly_price"].min()),
        int(df["monthly_price"].max()),
    ),  # Initial range (min, max)
)

subset = df[(df["monthly_price"] >= min_price) & (df["monthly_price"] <= max_price)]

# Streamlit UI
col1, col2, col3 = st.columns(3)

# Calculate properties added since last week
properties_last_week = get_recents(subset, 8)  # Last 7 days
# Display metric in the first column, restrict to 2 decimal places
col1.metric(label="Properties Added Since Last Week", value=f"{properties_last_week}")

# Calculate properties added since yesterday
properties_yesterday = get_recents(subset, 2)  # Last 1 day
# Display metric in the second column, restrict to 2 decimal places
col2.metric(label="Properties Added Since Yesterday", value=f"{properties_yesterday}")

# Calculate the total number of properties
total_properties = len(subset)
# Display metric in the third column, restrict to 2 decimal places
col3.metric(label="Total Properties", value=f"{total_properties}")

st.header("Property Distribution Map")
layer = pdk.Layer(
    "HexagonLayer",  # `type` positional argument is here
    subset[["longitude", "latitude"]],  # `data` positional argument is here
    get_position=["longitude", "latitude"],
    auto_highlight=True,
    elevation_scale=50,
    pickable=True,
    elevation_range=[0, 3000],
    extruded=True,
    coverage=1,
)

# Set the viewport location
view_state = pdk.ViewState(
    longitude=-1.415,
    latitude=52.2323,
    zoom=6,
    min_zoom=5,
    max_zoom=15,
    pitch=40.5,
    bearing=-27.36,
)

# Combine everything and render a viewport
r = pdk.Deck(layers=[layer], initial_view_state=view_state)
st.info(
    "The map displays the distribution of properties based on their location. The higher the concentration of properties, the higher the elevation."
)
st.pydeck_chart(r)

st.header("Histogram and Density Plots of Property Features")

st.info(
    "The following plots provide a visual representation of the distribution of property features such as bedrooms, bathrooms, and rental prices."
)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(plot_bedrooms_distribution(subset), use_container_width=True)
with col2:
    st.plotly_chart(plot_bathrooms_distribution(subset), use_container_width=True)

# Density plot for price
st.plotly_chart(plot_price_density(subset), use_container_width=True)
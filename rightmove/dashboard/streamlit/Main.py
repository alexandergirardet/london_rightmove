import streamlit as st
import requests
import pydeck as pdk
from wordcloud import WordCloud
import pandas as pd
from datetime import datetime, timedelta
import geopandas
import matplotlib.pyplot as plt
import json

@st.cache_data
def load_data():
    df = pd.read_parquet("/Users/alexander.girardet/Code/Personal/projects/rightmove_project/rightmove/backend/app/data.parquet")    #
    return df

def get_recents(subset, days):
    new_df = subset[subset['listingUpdateReason'] == 'new']

    today = pd.Timestamp(datetime.now(), tz='UTC')

    # Calculate the start date of the last week (7 days ago)
    date_start = today - timedelta(days=days)

    new_df['firstVisibleDate'] = pd.to_datetime(new_df['firstVisibleDate'], utc=True)

    # Corrected filtering to use new_df instead of df
    in_between_rows = new_df[(new_df['firstVisibleDate'] > date_start) & (new_df['firstVisibleDate'] <= today)]

    # Get the total number of rows
    total_rows = len(in_between_rows)
    return total_rows

def get_walk_score(subset):
    return subset['walk_score'].mean()

df = load_data()

min_price, max_price = st.slider(
    'Select a price range:',
    min_value=int(df['price'].min()),  # Minimum value for the slider
    max_value=int(df['price'].max()),  # Maximum value for the slider
    value=(int(df['price'].min()), int(df['price'].max()))  # Initial range (min, max)
)

subset = df[(df['price'] >= min_price) & (df['price'] <= max_price)]

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

# Calculate average walk score
walk_score = get_walk_score(subset)
# Display metric in the third column, restrict to 2 decimal places
col3.metric(label="Average Walk Score", value=f"{walk_score:.2f}")



st.write("All properties map")
layer = pdk.Layer(
    'HexagonLayer',  # `type` positional argument is here
    subset[['longitude', 'latitude']],  # `data` positional argument is here
    get_position=['longitude', 'latitude'],
    auto_highlight=True,
    elevation_scale=50,
    pickable=True,
    elevation_range=[0, 3000],
    extruded=True,
    coverage=1)

# Set the viewport location
view_state = pdk.ViewState(
    longitude=-1.415,
    latitude=52.2323,
    zoom=6,
    min_zoom=5,
    max_zoom=15,
    pitch=40.5,
    bearing=-27.36)

# Combine everything and render a viewport
r = pdk.Deck(layers=[layer], initial_view_state=view_state)

st.pydeck_chart(r)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk


@st.cache_data
def load_data():
    df = pd.read_parquet(
        "gs://rightmove-artifacts-ml/streamlit_data/2024-02-27-12-32-07/data.parquet"
    )
    df["monthly_price"] = df["price"] / 12
    return df


df = load_data()
# Streamlit page setup
st.title("Bedroom, Bathroom, and Location Relationship with Price")

import plotly.express as px


def get_average_price(df):
    avg_price = df["monthly_price"].mean()
    return avg_price


def get_average_bedrooms(df):
    avg_bedrooms = df["bedrooms"].mean()
    return avg_bedrooms


def get_average_bathrooms(df):
    avg_bathrooms = df["bathrooms"].mean()
    return avg_bathrooms


def plot_price_by_bedrooms(df):
    fig = px.box(
        df,
        x="bedrooms",
        y="monthly_price",
        title="Rental Price Distribution by Number of bedrooms",
    )
    fig.update_layout(
        xaxis=dict(title="Number of Bedrooms"),
        yaxis=dict(title="Rental Price"),
        plot_bgcolor="rgba(0,0,0,0)",
        title_x=0.5,
    )
    return fig


def plot_price_by_bathrooms(df):
    fig = px.box(
        df,
        x="bathrooms",
        y="monthly_price",
        title="Rental Price Distribution by Number of Bathrooms",
    )
    fig.update_layout(
        xaxis=dict(title="Number of Bathrooms"),
        yaxis=dict(title="Rental Price"),
        plot_bgcolor="rgba(0,0,0,0)",
        title_x=0.5,
    )

    return fig


st.sidebar.title("Filters")
min_price, max_price = st.sidebar.slider(
    "Select Rental Price Range",
    min_value=int(df["monthly_price"].min()),
    max_value=int(df["monthly_price"].max()),
    value=(int(df["monthly_price"].min()), int(df["monthly_price"].max())),
)

# Filtering the DataFrame based on the selected price range
filtered_df = df[
    (df["monthly_price"] >= min_price) & (df["monthly_price"] <= max_price)
]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        label="Average Monthly Price", value=f"{get_average_price(filtered_df):.2f}"
    )
# Average Bedrooms and Bathroom
with col2:
    st.metric(
        label="Average Number of Bathrooms",
        value=f"{get_average_bathrooms(filtered_df):.2f}",
    )

with col3:
    st.metric(
        label="Average Number of Bedrooms",
        value=f"{get_average_bedrooms(filtered_df):.2f}",
    )

st.plotly_chart(plot_price_by_bathrooms(filtered_df), use_container_width=True)
st.plotly_chart(plot_price_by_bedrooms(filtered_df), use_container_width=True)

max_rental_price = df["monthly_price"].max()

# Assuming you've already grouped your data as needed or if you're using individual points,
# you can directly use the rental_price for elevation. For a true mean aggregation, you'd need
# to aggregate your data by the hexagon/bin locations, which requires additional preprocessing.

# For color, normalize the rental_price to get a value between 0 and 255 for the color scale
df["color_value"] = (df["monthly_price"] / max_rental_price) * 255
df["color_value"] = df["color_value"].astype(
    int
)  # Ensure it's an integer for color coding


# Function to create the heatmap
st.info(
    "The following map shows the distribution of rental prices in the selected area. The elevation of the hexagons represents the mean rental price of properties within each hexagon."
)
@st.cache_resource
def create_hexagon_map(
    dataframe,
    lat_col="latitude",
    lon_col="longitude",
    value_col="monthly_price",
    radius=200,
):
    """Create a hexagon map where the elevation represents the mean rental price of properties within each hexagon.

    Args:
        dataframe (pd.DataFrame): The dataframe containing the data.
        lat_col (str): Column name for latitude values.
        lon_col (str): Column name for longitude values.
        value_col (str): Column name for the values to average (mean rental price).
        radius (int): Radius of the hexagons in meters.

    Returns:
        pydeck.Deck: A pydeck Deck object ready to be displayed.
    """
    # Aggregate data by hexagon
    layer = pdk.Layer(
        "HexagonLayer",
        dataframe[[lon_col, lat_col, value_col]],
        get_position=[lon_col, lat_col],
        auto_highlight=True,
        elevation_scale=50,  # Adjust based on your data's scale for better visualization
        pickable=True,
        elevation_range=[0, 3000],  # Max elevation in meters
        extruded=True,  # Make hexagon 3D
        coverage=4,
        opacity=0.3,
        radius=radius,  # Radius of hexagon in meters
        get_elevation="monthly_price",  # Use the 'elevation' column if you've aggregated data
        get_fill_color="[255, 255, color_value, 140]",
    )

    # Set the initial view
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

    return r


# Example usage
hex_map = create_hexagon_map(filtered_df)
st.pydeck_chart(hex_map)

import geopandas
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math


@st.cache_data
def load_geo_data():
    # Load your GeoPandas DataFrame here
    # For example: gdf = gpd.read_file('your_file_path.shp')
    # Returning an example gdf, replace this with your actual data loading
    gdf = geopandas.read_file(
        "/Users/alexander.girardet/Code/Personal/projects/rightmove_project/notebooks/serving/london_borough_stats.geojson"
    )
    return gdf


def plot_geo_data(gdf, column):
    # Create the figure and axis with a larger size for better visibility
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot the GeoDataFrame with a more appealing color map and adjust the legend
    gdf.plot(
        column=column,
        cmap="viridis",
        legend=True,
        ax=ax,
        legend_kwds={
            "label": f"{reversed_options[column]}",
            "orientation": "horizontal",
        },
    )

    # Adjust the figure layout to accommodate the legend and ensure no clipping
    plt.tight_layout()

    return fig


def distance_decay(distance):
    dist = distance / 1000  # Convert distance to kilometers
    score = math.e ** ((-5.0 * (dist / 4)) ** 5.0)
    return score


def plot_distance_decay():
    # Generate distances from 0 to 2000 meters
    distances = np.linspace(0, 2000, 500)
    scores = np.array([distance_decay(d) for d in distances])

    # Plotting the decay of distance
    plt.figure(figsize=(10, 6))
    plt.plot(distances, scores, label="Distance Decay", color="blue")
    plt.xlabel("Distance (meters)")
    plt.ylabel("Score")
    plt.title("Distance Decay Effect on Score")
    plt.grid(True)
    plt.xlim(0, 2000)  # Limit x-axis to 2000 meters
    plt.legend()
    plt.tight_layout()
    return plt


def calculate_amenity_walk_score(distances, amenity_weights):
    total_score = 0
    for amenity, distance in distances.items():
        decayed_distance = distance_decay(distance)
        weights = amenity_weights.get(
            amenity, [1]
        )  # Default weight if amenity not found
        # Assume the first weight for simplicity, could be adapted for multiple distances per amenity
        amenity_score = decayed_distance * weights[0]
        total_score += amenity_score
    return total_score


# def get_walk_score(subset):
#     return subset['walk_score'].mean()
#
# # walk_score = get_walk_score(subset)
# # Display metric in the third column, restrict to 2 decimal places
# col3.metric(label="Average Walk Score", value=f"{walk_score:.2f}")

amenity_weights = {
    "grocery": [3],
    "restaurants": [3],
    "shopping": [2],
    "coffee": [2],
    "banks": [1],
    "parks": [1],
    "schools": [1],
    "books": [1],
    "entertainment": [1],
}

# Streamlit app setup for interactive walk score explanation
st.title("Interactive Walk Score Explanation")

st.write(
    """
         This application demonstrates how the walk score for a property is calculated based on the distances to various amenities. 
         Walk score is a measure of how friendly an area is to walking with a score from 0 to 100, where higher scores indicate better walkability.
         """
)

st.header("Walk Score Visualization")

st.write(
    """
         In visualizing the walk score we consider the average price of properties, the walk score, and the property count. This could provide an indication
            of the relationship between the walk score and the average price of properties in a given area. Additionally, the property count could provide an
            indication of the demand, and supply for properties in a given area. Logically, the higher the walk score, the higher the density of properties in
            a given area, and the higher the average price of properties in a given area.
         """
)

options = {
    "Price": "avg_price",
    "Walk Score": "mean_walk_score",
    "Property Count": "property_count",
}

reversed_options = {value: key for key, value in options.items()}

# Use the dictionary keys as the display labels and get the selected option value
selected_label = st.selectbox("Choose attribute to visualize:", options.keys())

option_value = options[selected_label]

gdf = load_geo_data()

# Display the plot in Streamlit
st.pyplot(plot_geo_data(gdf, option_value))

st.header("Distance Decay Visualization")
st.write(
    "This plot shows the decay of scores with increasing distance for a single amenity. It illustrates how closer amenities contribute more significantly to the walk score."
)
fig = plot_distance_decay()
st.pyplot(fig)

st.header("Customize Your Walk Score")
st.write(
    "Adjust the sliders below to simulate distances to different amenities and calculate a simplified walk score."
)

# Example of creating sliders for different amenities (simplified version)

distances = {}
for amenity in amenity_weights.keys():
    distance = st.slider(f"Distance to nearest {amenity} (meters)", 0, 2000, 500, 50)
    distances[amenity] = distance

total_walk_score = calculate_amenity_walk_score(distances, amenity_weights)

walk_score = total_walk_score * 6.67

st.metric("Total Walk Score", walk_score)

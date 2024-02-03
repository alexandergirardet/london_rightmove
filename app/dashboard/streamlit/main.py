import os
import requests
import pandas as pd
import numpy as np
import pydeck as pdk
import streamlit as st
from data_processing.DataPreprocessor import DataPreprocessor

st.title('UK Rightmove properties')

# Assume DATA_URL is constant; define it outside the function
DATA_URL = 'http://fastapi_app:80/properties'

# Corrected caching decorator usage for data loading
@st.cache_data(allow_output_mutation=True)
def load_data():
    if os.environ.get("staging"):
        response = requests.get(DATA_URL)
        data = pd.read_json(response.json())
        data['date_updated'] = pd.to_datetime(data['date_updated'])
    else:
        local_data_url = '/Users/alexander.girardet/Code/Personal/projects/rightmove_project/notebooks/model_development/data/train.csv'
        data = pd.read_csv(local_data_url, index_col=0)

    return data

# Corrected caching decorator usage for data preprocessing
@st.cache_data(allow_output_mutation=True)
def preprocess_data(df):
    data_preprocessor = DataPreprocessor()
    processed_df = data_preprocessor.remove_anomalies(df)
    return processed_df

data_load_state = st.text('Loading data...')
df = load_data()
data_load_state.text('Loading data...done!')

df = preprocess_data(df)

# Slider for filtering data based on bedrooms
bedrooms = st.slider('Bedrooms', 0, 12, 2)
filtered_data = df[df['bedrooms'] == bedrooms]

# Check if map data needs to be recalculated (it doesn't in this case)
if 'map_data' not in st.session_state:
    st.session_state.map_data = df  # Store the entire dataset for map initially

# Create a 3D Column Layer for the map
layer = pdk.Layer(
    'HexagonLayer',
    st.session_state.map_data,  # Use session state data
    get_position=['longitude', 'latitude'],
    auto_highlight=True,
    elevation_scale=50,
    pickable=True,
    elevation_range=[0, 3000],
    extruded=True,
    coverage=1)

# Render the map with the layer
st.pydeck_chart(pdk.Deck(
    initial_view_state=pdk.ViewState(
        latitude=df['latitude'].mean(),
        longitude=df['longitude'].mean(),
        zoom=5,
        pitch=50,
    ),
    layers=[layer],
))

st.subheader('Raw data')
st.write(filtered_data)  # Display filtered data

st.subheader('Yearly price histogram')
hist_values = np.histogram(filtered_data['price'], bins=100)[0]
st.bar_chart(hist_values)

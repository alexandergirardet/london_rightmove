import streamlit as st
import pandas as pd
import numpy as np
import requests

st.title('UK Rightmove properties')

DATE_COLUMN = 'date_updated'
DATA_URL = 'http://fastapi_app:80/properties'
@st.cache_data
def load_data(nrows):

    response = requests.get(DATA_URL)

    data = pd.read_json(response.json())
    data['date_updated'] = pd.to_datetime(data['date_updated'])
    return data



data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data(1000)
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

st.subheader('Raw data')
st.write(data)

st.subheader('Yearly price histogram')

bedrooms = st.slider('bedrooms', 0, 12, 2)

filtered_data = data[data['bedrooms'] == bedrooms]

hist_values = np.histogram(filtered_data['yearly_price'], bins=100)[0]

st.bar_chart(hist_values)

st.subheader('Map of all listings')

st.subheader(f'All properties with {bedrooms} Bedrooms')
st.map(filtered_data)

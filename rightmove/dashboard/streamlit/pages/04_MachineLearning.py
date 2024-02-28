import json

import streamlit as st
import requests
import pandas as pd
from streamlit_folium import folium_static
import folium

# Function to send the HTTP POST request
def predict_property_value(features):
    url = "http://localhost:8000/predict"  # Replace with your actual endpoint URL
    response = requests.post(url, json=features)
    return response.json()

def get_walk_score(coordinates):
    print(coordinates)
    url = "http://localhost:8000/walk_score"  # Replace with your actual endpoint URL
    response = requests.post(url, json=coordinates)
    return response.json()


# Streamlit user interface setup
st.title("Property Value Prediction")

st.write(
    "Enter the property details below and choose to either generate a Walk Score based on the location or input it manually to see the impact on the property value prediction.")

# Input fields for property features
with st.form(key='property_details'):
    bedrooms = st.number_input("Number of Bedrooms", min_value=1, value=3)
    bathrooms = st.number_input("Number of Bathrooms", min_value=1, value=2)
    latitude = st.number_input("Latitude", value=51.53)
    longitude = st.number_input("Longitude", value=-0.06)

    # Instructions for Walk Score
    st.write("You can either generate a Walk Score based on the coordinates or input a Walk Score manually.")

    if st.form_submit_button("Generate Walk Score"):
        coordinates = {"longitude": longitude, "latitude": latitude}
        walk_score_generated = round(get_walk_score(coordinates)['walk_score'], 2)
        st.session_state.generated_walk_score = walk_score_generated
        st.success(f"Generated Walk Score: {walk_score_generated}")
    walk_score = st.number_input("Or Input Walk Score", min_value=0, value=50, key="manual_walk_score")

    # Use generated walk score if available, else use manual input
    final_walk_score = st.session_state.get('generated_walk_score', walk_score)

    submitted = st.form_submit_button("Confirm Inputs")
    if submitted:
        st.write("### Inputs for Prediction")
        st.write(f"- Number of Bedrooms: {bedrooms}")
        st.write(f"- Number of Bathrooms: {bathrooms}")
        st.write(f"- Latitude: {latitude}")
        st.write(f"- Longitude: {longitude}")
        st.write(f"- Walk Score: {final_walk_score}")
        st.write("Use the 'Launch Prediction' button below to predict the property value based on these inputs.")

# Button to launch prediction after reviewing inputs
if st.button("Launch Prediction"):
    features = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "walk_score": final_walk_score,
        "latitude": latitude,
        "longitude": longitude
    }
    prediction = predict_property_value(features)
    monthly_value = prediction["prediction"] / 12
    formatted_value = f"£{monthly_value:,.2f} per month"  # Formats the number with comma as thousands separator and two decimal places
    # st.success(f"Predicted Property Value: {formatted_value}")
    st.success(f"A property with {features['bedrooms']} bedrooms and {features['bathrooms']} bathrooms, located at ({features['latitude']:,.2f}, {features['longitude']:,.2f}), with a Walk Score of {features['walk_score']}, is estimated to be worth {formatted_value}")
st.subheader("Select Property Location on Map")
m = folium.Map(location=[latitude, longitude], zoom_start=11)  # Default location, change as needed
folium.Marker(location=[latitude, longitude], tooltip="Move this marker to your property location", draggable=True).add_to(m)
folium_static(m)

# Button to make prediction  # Convert from yearly to monthly


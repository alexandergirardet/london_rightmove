import requests
import pandas as pd

# Define the URL of your batch prediction endpoint
url = "http://127.0.0.1:8000/batch-predict"

# Create a DataFrame with your test properties. This should match the structure expected by your API.
# For example, if your Property model expects 'size' and 'location', your DataFrame should reflect that.
df = pd.read_csv(
    "gs://rightmove-artifacts-ml/data/2024-02-17-14-18-14/test.csv", index_col=0
)

df = df[["bedrooms", "bathrooms", "longitude", "latitude", "walk_score"]]

print(df.head())

# Convert the DataFrame to a list of dictionaries
properties_list = df.to_dict("records")


# Make a POST request to the batch prediction endpoint
response = requests.post(url, json=properties_list)

# Check the status code to ensure the request was successful
assert response.status_code == 200

# Convert the response to JSON and retrieve the predictions
predictions = response.json().get("predictions")

# Perform any additional checks you need on the predictions
# For example, check the number of predictions matches the number of input properties
assert len(predictions) == len(properties_list)

print("Predictions:", predictions)

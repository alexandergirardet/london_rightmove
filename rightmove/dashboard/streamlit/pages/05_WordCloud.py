import streamlit as st
import requests
import pydeck as pdk
from wordcloud import WordCloud
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import json


@st.cache_data
def load_data():
    df = pd.read_parquet(
        "gs://rightmove-artifacts-ml/streamlit_data/2024-02-27-12-32-07/data.parquet"
    )  #
    return df


df = load_data()


def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        text
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud)
    ax.axis("off")
    return fig


def fetch_corpus(category, df):
    category_df = df[df["price_category"] == category]
    combined_text = " ".join(category_df["text"].tolist())
    return combined_text


st.title("Wordcloud Generator")

# Category selection
category = st.selectbox("Select a category:", ("Expensive", "Cheap", "Average"))

corpus = fetch_corpus(category, df)

# Implement word filter mechanism to accept multiple words
filter_words = st.text_input(
    "Enter words to filter out (separated by commas) and regenerate wordcloud:"
)

if filter_words:
    # Split the filter_words by commas, strip spaces, and convert to lowercase for case-insensitive comparison
    filter_words_list = [word.strip().lower() for word in filter_words.split(",")]
    # Filter out the words
    filtered_corpus = " ".join(
        [word for word in corpus.split() if word.lower() not in filter_words_list]
    )
else:
    # If no filter words are provided, use the original corpus
    filtered_corpus = corpus

# Display the wordcloud
st.write("Generated Wordcloud:")
fig = generate_wordcloud(filtered_corpus)  # Generate wordcloud with filtered corpus
st.pyplot(fig)  # Display the figure

# UK Rightmove
Data Engineering batch processing project extracting and enhancing Rightmove data for downstream analysis and ML use cases. This project showcases a comprehensive data engineering pipeline focused on batch processing of Rightmove data, a prominent UK real estate website. The aim is to extract valuable information from Rightmove, enhance it for downstream analysis, and prepare it for machine learning applications. The pipeline encompasses various stages from data extraction to serving a front-end application, demonstrating a holistic approach to handling and presenting real estate data.

# Productionized Web Scraping with Scrapy and Scrapyd 
![image](https://github.com/alexandergirardet/london_rightmove/assets/123730494/49c28915-f7ca-4c6b-8512-558dfa4af9b3)
Overview: Utilizing Scrapy, a powerful Python framework for web crawling, the project starts by extracting real estate listings from Rightmove. Scrapy's asynchronous processing capability ensures efficient data retrieval.
Server Hosting with Scrapyd: To manage and schedule our Scrapy spiders, we use Scrapyd, a service for running Scrapy spiders. It allows us to deploy and run spiders remotely, providing a scalable solution for continuous data extraction.
Processing Data with Apache Beam and Walk Score Feature
Data Processing: Once extracted, the data undergoes processing using Apache Beam, a model for defining both batch and streaming data-parallel processing pipelines. This step involves cleaning, transformation, and enhancement of the data.
Walk Score Feature: A key enhancement is the addition of a "walk score", which rates the walkability of a property's location. This score is calculated using various datasets and adds significant value to the data set for potential homebuyers and renters.
FastAPI Backend
API Development: To make the processed data accessible, we've developed a backend using FastAPI. This modern, fast web framework for building APIs with Python 3.7+ ensures quick and efficient data retrieval.
API Capabilities: The backend supports various API calls, allowing users to query the dataset based on different criteria such as location, price range, and property type.
MongoDB for Storing Unstructured JSON Data
Database Choice: MongoDB, a NoSQL database, is used for storing the extracted and processed data. Its flexible schema for JSON-like documents makes it an ideal choice for our unstructured data.
Data Accessibility: The use of MongoDB facilitates easy access and manipulation of data, which is crucial for both the backend API and the front-end application.
Streamlit Front-End
Interactive Dashboard: To visualize the data, we employ Streamlit, a Python library that turns data scripts into shareable web apps.
User-Friendly Interface: The Streamlit dashboard offers an interactive and user-friendly interface for exploring the Rightmove data, with features such as property search, filtering, and visualization.
Orchestrating Batch Ingestion with Airflow
Workflow Management: Apache Airflow is used for orchestrating the entire batch ingestion process. This platform allows us to programmatically author, schedule, and monitor workflows.
Automation and Monitoring: With Airflow, each component of the pipeline, from data extraction to front-end serving, is automated and monitored, ensuring reliability and efficiency of the data processing.

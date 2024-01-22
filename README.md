# UK Rightmove
Data Engineering batch processing project extracting and enhancing Rightmove data for downstream analysis and ML use cases. This project showcases a comprehensive data engineering pipeline focused on batch processing of Rightmove data, a prominent UK real estate website. The aim is to extract valuable information from Rightmove, enhance it for downstream analysis, and prepare it for machine learning applications. The pipeline encompasses various stages from data extraction to serving a front-end application, demonstrating a holistic approach to handling and presenting real estate data.

# Productionized Web Scraping with Scrapy and Scrapyd 
![image](https://github.com/alexandergirardet/london_rightmove/assets/123730494/49c28915-f7ca-4c6b-8512-558dfa4af9b3)
Utilizing Scrapy, a powerful Python framework for web crawling, the project starts by extracting real estate listings from Rightmove. Scrapy is built on Twisted, an asynchronous networking framework. Asynchronous processing means Scrapy can handle large amounts of requests and data without blocking or waiting for each request to be completed. This allows us to extract a large amount of data in a short amount of time. We are also able to configure rate limits to avoid overloading Rightmove's servers.

## Server Hosting with Scrapy
To manage and schedule our Scrapy spiders, we use Scrapyd, a service for running Scrapy spiders. It allows you to deploy your Scrapy projects and control their spiders using a simple JSON API, providing a scalable solution for continuous data extraction. Additionally, it provides a Web UI for better visibility and monitoring into our extraction jobs, using logs. 

## Loading data to MongoDB
Within our cluster and MongoDB service is running allowing us to store unstructured JSON data in a raw properties database. This enables ML use cases by providing raw access to data. 

# Processing Data with Apache Beam and Walk Score Feature.

Once extracted, the data undergoes processing using Apache Beam, a model for defining both batch and streaming data-parallel processing pipelines. This step involves cleaning, transformation, and enhancement of the data in a batch process using the DirectRunner. 

## Walk Score
The research paper [Neighborhood Walkability and Housing Prices: A Correlation Study](https://www.mdpi.com/2071-1050/12/2/593) implies that neighbourhood walkability is highly predictive of housing prices. Adding this feature within the batch process could enable more accurate model predictions in downstream analysis. For further insights into how the walk score is calculated consult the notebooks associated with this project. 

# FastAPI Backend
To make the processed data accessible, we've developed a backend using FastAPI. This modern, fast web framework for building APIs with Python ensures quick and efficient data retrieval and decouples responsibilities between Streamlit for data visualization and FastAPI for processing data into a form that can be used for visualization.

# Streamlit Frontend
To visualize the data, we employ Streamlit, a Python library that turns data scripts into shareable web apps. The Streamlit dashboard offers an interactive and user-friendly interface for exploring the Rightmove data, with features such as property search, filtering, and visualization.

# Database
Database Choice: MongoDB, a NoSQL database, is used for storing the extracted and processed data. Its flexible schema for JSON-like documents makes it an ideal choice for our unstructured data.

# Workflow Management
Apache Airflow is used for orchestrating the entire batch ingestion process. This platform allows us to programmatically author, schedule, and monitor workflows.
With Airflow, each component of the pipeline, from data extraction to feature processing, is automated and monitored, ensuring reliability and efficiency of the data processing.

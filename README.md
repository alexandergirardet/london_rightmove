# Rightmove Rental Prediction System

In recent months, I've deepened my expertise in creating machine learning (ML) systems through comprehensive study and application of three pivotal areas: Data Engineering, MLOps, and ML Engineering, as structured by the Data Talk Club courses. My project, the Rightmove Rental Prediction System, encapsulates this journey, demonstrating a cohesive application of these skills.

The essence of this project lies in its comprehensive architecture, designed to predict rental prices with precision. It integrates:

1. **Data Engineering** through an asynchronous web scraper and batch ingestion pipelines, enabling efficient data extraction and preprocessing.
2. **ML Engineering** with a focus on model training and feature engineering, including the development of an innovative "Walk Score."
3. **MLOps** by implementing monitoring practices to ensure the system's reliability and performance over time.

### **Project Components**

1. **Extraction and Data Processing Pipeline**: Automated to handle large-scale data extraction, cleaning, and preparation.
2. **ML Training Pipeline**: Designed for iterative experimentation and training, leveraging a RandomForest model among others, to identify the most effective prediction method.
3. **MLOps Monitoring Pipeline**: Ensures model performance remains optimal through continuous monitoring for data drift and other potential issues.
4. **Model Serving API**: Utilizes FastAPI for efficient model deployment, allowing real-time predictions.
5. **Visualization Dashboard**: Built with Streamlit and Grafana, offering insightful data visualizations and monitoring dashboards to track system performance and data quality.

### **Infrastructure and Deployment**

My approach combines DevOps and software engineering principles, employing Terraform for infrastructure management and Docker Compose for containerization, across both AWS and GCP platforms. This dual-cloud strategy not only leverages the strengths of both services but also optimizes costs through their free tier options.

### **ML and MLOps Implementation**

The project showcases my ML and MLOps expertise through the development of a RandomForest model, enhanced by a unique feature, the Walk Score, to improve predictive accuracy. MLFlow serves as the backbone for experiment tracking and model registry, facilitating the model's evolution and serving.

### **Data Extraction and Processing**

![Data Extraction Pipeline](/static/images/Processing_pipeline_rightmove.png)


Choosing Rightmove, a leading UK property listing site, as the data source, I developed a Scrapy spider deployed on a Scrapyd server. This setup enhances control over scraping activities and integrates seamlessly with Airflow for orchestration, ensuring ethical data usage and compliance with best practices.

Data storage is managed through PostgreSQL and MongoDB, supporting structured and unstructured data, respectively. This configuration not only facilitates efficient data management but also integrates a custom Beam job to compute the Walk Score for enhanced model input.

### **ML Training with MLFlow**

![ML Training pipeline](/static/images/model_training_pipeline.png)

For the ML training component, MLFlow played a critical role as a central hub for experiment tracking, model versioning, and serving. This tool allowed for a systematic approach to managing the lifecycle of machine learning models. Here's how it was integrated into the workflow:

- **Experiment Tracking**: Every training run, along with its parameters, metrics, and outcomes, was logged in MLFlow. This facilitated a comprehensive analysis of each experiment, enabling quick iteration over models to find the best performing ones based on Root Mean Squared Error (RMSE) metrics.
- **Model Registry**: The most promising models, particularly the RandomForest model which outperformed others including XGBoost, were registered in MLFlow's model registry. This registry acted as a repository, making it simple to version, store, and access models for deployment.
- **Model Serving**: MLFlow also streamlined the deployment process. The serving component fetched the latest and most effective model version directly from the registry, ensuring that the prediction service always utilized the best available model.

The use of MLFlow not only brought organization and efficiency to the model training process but also ensured transparency and reproducibility, which are essential for collaboration and continuous improvement in ML projects.

## **DevOps and Scraper Monitoring**

The Rightmove Rental Prediction System employs a focused approach to monitor its web scraping operations, leveraging Grafana and PostgreSQL for a streamlined and effective oversight.

### **Monitoring Framework**

**Grafana Dashboard**: Provides real-time visualization of critical metrics such as success rates, error counts, and response times. This dashboard enables quick identification of performance issues or errors in the web scraping process.

**PostgreSQL**: Acts as the storage backbone for logging detailed metrics from each scraping session. This includes timestamps, counts of extracted records, and error logs, offering a comprehensive view for analysis and troubleshooting.

### **Key Objectives**

- **Efficiency and Error Management**: Monitoring ensures the scraper runs efficiently, with a quick response to any errors or bottlenecks.
- **Compliance and Rate Limiting**: Keeps the scraping activities within ethical and legal boundaries by tracking request rates and adherence to site policies.

### **DevOps Integration**

The setup integrates seamlessly with our DevOps practices, with Grafana alerts configured to trigger automated actions or notifications for immediate attention, ensuring the system's robustness and reliability.

#### System Monitoring
![Extraction Monitoring](/static/images/scrapy_monitoring.png)

System Monitoring of Scrapy Sessions

## **MLOps**

![MLOps Diagram](/static/images/mlops_pipeline.png)

Understanding and mitigating concept drift and data drift are critical for maintaining the performance of ML models in production. Here’s how these challenges were approached:

- **Concept Drift**: This occurs when the statistical properties of the target variable, which the model is trying to predict, change over time. This can degrade the model's performance because the patterns the model learned during training may no longer apply. To detect concept drift, the monitoring pipeline employed statistical tests and comparisons between predictions and actual outcomes over time. When significant drift was detected, a model retraining workflow was triggered, incorporating new data to adapt the model to the current reality.
- **Data Drift**: Data drift refers to changes in the input data's distribution. It's crucial to monitor because even if the target variable's distribution remains the same, changes in input data can lead to poor model performance. The project utilized Evidently to monitor key features' distributions, comparing incoming data against a historical baseline (the golden dataset). Alerts were set up to notify when data drift exceeded predefined thresholds, prompting an evaluation to determine if model retraining or adjustment in data preprocessing steps was necessary.

#### ML Model Monitoring
![Model Monitoring](/static/images/model_monitoring.png)

MLOps monitoring of Data and Concept Drift

### Addressing change

Grafana enables automated actions. In the case our model’s prediction performance drops below a certain threshold, we will trigger an automatic retraining of the model on new data that include the new patterns. This should ensure our model is up to date in an automated fashion.

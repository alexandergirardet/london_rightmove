# Start from a base image with Python installed
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app

# Install mlflow and dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default MLflow server port
EXPOSE 5001

# Start the MLflow server when the container starts
CMD mlflow server --backend-store-uri $MLFLOW_BACKEND_STORE_URI --default-artifact-root $MLFLOW_ARTIFACTS_DESTINATION --host 0.0.0.0 --port 5001

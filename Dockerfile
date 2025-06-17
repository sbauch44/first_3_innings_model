# Use an official Python runtime as a parent image
FROM python:3.11-slim-bullseye

# Set environment variables
# Good practice for Python logging in Docker
ENV PYTHONUNBUFFERED=1
# Placeholder for base file path (local or S3). Can be overridden at runtime.
# Example for local: ENV BASE_FILE_PATH="/app/clean_data/"
# Example for S3: ENV BASE_FILE_PATH="s3://your-s3-bucket-name/data/"
ENV BASE_FILE_PATH="s3://your-s3-bucket-name/data/"
# Placeholder for model path (local or S3). Can be overridden at runtime.
# Example for local: ENV MODEL_PATH="/app/baseball_simulator/" (if models are there) or "/app/models/"
# Example for S3: ENV MODEL_PATH="s3://your-s3-bucket-name/models/"
ENV MODEL_PATH="s3://your-s3-bucket-name/models/"
# Placeholder for scaler path (local or S3). Can be overridden at runtime.
# Example for local: ENV SCALER_PATH="/app/baseball_simulator/" (if scalers are there) or "/app/models/"
# Example for S3: ENV SCALER_PATH="s3://your-s3-bucket-name/models/"
ENV SCALER_PATH="s3://your-s3-bucket-name/models/"
# Placeholder for Fangraphs cookies JSON string. Must be valid JSON.
ENV FANGRAPHS_COOKIES_JSON="{}"
# Placeholder for Fangraphs headers JSON string. Must be valid JSON.
ENV FANGRAPHS_HEADERS_JSON="{}"
# Placeholder for AWS region, useful for boto3 if not configured elsewhere
ENV AWS_REGION="your-aws-region"

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
# --no-cache-dir reduces image size by not storing the pip download cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container at /app
# This includes the main simulator logic, models, and scalers if they are within this directory
COPY baseball_simulator/ /app/baseball_simulator/

# Copy the clean_data directory into the container at /app
# This is included as per requirements, though in a real-world S3-centric setup,
# this might be omitted if all data is indeed read from/written to S3.
# If BASE_FILE_PATH is set to a local path like "/app/clean_data/", this copy is essential.
COPY clean_data/ /app/clean_data/

# Define the command to run your application
# This is a sensible default and can be overridden when running the container.
CMD ["python", "baseball_simulator/main_daily_trigger.py"]

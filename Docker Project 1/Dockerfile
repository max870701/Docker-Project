# Using a lightweight base image
FROM python:3.9-slim

# Assign the working directory inside the container
WORKDIR /app

# Create Model Directory
RUN mkdir -p /app/models

# Copy required files first
COPY requirements.txt .

# Install the dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       # Add any necessary ubuntu packages here
    && pip3 install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application
COPY titanic-pred.py .
COPY datasets/ ./datasets/

# Run the application
CMD ["python3", "titanic-pred.py"]
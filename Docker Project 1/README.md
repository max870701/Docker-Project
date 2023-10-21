# Titanic Prediction with Docker
## Introduction
The goal of this project is to create a machine learning model that predicts the survival of passengers on the Titanic. The model is then deployed using Docker and Flask. The dataset used for this project is the Titanic dataset from Kaggle. The dataset contains information about the passengers on the Titanic. The dataset can be found [here](https://www.kaggle.com/c/titanic/data).

## Build Up Docker Image with Dockerfile
The Dockerfile is used to build up the Docker image. The Docker image is built up from the base image of Python 3.9. To build up the docker image, we use the following commands:
```docker build -t titanic-pred:1.0 .```

## Run Docker Container
To run the Docker container, we use the following command:
```docker run --name first_run titanic-pred:1.0```

## Check the Docker Process
To check the Docker process, we use the following command:
```docker ps```
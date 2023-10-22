# Titanic Predication Web Application with Docker
## Introduction
The goal of this project is to create a web application that predicts the survival of passengers on the Titanic. The dataset can be found at [Kaggle](https://www.kaggle.com/c/titanic/data).
## Frontend
* The fronted is built with Streamlit.
* User can enter the features of a passenger on the Titanic, and click the "Predict" button to get the prediction result.
## Backend
* The backend is built with FastAPI.
* Receive the features of a passenger from the frontend, and send the prediction result back to the frontend.
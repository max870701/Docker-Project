import streamlit as st
import pandas as pd
import numpy as np
import requests


header = st.container()
upload = st.container()
dataset = st.container()
eda = st.container()
preprocessing = st.container()
cleaned_dataframe = st.container()
features_target = st.container()
select_model = st.container()
training_model = st.container()
download_model = st.container()

# models = {
#     "Random Forest Regressor": "RandomForestRegressor",
#     "Random Forest Classifier": "RandomForestClassifier",
#     "Linear Regression": "LinearRegression",
#     "Logistic Regression": "LogisticRegression"
# }

# Initialize session_state variables
if 'models' not in st.session_state:
    st.session_state.models = {
        "Random Forest Regressor": "RandomForestRegressor",
        "Random Forest Classifier": "RandomForestClassifier",
        "Linear Regression": "LinearRegression",
        "Logistic Regression": "LogisticRegression"
    }
if 'train_data' not in st.session_state:
    st.session_state.train_data = None
if 'test_data' not in st.session_state:
    st.session_state.test_data = None
if 'preprocess_response' not in st.session_state:
    st.session_state.preprocess_response = None
if 'train_response' not in st.session_state:
    st.session_state.train_response = None
if 'features' not in st.session_state:
    st.session_state.features = []
if 'target' not in st.session_state:
    st.session_state.target = ''
if 'task' not in st.session_state:
    st.session_state.task = ''
if 'model_name' not in st.session_state:
    st.session_state.model_name = ''
if 'model_id' not in st.session_state:
    st.session_state.model_id = ''
if 'evaluation' not in st.session_state:
    st.session_state.evaluation = {}
if 'file_type' not in st.session_state:
    st.session_state.file_type = ''
if 'download_link' not in st.session_state:
    st.session_state.download_link = ''
if 'train_info' not in st.session_state:
    st.session_state.train_info = {}
# if 'train_model' not in st.session_state:
#     st.session_state.train_model = False

with header:
    st.header("Welcome to the Machine Learning Web App !")
    st.title("This is a simple web app to streamline your ML workflow.")

def convert_special_floats_to_str(data):
    """Convert special float values (NaN, Infinity, -Infinity) to their string representations."""
    if isinstance(data, (list, tuple)):
        return [convert_special_floats_to_str(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_special_floats_to_str(value) for key, value in data.items()}
    elif isinstance(data, float):
        if np.isnan(data):
            return "NaN"
        elif data == np.inf:
            return "Infinity"
        elif data == -np.inf:
            return "-Infinity"
        else:
            return data
    else:
        return data

def convert_str_to_special_floats(data):
    """Convert special float strings back to their float representations."""
    if isinstance(data, (list, tuple)):
        return [convert_str_to_special_floats(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_str_to_special_floats(value) for key, value in data.items()}
    elif data == "NaN":
        return np.nan
    elif data == "Infinity":
        return np.inf
    elif data == "-Infinity":
        return -np.inf
    else:
        return data

@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        data = pd.read_csv(file)
    elif file.name.endswith('.parquet'):
        data = pd.read_parquet(file)
    else:
        st.error('File format is not supported.')
        return None
    # data = data.astype({col: float for col in data.select_dtypes(include=[np.float64]).columns})
    return data

with upload:
    st.header("Upload your dataset")
    # 1 & 2
    train_file = st.file_uploader("Choose a CSV training set", type=["csv", "parquet"])
    test_file = st.file_uploader("Choose a CSV test set (optional)", type=["csv", "parquet"])

if train_file:
    train_data = load_data(train_file)
    if test_file:
        test_data = load_data(test_file)
    else:
        test_data = None

    with dataset:
        st.header("Dataset")
        st.write("Training data")
        st.write(train_data)
        if test_data is not None:
            st.write("Test data")
            st.write(test_data)
    # 3
    with eda:
        st.header("Exploratory Data Analysis")
        st.write("Training data")
        st.write(train_data.describe(include="all"))
        if test_data is not None:
            st.write("Test data")
            st.write(test_data.describe(include="all"))

    # data clearning
    with preprocessing:
        st.header("Data Preprocessing")
        if st.button("Preprocessing"):
            # Convert train_data and test_data using the function
            processed_train_data = convert_special_floats_to_str(train_data.to_dict(orient="records"))
            processed_test_data = convert_special_floats_to_str(test_data.to_dict(orient="records")) if st.session_state.test_data is not None else None

            # Send the request
            st.session_state.preprocess_response = requests.post("http://127.0.0.1:8000/preprocess/", 
                                                json={
                                                    "data": processed_train_data,
                                                    "test_data": processed_test_data
                                                })
            
            if st.session_state.preprocess_response.status_code == 200:
                st.session_state.train_data = pd.DataFrame(st.session_state.preprocess_response.json()["data"])
                st.session_state.test_data = pd.DataFrame(st.session_state.preprocess_response.json()["test_data"])
                st.write("Data has been preprocessed!")
            else:
                st.write("Error during preprocessing.")

    with cleaned_dataframe:
        st.header("Cleaned DataFrame")
        st.write("Cleaned Training data")
        st.write(st.session_state.train_data)
        st.write("Cleaned Test data")
        st.write(st.session_state.test_data)
    # 4
    with features_target:
        st.header("Features and Target")
        # features = st.multiselect("Choose your input features", train_data.columns.tolist())
        # target = st.selectbox("Choose your target column", [col for col in train_data.columns if col not in features])
        st.session_state.features = st.multiselect(
                                    "Choose your input features",
                                    st.session_state.train_data.columns.tolist()
                                    )
        st.session_state.target = st.selectbox(
                                    "Choose your target column",
                                    [col for col in st.session_state.train_data.columns if col not in st.session_state.features],
                                    index=st.session_state.train_data.columns.tolist().index(st.session_state.target) if st.session_state.target else 0
                                    )
    # 5
    with select_model:
        st.header("Select your model")
        st.session_state.task = st.selectbox("Choose your task type", ["classification", "regression"])
        st.session_state.model_name = st.selectbox("Choose your model", list(st.session_state.models.keys()))
    
    # 6 & 7 & 8
    with training_model:
        st.header("Train your model")
        if st.button("Train"):
            st.session_state.train_response = requests.post("http://127.0.0.1:8000/train/", 
                                    json={
                                        "data": st.session_state.train_data.to_dict(orient="records"), 
                                        "features": st.session_state.features, 
                                        "target": st.session_state.target, 
                                        "task": st.session_state.task,
                                        "model_name": st.session_state.model_name,
                                        "test_data": st.session_state.test_data.to_dict(orient="records") if st.session_state.test_data is not None else None
                                    })
            
            if st.session_state.train_response.status_code == 200:
                st.session_state.model_id = st.session_state.train_response.json()["model_id"]
                st.session_state.evaluation = st.session_state.train_response.json()["evaluation"]
                st.write(f"Model ID: {st.session_state.model_id}")
                st.write(f"Evaluation metrics for {st.session_state.model_name}:")
                for metric_name, metric_value in st.session_state.evaluation.items():
                    st.write(f"{metric_name}: {metric_value:.2f}")
            else:
                st.write("Error with the API call")

    with download_model:
        st.header("Download your model")
        st.session_state.file_type = st.selectbox("Choose a file type", ["pkl", "joblib"])
        if st.button("Download"):
            st.session_state.download_link = f"http://127.0.0.1:8000/download_model/{st.session_state.model_id}/{st.session_state.file_type}"
            st.write(f"[Download .{st.session_state.file_type} file]({st.session_state.download_link})")
        if st.button("Save Training Info"):
            train_info_response = requests.get(f"http://127.0.0.1:8000/get_train_info/{st.session_state.model_id}")
            st.session_state.train_info = train_info_response.json()
            st.write("Training Information:")
            st.write(st.session_state.train_info)
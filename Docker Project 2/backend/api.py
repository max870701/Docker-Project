from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report
import pandas as pd
import io
from uuid import uuid4
import pickle
import joblib
from preprocessing import data_preprocessing
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

models = {
    'RandomForestRegressor': RandomForestRegressor(),
    'RandomForestClassifier': RandomForestClassifier(),
    'LinearRegression': LinearRegression(),
    'LogisticRegression': LogisticRegression()
}

metrics = {
    "classification": {
        "accuracy_score": accuracy_score
    },
    "regression": {
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error
    }
}

models_saved = {}
train_info_saved = {}

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

@app.post("/upload/")
async def upload(file: UploadFile):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    return df.to_dict(orient="records")

@app.post("/train/")
async def train(data: dict):
    df = pd.DataFrame(data["data"])
    X = df[data["features"]]
    y = df[data["target"]]

    model = models.get(data["model_name"], RandomForestClassifier())

    test_df = pd.DataFrame(data["test_data"])
    X_test = test_df[data["features"]]
    y_test = test_df[data["target"]]
    
    model.fit(X, y)
    predictions = model.predict(X_test)

    evaluation = {}
    for metric_name, metric_func in metrics[data["task"]].items():
        evaluation[metric_name] = metric_func(y_test, predictions)

    # Generate a unique ID for the model and save it
    model_id = str(uuid4())
    # Save model
    models_saved[model_id] = {
        "model": model,
        "evaluation": evaluation
    }
    # Save training info
    train_info_saved[model_id] = {
        "model_name": data["model_name"],
        "features": data["features"],
        "target": data["target"],
        "evaluation": evaluation
    }

    return {"model_id": model_id, "evaluation": evaluation}

@app.get("/download_model/{model_id}/{file_type}")
async def download_model(model_id: str, file_type: str):
    model = models_saved.get(model_id, {}).get("model")
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    file_name = f"model_{model_id[:6]}.{file_type}"
    if file_type == "pkl":
        with open(f"./models/{file_name}", "wb") as f:
            pickle.dump(model, f)
    elif file_type == "joblib":
        joblib.dump(model, f"./models/{file_name}")
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    return FileResponse(f"./models/{file_name}", filename=file_name)

@app.get("/get_train_info/{model_id}")
async def get_train_info(model_id: str):
    return train_info_saved.get(model_id, {})

@app.post("/preprocess/")
async def preprocess_data(data_payload: dict):
    converted_data = convert_str_to_special_floats(data_payload["data"])
    converted_test_data = convert_str_to_special_floats(data_payload["test_data"]) if data_payload["test_data"] else None
    
    tmp = pd.DataFrame(converted_data)
    if converted_test_data:
        data = tmp
        test_data = pd.DataFrame(converted_test_data)
    else:
        data, test_data = train_test_split(tmp, test_size=0.2)

    processed_data, processed_test_data = data_preprocessing(data, test_data)

    return {
        "data": convert_special_floats_to_str(processed_data.to_dict(orient="records")),
        "test_data": convert_special_floats_to_str(processed_test_data.to_dict(orient="records")) if processed_test_data is not None else None
    }
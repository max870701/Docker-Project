from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI()

# 載入模型
model = joblib.load("titanic-model.joblib")

class InputData(BaseModel):
    features: list

@app.get("/")
async def root():
    return {"message": "Welcome to the model API!"}

@app.post("/predict/")
async def get_prediction(data: InputData):
    try:
        result = model.predict([data.features])
        return {"prediction": result[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
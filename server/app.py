from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
import joblib
import numpy as np
from pydantic import BaseModel
import json
import os
from datetime import datetime
from train import train_model

app = FastAPI()
client = MongoClient('mongo', 27017)
db = client.test_database
collection = db.test_collection

# Load model and model info
model = joblib.load("model.pkl")

# Create model_info.json if it doesn't exist
if not os.path.exists("model_info.json"):
    model_info = {
        "model_type": "DecisionTreeClassifier",
        "accuracy": 0.95,
        "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {},
        "test_size": 0.2,
        "random_state": 42
    }
    with open("model_info.json", "w") as f:
        json.dump(model_info, f)

class Item(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class TrainingParams(BaseModel):
    model_type: str
    test_size: float
    random_state: int
    params: dict

target_names = ['setosa', 'versicolor', 'virginica']


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/add/{fruit}")
async def add_fruit(fruit: str):
    id = collection.insert_one({"fruit": fruit}).inserted_id 
    return {"id": str(id)}

@app.get("/list")
async def list_fruits():
    return {"results": list(collection.find({}, {"_id": False}))}

@app.post("/predict")
def predict(item: Item):
    X_new = np.array([[item.sepal_length, item.sepal_width, item.petal_length, item.petal_width]])
    prediction = model.predict(X_new)
    pred_class = target_names[int(prediction[0])]
    return {"prediction": pred_class}

@app.get("/model-info")
def get_model_info():
    try:
        with open("model_info.json", "r") as f:
            model_info = json.load(f)
        return model_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model info: {str(e)}")

@app.post("/train")
def train_new_model(params: TrainingParams):
    try:
        # Call train_model function with provided parameters
        result = train_model(
            model_type=params.model_type,
            test_size=params.test_size,
            random_state=params.random_state,
            params=params.params
        )
        
        # Reload the model
        global model
        model = joblib.load("model.pkl")
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")
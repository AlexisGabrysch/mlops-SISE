from fastapi import FastAPI
from pymongo import MongoClient
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()
client = MongoClient('mongo', 27017)
db = client.test_database
collection = db.test_collection

model = joblib.load("model.pkl")

class Item(BaseModel) :
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

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
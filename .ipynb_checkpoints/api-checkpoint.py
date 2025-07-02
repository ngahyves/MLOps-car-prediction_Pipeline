# api.py

# --- Step 1 : Import the libraries ---

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np


# --- Step 2 : Create the API and load the model ---
app = FastAPI()

# loading the trained pipeline
model_pipeline = joblib.load("models/best_model.pkl")


# --- Step 3 : Data input format ---
class CarFeatures(BaseModel):
    manufacturer: str
    model_reduced: str
    transmission: str
    fuel: str
    title_status: str
    age: int
    odometer: float


# --- Step 4 : Prediction endpoint ---
@app.post("/predict")
def predict(features: CarFeatures):
    
    # 1. Convert the input data into a DataFrame
    input_df = pd.DataFrame([features.dict()])

    # 2. Prediction of price_log
    prediction_log = model_pipeline.predict(input_df)

    # 3. Prediction on real prices
    prediction_real_scale = np.expm1(prediction_log)

    # 4. Transform predictions into JSON
    return {"predicted_price": round(prediction_real_scale[0], 2)}


# --- Step 5 Welcoming endpoint.---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Car Price Prediction API!"}
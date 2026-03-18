from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import numpy as np
import os

# Get the folder where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)

class InputData(BaseModel):
    aqi: float
    age: float
    outdoor_hours: float
    smoking: int
    exercise: int
    condition: int

def get_aqi_category(aqi):
    if aqi <= 50:  return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"

@app.get("/")
def serve_home():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.post("/predict")
def predict(data: InputData):
    features = np.array([[
        data.aqi, data.age, data.smoking,
        data.outdoor_hours, data.exercise, data.condition
    ]])

    prediction = model.predict(features)[0]
    proba      = model.predict_proba(features)[0]
    confidence = round(max(proba) * 100, 1)

    labels = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}

    return {
        "prediction":     int(prediction),
        "label":          labels[prediction],
        "confidence":     confidence,
        "proba_low":      round(proba[0] * 100, 1),
        "proba_moderate": round(proba[1] * 100, 1),
        "proba_high":     round(proba[2] * 100, 1),
        "aqi_category":   get_aqi_category(data.aqi),
    }
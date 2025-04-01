"""
fast.py - API for predicting F1 lap times
"""

import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from pathlib import Path
from models.ml_logic.model import load_pipeline
import joblib

app = FastAPI(
    title="F1 Lap Time Prediction API",
    description="API for lap time prediction in Formula 1 races",
    version="1.0"
)

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

def load_model():
    """Load the trained model"""
    model_path = Path(__file__).parent / "raw_data" / "pipeline_model.pkl"
    return joblib.load(model_path)

@app.get("/predict")
def predict(Driver: str, LapNumber: float, Stint: float, Compound: str,
            TyreLife: float, Position: float, AirTemp: float, Humidity: float,
            Pressure: float, Rainfall: float, TrackTemp: float, Event_Year: int,
            GrandPrix: str):
    """
    Predicts lap time based on race parameters

    Parameters:
    - Driver: Driver's name (e.g. 'VER')
    - LapNumber: Lap number
    - Stint: Stint number
    - Compound: Tire compound (e.g. 'SOFT')
    - TyreLife: Tyre life in laps
    - Position: Current position in the race
    - AirTemp: Air temperature (°C)
    - Humidity: Relative humidity (%)
    - Pressure: Atmospheric pressure (hPa)
    - Rainfall: Rain (0 = no, 1 = yes)
    - TrackTemp: Track temperature (°C)
    - Event_Year: Year of the event
    - GrandPrix: Name of the Grand Prix (e.g. 'Bahrain')

    Returns:
    - predicted_lap_time: Predicted lap time in seconds
    """

    input_data = {
        'Driver': [Driver],
        'LapNumber': [LapNumber],
        'Stint': [Stint],
        'Compound': [Compound],
        'TyreLife': [TyreLife],
        'Position': [Position],
        'AirTemp': [AirTemp],
        'Humidity': [Humidity],
        'Pressure': [Pressure],
        'Rainfall': [Rainfall],
        'TrackTemp': [TrackTemp],
        'Event_Year': [Event_Year],
        'GrandPrix': [GrandPrix]
    }


    X_pred = pd.DataFrame(input_data)

    model = load_model()
    prediction = model.predict(X_pred)
    return {"predicted_lap_time": float(prediction[0])}


@app.get("/")
def root():
    return {'message': 'Welcome to the F1 lap time prediction API'}

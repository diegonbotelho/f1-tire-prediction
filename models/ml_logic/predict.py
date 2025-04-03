"""
Load the model and make predictions
"""
from pathlib import Path
import pandas as pd
from model import load_pipeline

def load_trained_model():
    """Load the trained pipeline"""
    model_path = Path(__file__).parent.parent.parent / "raw_data" / "model_pipeline.pkl"
    return load_pipeline(model_path)

def predict_lap_time(input_data: pd.DataFrame):
    """
    Makes predictions using the loaded model
    Args:
        input_data: DataFrame in the same format as the training
    """
    pipeline = load_trained_model()
    return pipeline.predict(input_data)

# Example of use:
if __name__ == "__main__":
    # Example data (replace with your real data)
    sample_data = pd.DataFrame({
        'Driver': ['VER'],
        'LapNumber': [2.0],
        'Stint': [1.0],
        'Compound': ['SOFT'],
        'TyreLife': [5.0],
        'Position': [2.0],
        'AirTemp': [23.8],
        'Humidity': [26.0],
        'Pressure': [1010.4],
        'Rainfall': [0],
        'TrackTemp': [29.0],
        'Event_Year': [2022],
        'GrandPrix': ['Bahrain']
    })

    prediction = predict_lap_time(sample_data)
    print(f"⏱️ Expected return time: {prediction[0]:.2f} seconds")

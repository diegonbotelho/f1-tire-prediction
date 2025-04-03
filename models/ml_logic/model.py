"""
Module for model loading and predictions
"""
import joblib
import pandas as pd
from pathlib import Path

DEFAULT_MODEL_FILENAME = 'model_pipeline.pkl'

def load_pipeline(filename=DEFAULT_MODEL_FILENAME):
    """Loads the pipeline from the raw_data folder correctly"""
    load_path = Path(__file__).parent.parent.parent / "raw_data" / filename

    return joblib.load(load_path)

def predict(input_data):
    """
    Makes predictions using the loaded pipeline.
    Args:
        input_data (pd.DataFrame): Input data in the same format as the training
    Returns:
        array: Model predictions
    """
    pipeline = load_pipeline()
    return pipeline.predict(input_data)

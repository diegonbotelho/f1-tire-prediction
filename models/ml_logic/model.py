"""
Module for model loading and predictions
"""
import joblib
import pandas as pd
from pathlib import Path

DEFAULT_MODEL_FILENAME = 'pipeline_model.pkl'

def load_pipeline(filename=DEFAULT_MODEL_FILENAME):
    """Loads the pipeline from the raw_data folder correctly"""
    load_path = Path(__file__).parent.parent.parent / "raw_data" / filename

    return joblib.load(load_path)

def predict(input_data):
    """
    Faz previsões usando o pipeline carregado
    Args:
        input_data (pd.DataFrame): Dados de entrada no mesmo formato do treino
    Returns:
        array: Previsões do modelo
    """
    pipeline = load_pipeline()
    return pipeline.predict(input_data)

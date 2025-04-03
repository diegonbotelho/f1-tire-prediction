"""
Integration test for the F1 Lap Time Prediction system
Tests data loading, cleaning, preprocessing, model training, saving, loading, and prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import pytest
import os
import sys

# Import your modules
from models.ml_logic.data import get_data, clean_data
from models.ml_logic.preprocessor import create_and_fit_pipeline, save_pipeline
from models.ml_logic.model import load_pipeline, predict

# Constants for testing
TEST_MODEL_PATH = Path(__file__).parent.parent / "raw_data" / "test_model_pipeline.pkl"
SAMPLE_DATA_SHAPE = (10, 17)  # Expected shape after cleaning

def test_data_loading_and_cleaning():
    """Test data loading and cleaning functions"""
    print("\n=== Testing data loading and cleaning ===")

    # Test data loading
    raw_data = get_data()
    assert isinstance(raw_data, pd.DataFrame), "Data should be a DataFrame"
    assert len(raw_data) > 0, "Data should not be empty"
    print("✓ Data loading successful")

    # Test data cleaning
    cleaned_data = clean_data(raw_data.copy())
    assert isinstance(cleaned_data, pd.DataFrame), "Cleaned data should be a DataFrame"
    assert cleaned_data.shape[1] == SAMPLE_DATA_SHAPE[1], f"Expected {SAMPLE_DATA_SHAPE[1]} columns after cleaning"
    assert not cleaned_data['Position'].isna().any(), "Position should have no NA values"
    assert cleaned_data['Rainfall'].isin([0, 1]).all(), "Rainfall should be 0 or 1"
    print("✓ Data cleaning successful")

    return cleaned_data

def test_preprocessing_and_training(cleaned_data):
    """Test pipeline creation and training"""
    print("\n=== Testing preprocessing and training ===")

    # Prepare data
    X = cleaned_data.drop(columns=['Sector1Time', 'Sector2Time', 'Sector3Time', 'LapTime'])
    y = cleaned_data['LapTime']

    # Test pipeline creation and training
    pipeline = create_and_fit_pipeline(X, y)
    assert hasattr(pipeline, 'predict'), "Pipeline should have predict method"
    assert len(pipeline.named_steps) == 2, "Pipeline should have 2 steps (preprocessor + model)"

    # Test predictions on training data
    y_pred = pipeline.predict(X.head())
    assert len(y_pred) == 5, "Should return prediction for each input row"
    assert all(isinstance(x, (np.floating, float)) for x in y_pred), "Predictions should be floats"
    print("✓ Pipeline training successful")

    # Test saving
    save_pipeline(pipeline, TEST_MODEL_PATH)
    assert TEST_MODEL_PATH.exists(), f"Modelo não encontrado em {TEST_MODEL_PATH.absolute()}"
    print("✓ Pipeline saving successful")

    return X.head()

def test_model_loading_and_prediction(test_data):
    """Test model loading and prediction"""
    print("\n=== Testing model loading and prediction ===")

    # Test loading
    pipeline = load_pipeline(TEST_MODEL_PATH)
    assert hasattr(pipeline, 'predict'), "Loaded pipeline should have predict method"
    print("✓ Model loading successful")

    # Simplified prediction test
    single_prediction = pipeline.predict(test_data.iloc[:1,:])

    # Verificações básicas
    assert isinstance(single_prediction, np.ndarray), "Prediction should be a numpy array"
    assert len(single_prediction) == 1, "Should return single prediction for single input"
    assert isinstance(single_prediction[0], (np.floating, float)), "Prediction should be float"

    print(f"✓ Prediction successful. Lap time: {single_prediction[0]:.2f} seconds")

    # Clean up
    os.remove(TEST_MODEL_PATH)

def run_full_test():
    """Run all tests sequentially"""
    try:
        print("Starting comprehensive system test...\n")

        # Test data components
        cleaned_data = test_data_loading_and_cleaning()

        # Test preprocessing and training
        test_data = test_preprocessing_and_training(cleaned_data)

        # Test model serving
        test_model_loading_and_prediction(test_data)

        print("\n✅ All tests passed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_full_test()

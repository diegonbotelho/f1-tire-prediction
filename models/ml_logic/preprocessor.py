"""
Module - data preprocessing and model training
"""
import pandas as pd

import joblib
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

def create_and_fit_pipeline(X, y):
    """
    Creates, fits and returns a complete pipeline with:
    - Preprocessing (scaling + encoding)
    - Model training
    """
    # 1. Preprocessing
    num_transf_minmax = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('minmax_scaler', MinMaxScaler())
    ])

    num_transf_robust = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('robust_scaler', RobustScaler())
    ])

    num_transf_combined = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('robust_scaler', RobustScaler()),
        ('minmax_scaler', MinMaxScaler())
    ])

    cat_transformer = Pipeline([
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num_minmax', num_transf_minmax, ['Position', 'Stint', 'TyreLife']),
            ('num_robust', num_transf_robust, ['AirTemp', 'TrackTemp', 'Humidity']),
            ('num_combined', num_transf_combined, ['Pressure']),
            ('cat', cat_transformer, ['Driver', 'GrandPrix', 'Compound'])
        ],
        remainder='drop'
    )

    # 2. Model
    model = RandomForestRegressor()

    # 3. Pipeline
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # 4. Train and return the pipeline
    full_pipeline.fit(X, y)
    return full_pipeline

DEFAULT_MODEL_FILENAME = 'model_pipeline.pkl'

def get_raw_data_path():
    """Returns the absolute path to the raw_data folder"""
    return Path(__file__).parent.parent.parent / "raw_data"

def save_pipeline(pipeline, filename=DEFAULT_MODEL_FILENAME):
    """Saves the pipeline in the raw_data folder"""
    save_dir = Path(__file__).parent.parent.parent / "raw_data"
    save_path = save_dir / filename

    # Save the model
    joblib.dump(pipeline, save_path)
    print(f"✅ Model save in: {save_path}")
    return save_path

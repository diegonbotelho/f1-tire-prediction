"""
Replicates the preprocessing pipeline from the notebook
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer

class F1Preprocessor:
    def __init__(self):
        self.numeric_features = [
            'LapNumber', 'Stint', 'TyreLife', 'Position',
            'AirTemp', 'TrackTemp', 'Humidity', 'Pressure',
            'Sector1Time', 'Sector2Time', 'Sector3Time'
        ]
        self.categorical_features = ['Driver', 'GrandPrix', 'Compound']

    def build_preprocessor(self):
        # Numerical transformers
        num_minmax = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ])

        num_robust = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])

        # Categorical transformer
        cat_transformer = Pipeline([
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num_minmax', num_minmax, ['Position', 'Stint', 'TyreLife']),
                ('num_robust', num_robust, ['AirTemp', 'TrackTemp', 'Humidity']),
                ('cat', cat_transformer, self.categorical_features)
            ],
            remainder='passthrough'
        )

        return Pipeline([
            ('preprocessor', preprocessor),
            ('final_scaler', MinMaxScaler())
        ])

    def prepare_data(self, df):
        """Replicates all preprocessing steps from the notebook"""
        # Convert lap times to seconds
        time_cols = ['Sector1Time', 'Sector2Time', 'Sector3Time', 'LapTime']
        for col in time_cols:
            if col in df.columns:
                df[col] = pd.to_timedelta(df[col]).dt.total_seconds()

        # Calculate lap percentage
        df['LapPct'] = df['LapNumber'] / df.groupby(['Event_Year', 'GrandPrix'])['LapNumber'].transform('max')

        # Remove outliers (same as notebook)
        q1 = df['LapTime'].quantile(0.25)
        q3 = df['LapTime'].quantile(0.75)
        iqr = q3 - q1
        df = df[(df['LapTime'] >= q1 - 1.5*iqr) & (df['LapTime'] <= q3 + 1.5*iqr)]

        return df

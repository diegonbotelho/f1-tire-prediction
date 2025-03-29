"""
Data Loading Utilities
"""

from pathlib import Path
import pandas as pd
from typing import Tuple

def load_data() -> pd.DataFrame:
    """Load race data from CSV"""
    return pd.read_csv(Path('../raw_data/df_all_races.csv'))

def load_processed_data() -> pd.DataFrame:
    """Load race data from CSV"""
    df = pd.read_csv(Path('../raw_data/df_all_races.csv'))

    # Ensure proper types
    time_cols = ['Sector1Time', 'Sector2Time', 'Sector3Time']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_timedelta(df[col]).dt.total_seconds()

    return df

def get_train_test_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Get processed data split for training"""
    df = load_processed_data()
    X = df.drop(columns=['LapTime'])
    y = df['LapTime']
    return X, y

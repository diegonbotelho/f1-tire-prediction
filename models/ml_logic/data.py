"""
Module for data loading and cleaning operations.
"""
from pathlib import Path
import pandas as pd
import numpy as np

def get_data():
    """
    Loads the raw data from CSV file
    Returns:
        pd.DataFrame: Raw data
    """
    # Get the path of the current script file:
    current_file = Path(__file__).resolve()

    # Get the parent directory of the current file:
    project_root = current_file.parent.parent.parent

    # Now, join paths in a platform-independent way:
    data_path = project_root / "raw_data" / "df_all_races.csv"
    data = pd.read_csv(data_path)

    return data

def exclude_outliers(df, feature):
    """
    Exclude outliers from a feature using IQR method
    Args:
        df (pd.DataFrame): Input dataframe
        feature (str): Column name to process
    Returns:
        pd.DataFrame: Data without outliers
    """
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return df_filtered

def clean_data(data):
    """
    Cleans the raw data by:
    - Removing first/last laps of each stint
    - Dropping unnecessary columns
    - Handling missing values
    - Converting boolean to numeric
    - Removing outliers
    - Calculating lap percentage
    Args:
        data (pd.DataFrame): Raw data
    Returns:
        pd.DataFrame: Cleaned data
    """
    # Remove first/last laps and pit stops
    first_and_last_laps_in_stint_indexes = data[
        (data.PitInTime.notna()) |
        (data.PitOutTime.notna()) |
        (data.LapNumber == 1.0)].index
    data = data.drop(first_and_last_laps_in_stint_indexes)

    # Drop unnecessary columns
    drop_columns = [
        'Time', 'DriverNumber', 'PitOutTime', 'PitInTime',
        'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime',
        'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'IsPersonalBest',
        'FreshTyre', 'Team', 'LapStartTime', 'LapStartDate', 'TrackStatus',
        'Deleted', 'DeletedReason', 'FastF1Generated', 'IsAccurate',
        'WindDirection', 'WindSpeed', 'Delta_Lap'
    ]
    data = data.drop(columns=drop_columns)

    # Drop rows with missing Position
    data = data.dropna(subset=['Position'])

    # Convert Rainfall to numeric
    data['Rainfall'] = data['Rainfall'].replace({False: 0, True: 1}).astype(int)

    # Remove LapTime outliers
    data = exclude_outliers(data, 'LapTime')

    # Drops each line where TrackStatus is not 1:
    # data = data[data['TrackStatus']==1]

    # Calculate lap percentage
    # data['LapPct'] = data['LapNumber'] / data.groupby(
    #     ['Event_Year', 'GrandPrix'])['LapNumber'].transform('max')

    return data

# Single test for data.py
if __name__ == "__main__":
    print("=== ISOLATED TEST data.py ===")

    # 1. Test charging
    print("\n1. Testing get_data()...")
    raw_df = get_data()
    print(f"✅ Uploaded data. Shape: {raw_df.shape}")
    print(f"Columns: {list(raw_df.columns)}")

    # 2. Test cleaning
    print("\n2. Testing clean_data()...")
    cleaned_df = clean_data(raw_df.copy())
    print(f"✅ Clean data. Shape: {cleaned_df.shape}")
    print(f"Columns removed: {set(raw_df.columns) - set(cleaned_df.columns)}")

    # 3. Test outliers
    print("\n3. Testing exclude_outliers()...")
    before = cleaned_df.shape[0]
    no_outliers_df = exclude_outliers(cleaned_df, 'LapTime')
    after = no_outliers_df.shape[0]
    print(f"✅ Outliers removed: {before - after} records")
    print(f"✅ Final Shape: {no_outliers_df.shape}")
    print(no_outliers_df.head(0))

    print("\n=== COMPLETE TESTS ===")

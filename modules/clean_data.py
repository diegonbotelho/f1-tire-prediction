### This module prepares and cleans the data

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
from sklearn.ensemble import RandomForestRegressor
import numpy as np

#remove future downcasting warning
pd.set_option('future.no_silent_downcasting', True)


def get_data():
    # Gets the gata and returns as a Pandas Dataframe.

    # Get the path of the current script file:
    current_file = Path(__file__).resolve()

    # Get the parent directory of the current file:
    current_dir = current_file.parent

    # Now, join paths in a platform-independent way:
    data_path = current_dir.parent / "raw_data" / "df_all_races.csv"

    data = pd.read_csv(data_path)

    return data

def exclude_outliers(df, feature):
    """
    Exclude outliers from a feature in a DataFrame using the IQR method.
    """
    # Calculate Q1, Q3, and IQR
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Filter the DataFrame to exclude outliers
    df_filtered = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return df_filtered

def clean_data(data):
    # Cleans the data, dropping unnecessary rows and columns

    # Cleans first and last laps of each stint:
    first_and_last_laps_in_stint_indexes = data[(data.PitInTime.notna()) |
    (data.PitOutTime.notna()) |
    (data.LapNumber == 1.0)].index
    data = data.drop(first_and_last_laps_in_stint_indexes)

    # Drops each line where TrackStatus is not 1:
    data = data[data['TrackStatus']==1]

    # Exclusion of features that are not necessary
    drop_columns = [
        'Time',
        'DriverNumber',
        'PitOutTime',
        'PitInTime',
        'Sector1SessionTime',
        'Sector2SessionTime',
        'Sector3SessionTime',
        'Sector1Time',
        'Sector2Time',
        'Sector3Time',
        'SpeedI1',
        'SpeedI2',
        'SpeedFL',
        'SpeedST',
        'IsPersonalBest',
        'FreshTyre',
        'Team',
        'LapStartTime',
        'LapStartDate',
        'TrackStatus',
        'Deleted',
        'DeletedReason',
        'FastF1Generated',
        'IsAccurate',
        'WindDirection',
        'WindSpeed',
        'Delta_Lap'
    ]
    data = data.drop(columns=drop_columns)

    # drop nan in position column
    data = data.dropna(subset=['Position'])

    # replacing rainfall with numeric values
    data['Rainfall'] = data['Rainfall'].replace({False: 0, True: 1}).astype(int)

    # remove outliers from the "laptime" column
    data = exclude_outliers(data, 'LapTime')

    # Preprocess LapNumber as Percentage of the race completed
    # First, calculate the max lap number for each unique GrandPrix/Event_Year combination
    data['LapPct'] = data['LapNumber'] / data.groupby(['Event_Year', 'GrandPrix'])['LapNumber'].transform('max')

    return data

def scale_and_encode(data):
    #scales and encodes the data:

    # StandardScaler Pipeline for numerical features
    num_transf_std = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('standard_scaler', StandardScaler())
    ])

    # MinMaxScaler Pipeline for numerical features
    nun_transf_minmax = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('minmax_scaler', MinMaxScaler())
    ])

    # RobustScaler Pipeline for numerical features
    num_transf_robust = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('robust_scaler', RobustScaler())
    ])

    # RobustScaler & MinMaxSacler Pipeline. Combination of these two methods
    num_transf_combined = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('robust_scaler', RobustScaler()),
        ('minmax_scaler', MinMaxScaler())
    ])

    # Categorical features Pipeline that will be encoded using OneHotEncoder
    cat_transformer = Pipeline([
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    numeric_preprocessor = ColumnTransformer(
        transformers=[
        #('num_std', num_transf_std, [numerical_columns]),              # StandardScale numerical features
            ('num_minmax', nun_transf_minmax, ['Position',
                                            'Stint',
                                            'TyreLife']),               # MinMaxScale numerical features
            ('num_robust', num_transf_robust, ['AirTemp',
                                            'TrackTemp',
                                            'Humidity',]),              # RobustScale numerical features
            ('num_combined', num_transf_combined, ['Pressure']),        # Combined two scalers 1° RobustScaler 2° MinMaxScaler
            ('cat', cat_transformer, ['Driver', 'GrandPrix', 'Compound']), # OneHotEncode categorical features
            ('passthrough_cols', 'passthrough', ['LapTime', 'LapPct'])     # passthrough means nothings will be done in these columns, it wil copy and paste to the output
        ],
        remainder='drop')                                                  # Columns in the original dataframe not mentioned in ColumnTransformer will be dropped

    data = numeric_preprocessor.fit_transform(data)

    data = pd.DataFrame(data, columns=numeric_preprocessor.get_feature_names_out())

    return data

def load_model():
    #loads the saved model

    # Get the path of the current script file:
    current_file = Path(__file__).resolve()

    # Get the parent directory of the current file:
    current_dir = current_file.parent

    # Now, join paths in a platform-independent way:
    model_path = current_dir.parent / "models" / "random_forest_modelo.pkl"

    model = joblib.load(model_path)

    return model

def get_mae(y,y_pred):
    #gets mean absolute error

    result = 0
    for i,j in zip(y,y_pred):
        result += np.abs(i-j)

    return result/len(y)

data = scale_and_encode(clean_data(get_data()))

X = data.drop(columns = ['passthrough_cols__LapTime'])
y = data['passthrough_cols__LapTime']

model = load_model()

y_pred = model.predict(X)

print(get_mae(y,y_pred))

import pandas as pd
import fastf1 as ff1 # type: ignore
import os

# Function to load data for a specific year and race
def load_race_data(year, race_name):
    try:
        session = ff1.get_session(year, race_name, 'R')  # Load race session
        session.load()                                   # Load all data
        laps = session.laps                              # Get lap data
        weather = laps.get_weather_data()                # Get weather data
        laps = laps.reset_index(drop=True)
        weather = weather.reset_index(drop=True)
        weather = weather.drop(columns=['Time'])         # Drop redundant 'Time' column
        joined_data = pd.concat([laps, weather], axis=1) # Join laps and weather data
        joined_data['Event_Year'] = year                 # Add year column for reference
        joined_data['GrandPrix'] = race_name             # Add Grand Prix name column
        return joined_data
    except Exception as e:
        print(f"Error loading data for {year} {race_name}: {e}")
        return pd.DataFrame()  # Return empty DataFrame in case of error


def process_data(all_races_data):
    """Process the raw race data by converting times and calculating deltas"""
    # Convert lap times to seconds
    all_races_data['LapTime'] = all_races_data['LapTime'].dt.total_seconds()
    all_races_data['Sector1Time'] = all_races_data['Sector1Time'].dt.total_seconds()
    all_races_data['Sector2Time'] = all_races_data['Sector2Time'].dt.total_seconds()
    all_races_data['Sector3Time'] = all_races_data['Sector3Time'].dt.total_seconds()
    all_races_data['Sector1SessionTime'] = all_races_data['Sector1SessionTime'].dt.total_seconds()
    all_races_data['Sector2SessionTime'] = all_races_data['Sector2SessionTime'].dt.total_seconds()
    all_races_data['Sector3SessionTime'] = all_races_data['Sector3SessionTime'].dt.total_seconds()

    # Calculate delta_lap (time difference between consecutive laps for each driver)
    all_races_data['Delta_Lap'] = 0.0

    for driver in all_races_data['Driver'].unique():
        driver_data = all_races_data[all_races_data['Driver'] == driver]
        delta_lap = driver_data['LapTime'].diff().fillna(0)
        all_races_data.loc[driver_data.index, 'Delta_Lap'] = delta_lap

    return all_races_data


def main():
    # List of races of 2024
    years = [2022, 2023, 2024]
    race_names = [
        'Bahrain', 'Saudi Arabia', 'Australia', 'Azerbaijan', 'Miami', 'Monaco', 'Spain',
        'Canada', 'Austria', 'Great Britain', 'Hungary', 'Belgium', 'Netherlands', 'Italy',
        'Singapore', 'Japan', 'USA', 'Mexico', 'Brazil', 'Abu Dhabi'
    ]

    # Load data for all races
    all_races_data = pd.DataFrame()
    for year in years:
        for race in race_names:
            print(f"Loading data for {year} {race}...")
            race_data = load_race_data(year, race)
            if not race_data.empty:
                all_races_data = pd.concat([all_races_data, race_data], ignore_index=True)

    # Check the loaded data
    print("\nTotal rows loaded:", len(all_races_data))
    print("\nColumns in the dataset:", all_races_data.columns)

    # Process the data
    all_races_df = process_data(all_races_data.copy())

    # Save the data
    raw_data_path = '/home/diego_nbotelho/code/diegonbotelho/f1-tire-prediction/raw_data'
    filename = 'df_all_races.csv'
    full_path = os.path.join(raw_data_path, filename)

    # Create directory if it doesn't exist
    os.makedirs(raw_data_path, exist_ok=True)

    # Save the dataframe to CSV
    all_races_df.to_csv(full_path, index=False)
    print(f"Data saved to {full_path}")


if __name__ == "__main__":
    main()

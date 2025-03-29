"""
Single Race Analysis - Demonstrative Example

Note: The full dataset is already available as all_races_df.csv in data/raw/
This script serves only for educational/demonstration purposes.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ml_logic.data import load_raw_data

def analyze_bahrain_2024():
    """Analyze Bahrain 2024 as example race"""
    # Load and filter data
    df = load_raw_data()
    bahrain = df[(df['GrandPrix'] == 'Bahrain') & (df['Event_Year'] == 2024)].copy()

    # Convert lap times to seconds
    time_cols = ['Sector1Time', 'Sector2Time', 'Sector3Time', 'LapTime']
    for col in time_cols:
        bahrain[col] = pd.to_timedelta(bahrain[col]).dt.total_seconds()

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Lap time distribution
    sns.histplot(bahrain['LapTime'], bins=30, ax=axes[0, 0], kde=True)
    axes[0, 0].set_title('Lap Time Distribution')

    # Tire performance
    sns.boxplot(data=bahrain, x='Compound', y='LapTime', ax=axes[0, 1])
    axes[0, 1].set_title('Tire Compound Performance')

    # Lap time evolution
    for driver in bahrain['Driver'].unique()[:5]:  # Top 5 drivers
        driver_data = bahrain[bahrain['Driver'] == driver]
        axes[1, 0].plot(driver_data['LapNumber'], driver_data['LapTime'], label=driver)
    axes[1, 0].set_title('Lap Time Evolution')
    axes[1, 0].legend()

    # Sector correlations
    sns.heatmap(bahrain[['Sector1Time', 'Sector2Time', 'Sector3Time']].corr(),
                annot=True, ax=axes[1, 1])
    axes[1, 1].set_title('Sector Time Correlations')

    plt.tight_layout()
    plt.savefig('race_analysis.png')
    plt.close()

    return bahrain.describe()

if __name__ == "__main__":
    print("Running single race analysis...")
    stats = analyze_bahrain_2024()
    print(stats)

"""
train_model.py - Generate and save model_pipeline.pkl
"""
from pathlib import Path
import pandas as pd
from data import get_data, clean_data
from preprocessor import create_and_fit_pipeline, save_pipeline

def train_and_save_model():
    # Get and clear data
    print("🔄 Retrieving and clearing data...")
    df = clean_data(get_data())

    #2 Prepare features and target
    print("⚙️ Preparing data for training...")
    X = df.drop(columns=['Sector1Time', 'Sector2Time', 'Sector3Time', 'LapTime'])
    y = df['LapTime']

    #3 Train and save the complete pipeline
    print("🔧 Training pipeline...")
    pipeline = create_and_fit_pipeline(X, y)

    #4 Save for future use
    print("💾 Saving a template...")
    save_path = Path(__file__).parent.parent.parent / "raw_data" / "model_pipeline.pkl"
    save_pipeline(pipeline, save_path)

    print(f"✅ Model trained and saved in: {save_path}")

if __name__ == "__main__":
    train_and_save_model()

"""
Download and prepare the Heart Disease UCI dataset.

Reads the processed Cleveland data, adds column headers,
and saves as a clean CSV for downstream processing.

Usage:
    python data/download_data.py
"""
import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_SOURCE = os.path.join(PROJECT_ROOT, "..", "heart+disease", "processed.cleveland.data")
RAW_OUTPUT = os.path.join(SCRIPT_DIR, "raw", "heart.csv")

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]


def download():
    """Copy raw UCI data and save with headers as CSV."""
    os.makedirs(os.path.dirname(RAW_OUTPUT), exist_ok=True)

    df = pd.read_csv(RAW_SOURCE, header=None, names=COLUMN_NAMES, na_values="?")

    # Convert target: 0 stays 0, values 1-4 become 1 (disease present)
    df["target"] = (df["target"] > 0).astype(int)

    df.to_csv(RAW_OUTPUT, index=False)
    print(f"Dataset saved to {RAW_OUTPUT}")
    print(f"  Shape: {df.shape}")
    print(f"  Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return RAW_OUTPUT


if __name__ == "__main__":
    download()

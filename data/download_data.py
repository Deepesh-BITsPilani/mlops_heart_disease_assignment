"""
Download and prepare the Heart Disease UCI dataset.

Reads the processed Cleveland data from a local copy if available,
otherwise downloads it from the UCI Machine Learning Repository.

Usage:
    python data/download_data.py
"""
import os
import pandas as pd
from urllib.request import urlretrieve

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
LOCAL_SOURCE = os.path.join(PROJECT_ROOT, "..", "heart+disease", "processed.cleveland.data")
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
RAW_OUTPUT = os.path.join(SCRIPT_DIR, "raw", "heart.csv")

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]


def _resolve_source():
    """Return path to the raw Cleveland data, downloading if needed."""
    if os.path.exists(LOCAL_SOURCE):
        print(f"  Using local dataset: {LOCAL_SOURCE}")
        return LOCAL_SOURCE

    downloaded = os.path.join(SCRIPT_DIR, "raw", "processed.cleveland.data")
    os.makedirs(os.path.dirname(downloaded), exist_ok=True)
    print(f"  Local dataset not found. Downloading from UCI repository...")
    urlretrieve(UCI_URL, downloaded)
    print(f"  Downloaded to {downloaded}")
    return downloaded


def download():
    """Load raw UCI data (local or remote) and save with headers as CSV."""
    os.makedirs(os.path.dirname(RAW_OUTPUT), exist_ok=True)

    source = _resolve_source()
    df = pd.read_csv(source, header=None, names=COLUMN_NAMES, na_values="?")

    df["target"] = (df["target"] > 0).astype(int)

    df.to_csv(RAW_OUTPUT, index=False)
    print(f"Dataset saved to {RAW_OUTPUT}")
    print(f"  Shape: {df.shape}")
    print(f"  Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return RAW_OUTPUT


if __name__ == "__main__":
    download()

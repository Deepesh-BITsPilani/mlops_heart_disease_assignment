"""
Data preprocessing pipeline for the Heart Disease dataset.

Handles missing value imputation, train/test splitting,
and builds a reusable sklearn preprocessing pipeline.
"""
import os
import sys
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import FEATURE_NAMES, TARGET_NAME, TEST_SIZE, RANDOM_STATE  # noqa: E402


def load_raw_data(path=None):
    """Load the raw heart disease CSV."""
    if path is None:
        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "raw", "heart.csv"
        )
    df = pd.read_csv(path)
    return df


def clean_data(df):
    """Handle missing values and ensure correct dtypes."""
    df = df.copy()
    for col in ["ca", "thal"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Impute missing values with median
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def build_preprocessor():
    """Build a reusable sklearn preprocessing pipeline."""
    numeric_features = FEATURE_NAMES
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
    ])
    return preprocessor


def load_and_split(raw_path=None):
    """Load data, clean it, and split into train/test DataFrames."""
    df = load_raw_data(raw_path)
    df = clean_data(df)

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=df[TARGET_NAME],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_splits(train_df, test_df, output_dir=None):
    """Save train/test CSVs to disk."""
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "processed"
        )
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"  Train samples: {len(train_df)} -> {train_path}")
    print(f"  Test samples:  {len(test_df)} -> {test_path}")
    return train_path, test_path


def run():
    """Execute the full data preparation step."""
    print("=" * 60)
    print("STEP 1: DATA PREPARATION")
    print("=" * 60)

    train_df, test_df = load_and_split()
    train_path, test_path = save_splits(train_df, test_df)

    stats = {
        "total_samples": len(train_df) + len(test_df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "features": FEATURE_NAMES,
        "target_distribution_train": train_df[TARGET_NAME].value_counts().to_dict(),
        "target_distribution_test": test_df[TARGET_NAME].value_counts().to_dict(),
    }
    stats_path = os.path.join(os.path.dirname(train_path), "data_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved -> {stats_path}")
    print("  Data preparation complete.")
    return train_path, test_path


if __name__ == "__main__":
    run()

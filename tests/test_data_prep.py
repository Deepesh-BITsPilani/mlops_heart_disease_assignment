"""Unit tests for data preparation module."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_prep.prepare import load_raw_data, clean_data, load_and_split, build_preprocessor  # noqa: E402
from configs.config import FEATURE_NAMES, TARGET_NAME, COLUMN_NAMES  # noqa: E402


@pytest.fixture
def raw_data():
    return load_raw_data()


@pytest.fixture
def clean_df(raw_data):
    return clean_data(raw_data)


def test_raw_data_loads(raw_data):
    """Raw CSV loads with correct shape and columns."""
    assert raw_data.shape[1] == 14
    assert list(raw_data.columns) == COLUMN_NAMES
    assert raw_data.shape[0] > 0


def test_raw_data_has_expected_rows(raw_data):
    """Cleveland dataset should have 303 rows."""
    assert raw_data.shape[0] == 303


def test_target_is_binary(raw_data):
    """Target should be binary (0 or 1) after download script."""
    assert set(raw_data[TARGET_NAME].dropna().unique()).issubset({0, 1})


def test_clean_data_no_missing(clean_df):
    """After cleaning, no missing values remain."""
    assert clean_df.isnull().sum().sum() == 0


def test_clean_data_preserves_shape(raw_data, clean_df):
    """Cleaning should not drop rows (we impute, not drop)."""
    assert clean_df.shape[0] == raw_data.shape[0]


def test_load_and_split_sizes():
    """Train/test split produces correct proportions (~80/20)."""
    train_df, test_df = load_and_split()
    total = len(train_df) + len(test_df)
    assert total == 303
    assert len(test_df) / total == pytest.approx(0.2, abs=0.03)


def test_load_and_split_stratified():
    """Split should maintain target distribution."""
    train_df, test_df = load_and_split()
    train_ratio = train_df[TARGET_NAME].mean()
    test_ratio = test_df[TARGET_NAME].mean()
    assert abs(train_ratio - test_ratio) < 0.1


def test_load_and_split_columns():
    """Split DataFrames should have all expected columns."""
    train_df, test_df = load_and_split()
    assert list(train_df.columns) == COLUMN_NAMES
    assert list(test_df.columns) == COLUMN_NAMES


def test_preprocessor_transforms():
    """Preprocessor should transform features to correct shape."""
    preprocessor = build_preprocessor()
    train_df, _ = load_and_split()
    X = train_df[FEATURE_NAMES]
    preprocessor.fit(X)
    X_transformed = preprocessor.transform(X)
    assert X_transformed.shape == X.shape

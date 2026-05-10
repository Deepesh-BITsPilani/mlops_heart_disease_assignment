"""Unit tests for model training and prediction."""
import os
import sys
import pytest
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_prep.prepare import load_and_split  # noqa: E402
from src.training.train import build_pipeline  # noqa: E402
from configs.config import FEATURE_NAMES, TARGET_NAME, RANDOM_STATE  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402


@pytest.fixture(scope="module")
def data():
    train_df, test_df = load_and_split()
    X_train = train_df[FEATURE_NAMES].values
    y_train = train_df[TARGET_NAME].values
    X_test = test_df[FEATURE_NAMES].values
    y_test = test_df[TARGET_NAME].values
    return X_train, y_train, X_test, y_test


@pytest.fixture(scope="module")
def trained_lr(data):
    X_train, y_train, _, _ = data
    pipeline = build_pipeline(LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))
    pipeline.fit(X_train, y_train)
    return pipeline


@pytest.fixture(scope="module")
def trained_rf(data):
    X_train, y_train, _, _ = data
    pipeline = build_pipeline(RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100))
    pipeline.fit(X_train, y_train)
    return pipeline


def test_lr_trains(trained_lr, data):
    """Logistic Regression pipeline should train and score >70%."""
    _, _, X_test, y_test = data
    acc = trained_lr.score(X_test, y_test)
    assert acc > 0.70, f"LR accuracy too low: {acc}"


def test_rf_trains(trained_rf, data):
    """Random Forest pipeline should train and score >70%."""
    _, _, X_test, y_test = data
    acc = trained_rf.score(X_test, y_test)
    assert acc > 0.70, f"RF accuracy too low: {acc}"


def test_prediction_is_binary(trained_lr, data):
    """Predictions should be 0 or 1."""
    _, _, X_test, _ = data
    preds = trained_lr.predict(X_test)
    assert set(preds).issubset({0, 1})


def test_probabilities_sum_to_one(trained_lr, data):
    """Prediction probabilities should sum to 1."""
    _, _, X_test, _ = data
    probas = trained_lr.predict_proba(X_test)
    sums = probas.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-6)


def test_probabilities_shape(trained_rf, data):
    """Probability output should be (n_samples, 2)."""
    _, _, X_test, _ = data
    probas = trained_rf.predict_proba(X_test)
    assert probas.shape == (X_test.shape[0], 2)


def test_single_prediction(trained_lr):
    """Model should handle a single sample."""
    sample = np.array([[63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6]])
    pred = trained_lr.predict(sample)
    proba = trained_lr.predict_proba(sample)
    assert pred[0] in [0, 1]
    assert proba.shape == (1, 2)


def test_saved_model_exists():
    """Check that training produces a saved model file."""
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "model", "artifacts", "heart_disease_model.pkl"
    )
    assert os.path.exists(model_path), f"Model file not found: {model_path}"


def test_saved_model_loads_and_predicts():
    """Saved model should load and produce valid predictions."""
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "model", "artifacts", "heart_disease_model.pkl"
    )
    if not os.path.exists(model_path):
        pytest.skip("Model not yet trained")

    model = joblib.load(model_path)
    sample = np.array([[63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6]])
    pred = model.predict(sample)
    assert pred[0] in [0, 1]

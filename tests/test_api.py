"""Unit tests for the FastAPI serving endpoint."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient  # noqa: E402
from src.serving.app import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


SAMPLE_INPUT = {
    "age": 63.0,
    "sex": 1.0,
    "cp": 1.0,
    "trestbps": 145.0,
    "chol": 233.0,
    "fbs": 1.0,
    "restecg": 2.0,
    "thalach": 150.0,
    "exang": 0.0,
    "oldpeak": 2.3,
    "slope": 3.0,
    "ca": 0.0,
    "thal": 6.0,
}


def test_root(client):
    """GET / should return API info."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "service" in data
    assert data["service"] == "Heart Disease Prediction API"


def test_health(client):
    """GET /health should return healthy status."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_valid(client):
    """POST /predict with valid input returns prediction."""
    resp = client.post("/predict", json=SAMPLE_INPUT)
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1]
    assert "label" in data
    assert data["label"] in ["Disease", "No Disease"]
    assert "confidence" in data
    assert 0 <= data["confidence"] <= 1
    assert "probabilities" in data
    assert "no_disease" in data["probabilities"]
    assert "disease" in data["probabilities"]


def test_predict_probabilities_sum(client):
    """Probabilities should sum to ~1."""
    resp = client.post("/predict", json=SAMPLE_INPUT)
    data = resp.json()
    total = data["probabilities"]["no_disease"] + data["probabilities"]["disease"]
    assert abs(total - 1.0) < 0.01


def test_predict_missing_field(client):
    """POST /predict with missing field should return 422."""
    incomplete = {k: v for k, v in SAMPLE_INPUT.items() if k != "age"}
    resp = client.post("/predict", json=incomplete)
    assert resp.status_code == 422


def test_predict_empty_body(client):
    """POST /predict with empty body should return 422."""
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


def test_docs_endpoint(client):
    """Swagger docs should be accessible."""
    resp = client.get("/docs")
    assert resp.status_code == 200

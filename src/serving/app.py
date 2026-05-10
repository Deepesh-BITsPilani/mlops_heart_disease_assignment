"""
Heart Disease Prediction API built with FastAPI.

Endpoints:
  GET  /         - API info
  GET  /health   - Health check
  POST /predict  - Single prediction (13 features -> disease risk)
"""
import os
import time
import json
import logging
import joblib
import numpy as np
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}',
)
logger = logging.getLogger("heart-disease-api")

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "model", "artifacts", "heart_disease_model.pkl"),
)

model = None


@asynccontextmanager
async def lifespan(app):
    load_model()
    yield


app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts heart disease risk from patient health data",
    version="1.0.0",
    lifespan=lifespan,
)

from src.monitoring.middleware import setup_monitoring, record_prediction  # noqa: E402
setup_monitoring(app)


class PredictionRequest(BaseModel):
    age: float = Field(..., description="Age in years")
    sex: float = Field(..., description="Sex (1=male, 0=female)")
    cp: float = Field(..., description="Chest pain type (1-4)")
    trestbps: float = Field(..., description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., description="Serum cholesterol (mg/dl)")
    fbs: float = Field(..., description="Fasting blood sugar > 120 mg/dl (1=true, 0=false)")
    restecg: float = Field(..., description="Resting ECG results (0-2)")
    thalach: float = Field(..., description="Max heart rate achieved")
    exang: float = Field(..., description="Exercise induced angina (1=yes, 0=no)")
    oldpeak: float = Field(..., description="ST depression induced by exercise")
    slope: float = Field(..., description="Slope of peak exercise ST segment (1-3)")
    ca: float = Field(..., description="Number of major vessels colored by fluoroscopy (0-3)")
    thal: float = Field(..., description="Thal (3=normal, 6=fixed defect, 7=reversible defect)")


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    probabilities: dict
    input_features: dict


def load_model():
    """Load the trained model from disk."""
    global model
    if model is not None:
        return model

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)
    return model


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    log_data = {
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "duration_ms": round(duration * 1000, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    logger.info(json.dumps(log_data))
    return response


@app.get("/")
def root():
    return {
        "service": "Heart Disease Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "API info",
            "GET /health": "Health check",
            "GET /docs": "Swagger UI",
            "POST /predict": "Single prediction",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = np.array([[
        request.age, request.sex, request.cp, request.trestbps,
        request.chol, request.fbs, request.restecg, request.thalach,
        request.exang, request.oldpeak, request.slope, request.ca,
        request.thal,
    ]])

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    label = "Disease" if prediction == 1 else "No Disease"
    confidence = float(probabilities[prediction])

    record_prediction(label)

    return PredictionResponse(
        prediction=prediction,
        label=label,
        confidence=round(confidence, 4),
        probabilities={
            "no_disease": round(float(probabilities[0]), 4),
            "disease": round(float(probabilities[1]), 4),
        },
        input_features=request.model_dump(),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

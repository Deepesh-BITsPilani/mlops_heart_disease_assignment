# Heart Disease MLOps Pipeline

End-to-end MLOps pipeline for predicting heart disease risk using the UCI Heart Disease dataset, featuring model training with experiment tracking, CI/CD automation, Docker containerization, Kubernetes deployment, and production monitoring.

## Architecture

```
UCI Dataset -> download_data.py -> prepare.py -> train/test CSVs
                                                      |
                                              train.py + MLflow
                                                      |
                                              Model Artifact (.pkl)
                                                      |
                                          FastAPI Serving (app.py)
                                                      |
                                    Docker Container -> Kubernetes
                                                      |
                                        Prometheus -> Grafana Monitoring
```

## Quick Start

### 1. Setup Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
python data/download_data.py
python -c "from src.data_prep.prepare import run; run()"
```

### 3. Train Models

```bash
python -c "from src.training.train import run; run()"
```

This trains Logistic Regression and Random Forest with GridSearchCV, logs experiments to MLflow, and saves the best model.

### 4. View MLflow Experiments

```bash
mlflow ui --backend-store-uri ./mlruns --port 5001
```

Open http://localhost:5001 to compare experiment runs.

### 5. Run Tests

```bash
python -m pytest tests/ -v
```

### 6. Run API Locally

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8080
```

### 7. Docker

```bash
docker build -t heart-disease-api .
docker run -p 8080:8080 heart-disease-api
```

### 8. Docker Compose (with Monitoring)

```bash
docker compose up --build
```

Services:
- **API**: http://localhost:8080
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### 9. Kubernetes (Minikube)

```bash
minikube start
eval $(minikube docker-env)
docker build -t heart-disease-api:latest .
kubectl apply -f k8s/
minikube service heart-disease-api --url
```

## API Usage

### Health Check

```bash
curl http://localhost:8080/health
```

### Predict

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 2, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6
  }'
```

### Response Format

```json
{
  "prediction": 0,
  "label": "No Disease",
  "confidence": 0.8234,
  "probabilities": {
    "no_disease": 0.8234,
    "disease": 0.1766
  }
}
```

### Swagger Docs

Open http://localhost:8080/docs for interactive API documentation.

## Project Structure

```
heart-disease-mlops/
├── data/
│   ├── download_data.py          # Dataset acquisition script
│   ├── raw/                      # Raw dataset
│   └── processed/                # Cleaned train/test CSVs
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   └── 02_training.ipynb         # Model development notebook
├── src/
│   ├── data_prep/prepare.py      # Preprocessing pipeline
│   ├── training/train.py         # Training + MLflow tracking
│   ├── evaluation/evaluate.py    # Model evaluation
│   ├── serving/app.py            # FastAPI prediction API
│   └── monitoring/middleware.py  # Prometheus metrics
├── tests/                        # Unit tests (pytest)
├── model/artifacts/              # Saved model + metrics
├── configs/config.py             # Central configuration
├── k8s/                          # Kubernetes manifests
├── monitoring/                   # Prometheus & Grafana configs
├── Dockerfile
├── docker-compose.yml
├── Jenkinsfile                   # Jenkins CI/CD pipeline
├── .github/workflows/ci.yml     # GitHub Actions pipeline
├── requirements.txt
└── README.md
```

## Dataset

**Heart Disease UCI** (Cleveland subset) from the UCI Machine Learning Repository.

- 303 patients, 13 features, binary target
- Features: age, sex, chest pain type, blood pressure, cholesterol, etc.
- Target: 0 = No disease, 1 = Disease present

## Models

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | 0.85 | 0.81 | 0.89 | 0.85 | 0.96 |
| Random Forest | 0.90 | 0.87 | 0.93 | 0.90 | 0.96 |

Both models use an sklearn Pipeline with median imputation and standard scaling.

## CI/CD

Two pipeline options provided:

1. **Jenkins** (`Jenkinsfile`): Checkout -> Lint -> Data Prep -> Test -> Train -> Docker Build -> Push
2. **GitHub Actions** (`.github/workflows/ci.yml`): Same stages, triggered on push/PR to main

## Monitoring

- **Structured JSON logging** for all API requests
- **Prometheus** metrics: request count, latency, error rate, prediction distribution
- **Grafana** dashboard: pre-configured with API performance panels




# p.s.  "dekausha" is Company specific userId of mine. 

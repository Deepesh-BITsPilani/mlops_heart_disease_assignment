# Heart Disease MLOps Pipeline - Assignment Report

**Course**: MLOps (S2-25_AMLCSZG523)  
**Assignment**: End-to-End ML Model Development, CI/CD, and Production Deployment

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Data Acquisition & EDA](#2-data-acquisition--eda)
3. [Feature Engineering & Model Development](#3-feature-engineering--model-development)
4. [Experiment Tracking](#4-experiment-tracking)
5. [Model Packaging & Reproducibility](#5-model-packaging--reproducibility)
6. [CI/CD Pipeline & Automated Testing](#6-cicd-pipeline--automated-testing)
7. [Model Containerization](#7-model-containerization)
8. [Production Deployment](#8-production-deployment)
9. [Monitoring & Logging](#9-monitoring--logging)
10. [Architecture Overview](#10-architecture-overview)

---

## 1. Introduction

This report documents the end-to-end MLOps pipeline built for the **Heart Disease UCI Dataset**. The goal is to predict heart disease risk from patient health data and deploy the solution as a monitored, cloud-ready API following modern MLOps best practices.

**Dataset**: Heart Disease UCI (Cleveland subset) - 303 patients, 13 clinical features, binary classification target.

**Tech Stack**: Python, scikit-learn, MLflow, FastAPI, Docker, Kubernetes (Minikube), Jenkins/GitHub Actions, Prometheus, Grafana.

---

## 2. Data Acquisition & EDA

### 2.1 Data Acquisition

The dataset was obtained from the UCI Machine Learning Repository. A download script (`data/download_data.py`) reads the raw processed Cleveland data, adds proper column headers, converts the multi-class target (0-4) to binary (0/1), and saves it as a clean CSV.

**Features used (14 attributes)**:
- `age`: Age in years
- `sex`: Sex (1=male, 0=female)
- `cp`: Chest pain type (1-4)
- `trestbps`: Resting blood pressure (mm Hg)
- `chol`: Serum cholesterol (mg/dl)
- `fbs`: Fasting blood sugar > 120 mg/dl
- `restecg`: Resting ECG results (0-2)
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise induced angina
- `oldpeak`: ST depression induced by exercise
- `slope`: Slope of peak exercise ST segment
- `ca`: Number of major vessels colored by fluoroscopy (0-3)
- `thal`: Thalassemia (3=normal, 6=fixed defect, 7=reversible defect)

### 2.2 Data Cleaning

- **Missing values**: `ca` (4 missing) and `thal` (2 missing) contained `?` markers, handled via median imputation.
- **Target encoding**: Values 1-4 mapped to 1 (disease present), value 0 kept as 0 (no disease).
- **Final dataset**: 303 rows, 14 columns (13 features + 1 target), no missing values after cleaning.

### 2.3 Exploratory Data Analysis

The EDA notebook (`notebooks/01_eda.ipynb`) includes:

- **Class distribution**: ~54% no disease (164), ~46% disease (139) - reasonably balanced.
- **Feature histograms**: Continuous features (age, trestbps, chol, thalach, oldpeak) show approximately normal distributions. Categorical features (sex, cp, fbs, restecg, exang, slope, thal) show expected discrete distributions.
- **Correlation heatmap**: Strong correlations with target: `cp` (0.43), `thalach` (-0.42), `exang` (0.44), `oldpeak` (0.43), `slope` (0.35), `ca` (0.47), `thal` (0.52).
- **Box plots**: Disease patients show lower max heart rate (`thalach`), higher ST depression (`oldpeak`), and tend to be slightly older.
- **Categorical analysis**: Chest pain type 4 (asymptomatic) is highly associated with disease. Exercise-induced angina strongly indicates disease.

*See screenshots folder for all EDA visualizations.*

---

## 3. Feature Engineering & Model Development

### 3.1 Feature Pipeline

All 13 features are treated as numeric. The preprocessing pipeline consists of:
1. **SimpleImputer** (median strategy) - handles any remaining missing values
2. **StandardScaler** - normalizes features to zero mean and unit variance

This is implemented as a scikit-learn `Pipeline` that ensures full reproducibility.

### 3.2 Models Trained

Two classification models were trained and compared:

1. **Logistic Regression**: Linear model with L2 regularization. Hyperparameters tuned via GridSearchCV:
   - `C`: [0.01, 0.1, 1.0, 10.0]
   - Best: C=0.1

2. **Random Forest**: Ensemble of decision trees. Hyperparameters tuned via GridSearchCV:
   - `n_estimators`: [100, 200]
   - `max_depth`: [5, 10, None]
   - `min_samples_split`: [2, 5]
   - `min_samples_leaf`: [1, 2]
   - Best: n_estimators=200, max_depth=5, min_samples_split=2, min_samples_leaf=1

### 3.3 Evaluation Results

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| Accuracy | 0.8525 | 0.9016 |
| Precision | 0.8065 | 0.8667 |
| Recall | 0.8929 | 0.9286 |
| F1 Score | 0.8475 | 0.8966 |
| ROC-AUC | 0.9578 | 0.9567 |

### 3.4 Cross-Validation Results (5-fold)

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| CV Accuracy | 0.8263 | 0.8221 |
| CV Precision | 0.8392 | 0.8408 |
| CV Recall | 0.7743 | 0.7557 |
| CV F1 | 0.8034 | 0.7948 |
| CV ROC-AUC | 0.8914 | 0.8940 |

### 3.5 Model Selection

The best model was selected based on **test ROC-AUC score**. Both models achieved nearly identical ROC-AUC (~0.96), with Logistic Regression having a marginal edge. The selected model is saved as the production artifact.

*ROC curve comparison and confusion matrices are saved in the screenshots and model artifacts folders.*

---

## 4. Experiment Tracking

### 4.1 MLflow Integration

MLflow is integrated into the training pipeline (`src/training/train.py`) with file-based tracking (`./mlruns`). For each training run, the following are logged:

- **Parameters**: Model type, all hyperparameters (from GridSearchCV best)
- **Metrics**: accuracy, precision, recall, F1, ROC-AUC (both test and 5-fold CV)
- **Artifacts**: Confusion matrix plot, ROC curve plot, trained model (MLflow sklearn format)
- **Tags**: dataset name, dataset version, run description

### 4.2 Experiment Comparison

The MLflow UI (`mlflow ui --backend-store-uri ./mlruns`) provides:
- Side-by-side comparison of Logistic Regression vs Random Forest
- Metric charts showing performance across runs
- Artifact browser for downloading models and plots

*To reproduce: Run `mlflow ui --port 5001` from the project root and navigate to http://localhost:5001.*

---

## 5. Model Packaging & Reproducibility

### 5.1 Model Format

The final model is saved in two formats:
1. **Joblib pickle** (`model/artifacts/heart_disease_model.pkl`) - for direct loading in the serving API
2. **MLflow model format** (in `mlruns/`) - for MLflow model registry and serving

### 5.2 Sklearn Pipeline

The model is a complete scikit-learn `Pipeline` containing:
```
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(...) or RandomForestClassifier(...)),
])
```

This ensures that **all preprocessing transformations** are bundled with the model, guaranteeing reproducibility regardless of deployment environment.

### 5.3 Dependency Management

All dependencies are pinned in `requirements.txt` with exact versions. To reproduce the environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python data/download_data.py
python -c "from src.data_prep.prepare import run; run()"
python -c "from src.training.train import run; run()"
```

---

## 6. CI/CD Pipeline & Automated Testing

### 6.1 Unit Tests

24 unit tests organized in three test files:

- **test_data_prep.py** (9 tests): Data loading, missing value handling, shape validation, target encoding, train/test split proportions, stratification, preprocessor transforms.
- **test_model.py** (8 tests): Model training accuracy thresholds, prediction format (binary), probability sums, single-sample prediction, saved model loading.
- **test_api.py** (7 tests): Root endpoint, health check, valid prediction, probability sums, missing field validation, empty body validation, Swagger docs.

All tests pass: `python -m pytest tests/ -v` -> **24 passed**.

### 6.2 Jenkins Pipeline

The `Jenkinsfile` defines a multi-stage pipeline:

1. **Checkout**: Pull source code
2. **Setup Python Environment**: Create venv and install dependencies
3. **Lint**: Run flake8 for code quality
4. **Prepare Data**: Download and preprocess dataset
5. **Unit Tests**: Run pytest with JUnit XML reporting
6. **Train Model**: Execute the training pipeline
7. **Build Docker Image**: Build container for serving
8. **Push Docker Image**: Push to registry (on main branch only)

Post-build: Archives model artifacts and test results.

### 6.3 GitHub Actions

The `.github/workflows/ci.yml` workflow provides equivalent functionality:
- Triggered on push/PR to main branch
- Python 3.12 with pip caching
- Same stages as Jenkins
- Additionally tests the Docker container with health check and prediction verification
- Uploads model artifacts and test results

---

## 7. Model Containerization

### 7.1 FastAPI Application

The serving API (`src/serving/app.py`) is built with FastAPI and provides:

- **GET /**: API information and available endpoints
- **GET /health**: Health check (model loaded status)
- **GET /docs**: Auto-generated Swagger UI documentation
- **POST /predict**: Accepts JSON with 13 features, returns prediction, confidence, and probabilities

Request validation is handled by Pydantic models with field descriptions for each clinical feature.

### 7.2 Dockerfile

```dockerfile
FROM python:3.12-slim
# Non-root user for security
# Requirements-first for Docker layer caching
# Health check built into the image
# Runs uvicorn on port 8080
```

Key security measures:
- Non-root user (`appuser`)
- Minimal base image (`python:3.12-slim`)
- Built-in HEALTHCHECK directive

### 7.3 Local Verification

```bash
docker build -t heart-disease-api .
docker run -p 8080:8080 heart-disease-api

# Test health
curl http://localhost:8080/health

# Test prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```

---

## 8. Production Deployment

### 8.1 Kubernetes Manifests

Three Kubernetes manifests are provided in `k8s/`:

- **deployment.yaml**: 2 replicas, resource limits (256Mi-512Mi memory, 250m-500m CPU), liveness and readiness probes on `/health`, rolling update strategy.
- **service.yaml**: LoadBalancer type service exposing port 80 -> 8080.
- **ingress.yaml**: Optional Nginx ingress for path-based routing.

### 8.2 Minikube Deployment

```bash
minikube start
eval $(minikube docker-env)
docker build -t heart-disease-api:latest .
kubectl apply -f k8s/
minikube service heart-disease-api --url
```

### 8.3 Verification

After deployment:
- `kubectl get pods` - shows 2 running replicas
- `kubectl get svc` - shows LoadBalancer service with external URL
- `curl <URL>/health` - returns healthy status
- `curl -X POST <URL>/predict ...` - returns prediction

---

## 9. Monitoring & Logging

### 9.1 Structured Logging

All API requests are logged in JSON format with:
- HTTP method, path, status code
- Response latency (ms)
- UTC timestamp

### 9.2 Prometheus Metrics

The API exposes a `/metrics` endpoint via `prometheus-fastapi-instrumentator`:
- `http_requests_total`: Total request count by method/handler/status
- `http_request_duration_seconds`: Request latency histogram
- `http_requests_inprogress`: Currently processing requests
- `heart_disease_predictions_total`: Custom counter for prediction distribution (Disease/No Disease)

### 9.3 Grafana Dashboard

A pre-configured Grafana dashboard (`monitoring/grafana/dashboards/api-dashboard.json`) provides:
- Request rate (req/s) time series
- p95 latency time series
- Prediction distribution pie chart
- Error rate stat panel
- Requests in-progress gauge

### 9.4 Docker Compose Stack

```bash
docker compose up --build
```

Starts three services:
- **App** (port 8080): FastAPI prediction API
- **Prometheus** (port 9090): Metrics collection and storage
- **Grafana** (port 3000): Visualization dashboard (login: admin/admin)

---

## 10. Architecture Overview

```
                    ┌─────────────────────────────────────┐
                    │           Data Pipeline              │
                    │                                     │
                    │  UCI Dataset → download_data.py     │
                    │       → prepare.py → train/test CSV │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │         Training Pipeline            │
                    │                                     │
                    │  train.py → GridSearchCV (LR + RF)  │
                    │       → MLflow Experiment Tracking  │
                    │       → Best Model Artifact (.pkl)  │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │          CI/CD Pipeline              │
                    │                                     │
                    │  Git Push → Jenkins / GitHub Actions │
                    │       → Lint → Test → Train         │
                    │       → Docker Build → Push          │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │        Deployment & Serving          │
                    │                                     │
                    │  Docker Image → Kubernetes (Minikube)│
                    │       → FastAPI Service (2 replicas) │
                    │       → LoadBalancer / Ingress       │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │          Monitoring Stack            │
                    │                                     │
                    │  API → Prometheus (metrics scraping) │
                    │       → Grafana (dashboard)         │
                    │       → JSON structured logging     │
                    └─────────────────────────────────────┘
```

---

## Repository Link

GitHub: https://github.com/Deepesh-BITsPilani/mlops_heart_disease_assignment

## Demo Video

https://drive.google.com/file/d/1kcl4uasG1mN_f0jjZLTgqoPndGuqbrDt/view?usp=sharing

## Deliverables Checklist

- [x] Code, Dockerfile(s), requirements.txt
- [x] Cleaned dataset and download script
- [x] Jupyter notebooks (EDA, training)
- [x] test/ folder with unit tests (24 tests)
- [x] GitHub Actions workflow YAML and Jenkinsfile
- [x] Kubernetes deployment manifests
- [x] Screenshot folder for reporting
- [x] Final written report

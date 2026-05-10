"""Central configuration for the Heart Disease MLOps pipeline."""

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

TARGET_NAME = "target"
TARGET_LABELS = ["No Disease", "Disease"]

COLUMN_NAMES = FEATURE_NAMES + [TARGET_NAME]

RANDOM_STATE = 42
TEST_SIZE = 0.2

MODEL_PARAMS_RF = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
}

MODEL_PARAMS_LR = {
    "C": 1.0,
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
    "solver": "lbfgs",
}

MODEL_FILENAME = "heart_disease_model.pkl"
METRICS_FILENAME = "metrics.json"

MLFLOW_EXPERIMENT_NAME = "heart-disease-classification"
MLFLOW_TRACKING_URI = "file:./mlruns"

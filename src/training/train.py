"""
Heart Disease model training with MLflow experiment tracking.

Trains Logistic Regression and Random Forest classifiers,
performs hyperparameter tuning via GridSearchCV, logs all
experiments to MLflow, and saves the best model.
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import GridSearchCV, cross_validate  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
)
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import (  # noqa: E402
    FEATURE_NAMES, TARGET_NAME, RANDOM_STATE,
    MODEL_FILENAME, METRICS_FILENAME,
    MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI,
)


def build_pipeline(model):
    """Build a full sklearn pipeline: impute -> scale -> classify."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", model),
    ])


def evaluate_model(pipeline, X_test, y_test):
    """Compute evaluation metrics and return as dict."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    return metrics, y_pred, y_proba


def plot_confusion_matrix(y_test, y_pred, title, save_path):
    """Save a confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["No Disease", "Disease"],
        cmap="Blues", ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_roc_curve(y_test, y_proba, model_name, save_path):
    """Save an ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve - {model_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def cross_validate_model(pipeline, X, y):
    """Run 5-fold cross-validation and return mean scores."""
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv_results = cross_validate(
        pipeline, X, y, cv=5, scoring=scoring, return_train_score=False,
    )
    return {
        f"cv_{metric}": float(np.mean(cv_results[f"test_{metric}"]))
        for metric in scoring
    }


def train_and_log(model, model_name, X_train, y_train, X_test, y_test,
                  param_grid, artifact_dir):
    """Train a model with GridSearchCV, log everything to MLflow."""
    os.makedirs(artifact_dir, exist_ok=True)

    pipeline = build_pipeline(model)

    prefixed_grid = {
        f"classifier__{k}": v for k, v in param_grid.items()
    }

    grid_search = GridSearchCV(
        pipeline, prefixed_grid, cv=5, scoring="roc_auc",
        n_jobs=-1, verbose=0, refit=True,
    )
    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_

    metrics, y_pred, y_proba = evaluate_model(best_pipeline, X_test, y_test)
    cv_metrics = cross_validate_model(best_pipeline, X_train, y_train)

    cm_path = plot_confusion_matrix(
        y_test, y_pred, f"Confusion Matrix - {model_name}",
        os.path.join(artifact_dir, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"),
    )
    roc_path = plot_roc_curve(
        y_test, y_proba, model_name,
        os.path.join(artifact_dir, f"roc_curve_{model_name.lower().replace(' ', '_')}.png"),
    )

    with mlflow.start_run(run_name=model_name):
        best_params = {
            k.replace("classifier__", ""): v
            for k, v in grid_search.best_params_.items()
        }
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", model_name)

        mlflow.log_metrics(metrics)
        mlflow.log_metrics(cv_metrics)

        mlflow.set_tag("dataset", "heart-disease-uci-cleveland")
        mlflow.set_tag("dataset_version", "1.0")
        mlflow.set_tag("description", f"{model_name} with GridSearchCV tuning")

        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)

        mlflow.sklearn.log_model(best_pipeline, "model")

    print(f"\n  {model_name} Results:")
    print(f"    Best params: {best_params}")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")
    for k, v in cv_metrics.items():
        print(f"    {k}: {v:.4f}")

    return best_pipeline, metrics, cv_metrics


def run(train_path=None, test_path=None, artifact_dir=None):
    """Execute full training pipeline for both models."""
    project_root = os.path.join(os.path.dirname(__file__), "..", "..")

    if train_path is None:
        train_path = os.path.join(project_root, "data", "processed", "train.csv")
    if test_path is None:
        test_path = os.path.join(project_root, "data", "processed", "test.csv")
    if artifact_dir is None:
        artifact_dir = os.path.join(project_root, "model", "artifacts")

    print("=" * 60)
    print("STEP 2: MODEL TRAINING & EXPERIMENT TRACKING")
    print("=" * 60)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df[FEATURE_NAMES].values
    y_train = train_df[TARGET_NAME].values
    X_test = test_df[FEATURE_NAMES].values
    y_test = test_df[TARGET_NAME].values

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # --- Logistic Regression ---
    lr_model = LogisticRegression(random_state=RANDOM_STATE)
    lr_param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "max_iter": [1000],
        "solver": ["lbfgs"],
    }
    lr_pipeline, lr_metrics, lr_cv = train_and_log(
        lr_model, "Logistic Regression",
        X_train, y_train, X_test, y_test,
        lr_param_grid, artifact_dir,
    )

    # --- Random Forest ---
    rf_model = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    rf_pipeline, rf_metrics, rf_cv = train_and_log(
        rf_model, "Random Forest",
        X_train, y_train, X_test, y_test,
        rf_param_grid, artifact_dir,
    )

    # --- Select best model ---
    if lr_metrics["roc_auc"] >= rf_metrics["roc_auc"]:
        best_pipeline, best_name = lr_pipeline, "Logistic Regression"
        best_metrics = lr_metrics
    else:
        best_pipeline, best_name = rf_pipeline, "Random Forest"
        best_metrics = rf_metrics

    print(f"\n  Best model: {best_name} (ROC-AUC: {best_metrics['roc_auc']:.4f})")

    # Save best model
    model_path = os.path.join(artifact_dir, MODEL_FILENAME)
    joblib.dump(best_pipeline, model_path)
    print(f"  Model saved -> {model_path}")

    # Save metrics
    all_metrics = {
        "best_model": best_name,
        "logistic_regression": lr_metrics,
        "logistic_regression_cv": lr_cv,
        "random_forest": rf_metrics,
        "random_forest_cv": rf_cv,
    }
    metrics_path = os.path.join(artifact_dir, METRICS_FILENAME)
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Metrics saved -> {metrics_path}")

    # Combined ROC plot
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, pipe, color in [
        ("Logistic Regression", lr_pipeline, "#1f77b4"),
        ("Random Forest", rf_pipeline, "#ff7f0e"),
    ]:
        y_prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, linewidth=2, color=color,
                label=f"{name} (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    combined_roc_path = os.path.join(artifact_dir, "roc_curve_comparison.png")
    plt.savefig(combined_roc_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("  Training complete.")
    return best_pipeline, all_metrics


if __name__ == "__main__":
    run()

"""
Model evaluation module.

Loads a trained model and evaluates it on test data,
producing metrics and diagnostic plots.
"""
import os
import sys
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import FEATURE_NAMES, TARGET_NAME  # noqa: E402


def evaluate(model_path, test_path, output_dir=None):
    """Load model, evaluate on test set, return metrics."""
    project_root = os.path.join(os.path.dirname(__file__), "..", "..")
    if output_dir is None:
        output_dir = os.path.join(project_root, "model", "artifacts")

    model = joblib.load(model_path)
    test_df = pd.read_csv(test_path)

    X_test = test_df[FEATURE_NAMES].values
    y_test = test_df[TARGET_NAME].values

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    report = classification_report(y_test, y_pred, target_names=["No Disease", "Disease"])
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"\n{report}")

    return metrics, report, cm


def run(model_path=None, test_path=None):
    """Execute full evaluation."""
    project_root = os.path.join(os.path.dirname(__file__), "..", "..")
    if model_path is None:
        model_path = os.path.join(project_root, "model", "artifacts", "heart_disease_model.pkl")
    if test_path is None:
        test_path = os.path.join(project_root, "data", "processed", "test.csv")

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    metrics, report, cm = evaluate(model_path, test_path)
    return metrics


if __name__ == "__main__":
    run()

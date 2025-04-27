"""modeling.py

Functions for training, evaluating, and logging models with MLflow.
"""

import time
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train model, compute metrics, produce plots, and return results dict."""
    # Train time
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Metrics
    accuracy     = accuracy_score(y_test, y_pred)
    precision    = precision_score(y_test, y_pred)
    recall       = recall_score(y_test, y_pred)
    f1           = f1_score(y_test, y_pred)
    roc_auc      = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    avg_precision = average_precision_score(y_test, y_pred_proba) if y_pred_proba is not None else None

    # Print
    print(f"\n--- {model_name} Results ---")
    print(f"Training time: {train_time:.2f}s")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Avg Precision: {avg_precision:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Neg','Pos'], yticklabels=['Neg','Pos'])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.show()

    # ROC & PR Curves
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
        plt.plot([0,1],[0,1],'k--')
        plt.title(f"ROC Curve - {model_name}")
        plt.xlabel('FPR'); plt.ylabel('TPR')
        plt.legend(); plt.show()

        pr, rc, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.figure(figsize=(6, 4))
        plt.plot(rc, pr, label=f'AP = {avg_precision:.4f}')
        plt.title(f"Precision-Recall Curve - {model_name}")
        plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.legend(); plt.show()

    return {
        'model': model,
        'train_time': train_time,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision
    }

def log_model_to_mlflow(model, results: dict, model_name: str, params: dict = None):
    """Log parameters, metrics, and model artifact to MLflow."""
    with mlflow.start_run(run_name=model_name):
        if params:
            mlflow.log_params(params)
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'avg_precision', 'train_time']:
            val = results.get(metric)
            if val is not None:
                mlflow.log_metric(metric, val)
        if isinstance(model, xgb.XGBModel):
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")
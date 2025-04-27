"""
drift_detection.py

Detect dataset and classification drift between a reference and current dataset
using Evidently, and log the HTML report to MLflow.
"""

import pandas as pd
import mlflow
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric, ClassificationDriftMetric

def detect_drift(
    reference_path: str,
    current_path: str,
    column_mapping: dict,
    output_html: str = "drift_report.html",
    mlflow_experiment: str = "DriftDetection"
):
    """
    Loads reference and current datasets, computes drift metrics, saves
    an HTML report, and logs it to MLflow.

    Parameters
    ----------
    reference_path : str
        Path to the reference dataset (CSV or Parquet).
    current_path : str
        Path to the current dataset (CSV or Parquet).
    column_mapping : dict
        Evidently column mapping, e.g.:
          {
            "target": "target",
            "prediction": "prediction",
            "numerical_features": [...],
            "categorical_features": [...]
          }
    output_html : str, default="drift_report.html"
        Filename for the saved Evidently HTML report.
    mlflow_experiment : str, default="DriftDetection"
        Name of the MLflow experiment under which to log the run.
    """
    # Set MLflow experiment
    mlflow.set_experiment(mlflow_experiment)

    # Load datasets
    if reference_path.endswith(".parquet"):
        ref = pd.read_parquet(reference_path)
    else:
        ref = pd.read_csv(reference_path)

    if current_path.endswith(".parquet"):
        curr = pd.read_parquet(current_path)
    else:
        curr = pd.read_csv(current_path)

    # Start MLflow run
    with mlflow.start_run(run_name="drift_check"):
        # Build Evidently report
        report = Report(metrics=[
            DatasetDriftMetric(),
            ClassificationDriftMetric()
        ])
        report.run(reference_data=ref, current_data=curr, column_mapping=column_mapping)

        # Save and log HTML report
        report.save_html(output_html)
        mlflow.log_artifact(output_html)
        print(f"Drift report saved and logged as MLflow artifact: {output_html}")

if __name__ == "__main__":
    # Example usage
    mapping = {
        "target": "target",
        "prediction": "prediction",
        "numerical_features": ["goal", "campaign_duration", "goal_log", "goal_per_day"],
        "categorical_features": ["staff_pick", "main_category_encoded", "country_encoded"]
    }
    detect_drift(
        reference_path="data/reference.parquet",
        current_path="data/current.parquet",
        column_mapping=mapping,
        output_html="drift_report.html"
    )
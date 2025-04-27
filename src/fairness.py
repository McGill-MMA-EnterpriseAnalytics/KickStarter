"""
fairness.py

Utilities for evaluating model fairness across sensitive attributes
using Fairlearn metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    equalized_odds_difference
)
from sklearn.metrics import accuracy_score

def evaluate_fairness_by_group(
    y_true: pd.Series,
    y_pred: pd.Series,
    X_test_df: pd.DataFrame,
    protected_vars: list,
    plot: bool = True
) -> pd.DataFrame:
    """
    Evaluate fairness metrics by group for one or more protected attributes.

    For each attribute in `protected_vars`, computes:
      - Accuracy by group
      - Selection rate by group
      - Demographic parity difference
      - Equalized odds difference

    Optionally plots the per‐group metrics.

    Args:
        y_true: True labels (array‐like).
        y_pred: Predicted labels (array‐like).
        X_test_df: DataFrame containing the same index as y_true/y_pred,
                   with columns for each protected variable.
        protected_vars: List of column names in X_test_df to treat as sensitive features.
        plot: If True, generates a bar chart of accuracy and selection rate by group.

    Returns:
        A pandas DataFrame with columns:
          - attribute
          - demographic_parity_diff
          - equalized_odds_diff
    """
    results = []

    for var in protected_vars:
        print(f"\n--- Evaluating fairness by '{var}' ---")
        group = X_test_df[var].astype(str)

        # Build a MetricFrame
        mf = MetricFrame(
            metrics={
                "accuracy": accuracy_score,
                "selection_rate": selection_rate
            },
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=group
        )

        dp = demographic_parity_difference(
            y_true, y_pred, sensitive_features=group
        )
        eo = equalized_odds_difference(
            y_true, y_pred, sensitive_features=group
        )

        # Append summary metrics
        results.append({
            "attribute": var,
            "demographic_parity_diff": dp,
            "equalized_odds_diff": eo
        })

        # Print group‐level metrics
        print(mf.by_group)
        print(f"Demographic Parity Difference: {dp:.4f}")
        print(f"Equalized Odds Difference:      {eo:.4f}")

        # Plot if requested
        if plot:
            ax = mf.by_group.plot(
                kind="bar",
                figsize=(10, 5),
                title=f"Fairness Metrics by '{var}'",
                legend=True
            )
            ax.set_ylabel("Metric Value")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    # Return a summary DataFrame
    return pd.DataFrame(results)
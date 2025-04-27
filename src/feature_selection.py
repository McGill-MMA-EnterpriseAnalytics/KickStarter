"""
feature_selection.py

Provides feature selection utilities:
- select_via_rf: Random Forest importance–based selection
- select_via_anova: ANOVA F-test–based selection
- select_via_rfecv: Recursive Feature Elimination with Cross-Validation
"""

import pandas as pd
from typing import List
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

def select_via_rf(X: pd.DataFrame, y: pd.Series, n_features: int) -> List[str]:
    """
    Select the top `n_features` based on Random Forest feature importances.

    Args:
        X: feature DataFrame
        y: target Series
        n_features: number of features to select

    Returns:
        List of selected feature names.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    selected = importances.sort_values(ascending=False).head(n_features).index.tolist()
    return selected

def select_via_anova(X: pd.DataFrame, y: pd.Series, k: int) -> List[str]:
    """
    Select the top `k` features based on ANOVA F-test scores.

    Args:
        X: feature DataFrame
        y: target Series
        k: number of features to select

    Returns:
        List of selected feature names.
    """
    skb = SelectKBest(score_func=f_classif, k=k)
    skb.fit(X, y)
    selected = X.columns[skb.get_support()].tolist()
    return selected

def select_via_rfecv(X: pd.DataFrame, y: pd.Series, min_features: int = 10) -> List[str]:
    """
    Perform Recursive Feature Elimination with Cross-Validation (RFECV).

    Args:
        X: feature DataFrame
        y: target Series
        min_features: minimum number of features to select

    Returns:
        List of selected feature names.
    """
    lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    rfecv = RFECV(
        estimator=lr,
        step=1,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc',
        min_features_to_select=min_features,
        n_jobs=-1
    )
    rfecv.fit(X, y)
    selected = X.columns[rfecv.support_].tolist()
    return selected
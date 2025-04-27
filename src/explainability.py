"""
explainability.py

Functions to analyze and visualize feature importance and learning curves
for the selected best model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import learning_curve, StratifiedKFold

def plot_feature_importance(best_model, model_name, X_train, X_test=None,
                            feature_names=None, shap_sample_size=500):
    """
    Analyze and plot feature importance for the best model.

    Args:
        best_model: trained estimator
        model_name: name string of the model
        X_train: DataFrame or array used for training
        X_test: DataFrame or array for SHAP sampling (optional)
        feature_names: list of feature names
        shap_sample_size: max samples for SHAP
    """
    print(f"\nAnalyzing feature importance for the best model: {model_name}")

    # Determine feature list
    if feature_names is None and hasattr(X_train, 'columns'):
        feature_names = X_train.columns.tolist()
    elif feature_names is None:
        feature_names = [f"f{i}" for i in range(X_train.shape[1])]

    # Logistic Regression
    if hasattr(best_model, 'coef_') and 'Logistic' in model_name:
        coefs = np.abs(best_model.coef_[0])
        df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': coefs})
        df_imp = df_imp.sort_values('Importance', ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=df_imp.head(20))
        plt.title(f'Feature Importance for {model_name}')
        plt.tight_layout()
        plt.show()

    # Tree-based models
    elif hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        df_imp = df_imp.sort_values('Importance', ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=df_imp.head(20))
        plt.title(f'Feature Importance for {model_name}')
        plt.tight_layout()
        plt.show()

        # SHAP analysis
        if X_test is not None:
            print("\nCalculating SHAP values for interpretation...")
            try:
                if isinstance(X_test, pd.DataFrame):
                    shap_sample = X_test.sample(min(shap_sample_size, len(X_test)), random_state=42)
                else:
                    shap_sample = X_test[:shap_sample_size]
                explainer = shap.Explainer(best_model)
                shap_values = explainer(shap_sample)

                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, shap_sample, feature_names=feature_names, show=False)
                plt.title(f'SHAP Summary Plot for {model_name}')
                plt.tight_layout()
                plt.show()

                top_feats = df_imp['Feature'].head(2).tolist()
                for feat in top_feats:
                    plt.figure(figsize=(10, 6))
                    shap.dependence_plot(feat, shap_values.values, shap_sample,
                                         feature_names=feature_names, show=False)
                    plt.title(f'SHAP Dependence Plot for {feat}')
                    plt.tight_layout()
                    plt.show()
            except Exception as e:
                print(f"Error calculating SHAP values: {e}")
                print("Continuing without SHAP analysis...")

    # MLP models
    elif hasattr(best_model, 'coefs_'):
        weights = best_model.coefs_[0]
        imp = np.mean(np.abs(weights), axis=1)
        df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': imp})
        df_imp = df_imp.sort_values('Importance', ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=df_imp.head(20))
        plt.title(f'Feature Importance for {model_name} (MLP coefficients)')
        plt.tight_layout()
        plt.show()

    else:
        print(f"No built-in feature importance available for {model_name}, skipping this step.")

def plot_learning_curves(best_model, X_train, y_train, 
                         train_sizes=None, cv_splits=5, scoring='roc_auc', n_jobs=-1):
    """
    Plot learning curves for the best model.

    Args:
        best_model: trained estimator
        X_train: training feature set
        y_train: training labels
        train_sizes: array of training sizes
        cv_splits: number of CV folds
        scoring: scoring metric
        n_jobs: parallel jobs
    """
    print(f"\nAnalyzing learning curves for the best model: {best_model}")

    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    train_sizes, train_scores, test_scores = learning_curve(
        best_model, X_train, y_train,
        train_sizes=train_sizes,
        cv=StratifiedKFold(n_splits=cv_splits),
        scoring=scoring,
        n_jobs=n_jobs
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores, axis=1)
    test_mean  = np.mean(test_scores, axis=1)
    test_std   = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    plt.title(f'Learning Curves for {type(best_model).__name__}')
    plt.xlabel('Training Set Size')
    plt.ylabel(scoring)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    gap = train_mean[-1] - test_mean[-1]
    print(f"Final gap between training and validation scores: {gap:.4f}")

    if gap > 0.05:
        print("The model shows signs of overfitting (high training score, lower validation score). Consider more regularization or simpler model.")
    elif test_mean[-1] < 0.7:
        print("The model shows signs of underfitting (low validation score). Consider more complex model or additional features.")
    else:
        print("The model shows good fit (similar training and validation scores at a good level).")
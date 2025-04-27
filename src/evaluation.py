# src/evaluation.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def make_comparison_df(results: dict, optimized_prefix: bool = False) -> pd.DataFrame:
    """
    Convert a dict of model results into a DataFrame.
    
    Args:
        results: dict mapping model names to metrics dicts.
        optimized_prefix: if True, prefix model names with 'Optimized '.
        
    Returns:
        DataFrame with columns:
          ['Model','Accuracy','Precision','Recall','F1 Score','ROC AUC',
           'Avg Precision','Training Time (s)']
    """
    data = {
        'Model': [
            (f"Optimized {name}" if optimized_prefix else name)
            for name in results.keys()
        ],
        'Accuracy':         [res['accuracy']       for res in results.values()],
        'Precision':        [res['precision']      for res in results.values()],
        'Recall':           [res['recall']         for res in results.values()],
        'F1 Score':         [res['f1']             for res in results.values()],
        'ROC AUC':          [res['roc_auc']        for res in results.values()],
        'Avg Precision':    [res['avg_precision']  for res in results.values()],
        'Training Time (s)':[res['train_time']     for res in results.values()],
    }
    return pd.DataFrame(data)

def combine_comparisons(
    baseline_df: pd.DataFrame,
    optimized_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Concatenate baseline and optimized DataFrames and sort by ROC AUC descending.
    
    Args:
        baseline_df: DataFrame of baseline results.
        optimized_df: DataFrame of optimized results.
    
    Returns:
        Combined DataFrame sorted by 'ROC AUC'.
    """
    combined = pd.concat([baseline_df, optimized_df], ignore_index=True)
    return combined.sort_values('ROC AUC', ascending=False).reset_index(drop=True)

def plot_roc_auc_comparison(
    baseline_df: pd.DataFrame,
    optimized_df: pd.DataFrame,
    metric: str = 'ROC AUC'
):
    """
    Bar-plot comparing a metric (e.g. ROC AUC) between baseline and optimized models.
    
    Args:
        baseline_df: DataFrame with baseline results (has 'Model' & metric col).
        optimized_df: DataFrame with optimized results.
        metric: column name to compare.
    """
    b = baseline_df[['Model', metric]].copy()
    b['Type'] = 'Baseline'
    o = optimized_df[['Model', metric]].copy()
    o['Type'] = 'Optimized'
    # strip redundant prefix for readability
    o['Model'] = o['Model'].str.replace('Optimized ', '', regex=False)

    df = pd.concat([b, o], ignore_index=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y=metric, hue='Type', data=df)
    plt.title(f'Baseline vs Optimized {metric}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_best_model_metrics(
    combined_df: pd.DataFrame,
    metrics: list[str] = None
):
    """
    Bar-plot of multiple metrics for the top-ranked model in combined_df.
    
    Args:
        combined_df: DataFrame sorted by primary metric (highest first).
        metrics: list of metric column names to plot (default common set).
    """
    if metrics is None:
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    best_model = combined_df.iloc[0]['Model']
    best_vals = combined_df.loc[
        combined_df['Model'] == best_model, metrics
    ].melt(var_name='Metric', value_name='Value')

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Metric', y='Value', data=best_vals)
    plt.title(f'Metrics for Best Model: {best_model}')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

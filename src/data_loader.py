# src/data_loader.py

import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_clean_data(
    filepath: str,
    target_col: str = "target",
    drop_cols: list[str] = ["state"],
    drop_non_numeric: bool = True,
    drop_na: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load a CSV into a DataFrame, clean it, and split out features and target.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    target_col : str, default="target"
        Name of the target column in the CSV.
    drop_cols : list[str], default=["state"]
        Other columns to drop (e.g. 'state').
    drop_non_numeric : bool, default=True
        If True, drop all non-numeric columns.
    drop_na : bool, default=True
        If True, drop any rows with missing values.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    """
    # 1) Load
    df = pd.read_csv(filepath)

    # 2) Drop rows with any nulls
    if drop_na:
        df = df.dropna()

    # 3) Drop non-numeric columns
    if drop_non_numeric:
        non_numeric = df.select_dtypes(exclude=["int64", "float64"]).columns
        df = df.drop(columns=non_numeric)

    # 4) Extract target
    y = df[target_col]

    # 5) Drop target and any other specified cols
    X = df.drop(columns=[target_col, *drop_cols], errors="ignore")

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split features and target into train/test sets.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Seed used by the random number generator.
    stratify : bool, default=True
        If True, stratify splits by y to preserve class balance.

    Returns
    -------
    X_train : pd.DataFrame
    X_test : pd.DataFrame
    y_train : pd.Series
    y_test : pd.Series
    """
    strat = y if stratify else None
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat
    )
#!/usr/bin/env python3
# preprocessing.py

import pandas as pd
import numpy as np
import mlflow
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA

def main(
    input_path: str = "Kickstarter.xlsx",
    raw_output_path: str = "kickstarter_processed_with_categorical.csv",
    final_output_path: str = "kickstarter_final_processed.csv"
):
    """
    End-to-end preprocessing for the Kickstarter dataset:
    1. Load & filter
    2. Leakage removal
    3. Missing-value imputation
    4. Date/time feature engineering
    5. Numeric, cyclic, campaign & goal features
    6. Categorical encoding (one-hot, target-encoding, label)
    7. Entity embeddings
    8. PCA
    9. Final cleanup & save
    """
    mlflow.start_run(run_name="preprocessing")

    # --------------------
    # 1) Load & initial filter
    # --------------------
    df = pd.read_excel(input_path)
    df = df[df['state'].isin(['successful', 'failed'])]
    df['target'] = (df['state'] == 'successful').astype(int)
    df = df.drop(columns=['id', 'name'], errors='ignore')

    # --------------------
    # 2) Leakage removal
    # --------------------
    leakage_features = [
        'pledged', 'backers_count', 'usd_pledged',
        'state_changed_at', 'state_changed_at_weekday',
        'state_changed_at_month', 'state_changed_at_day',
        'state_changed_at_yr', 'state_changed_at_hr',
        'spotlight'
    ]
    if 'staff_pick.1' in df.columns:
        leakage_features.append('staff_pick.1')
    to_drop = [f for f in leakage_features if f in df.columns]
    mlflow.log_param("removed_leakage_features", to_drop)
    df = df.drop(columns=to_drop, errors='ignore')

    # --------------------
    # 3) Missing-value imputation
    # --------------------
    # Identify columns by type
    numeric_cols     = df.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object','bool']).columns.tolist()
    datetime_cols    = df.select_dtypes(include=['datetime64']).columns.tolist()

    # Numeric KNN imputation
    numeric_missing = [c for c in numeric_cols if df[c].isna().any()]
    if numeric_missing:
        imputer_cols = numeric_missing + ['goal']
        scaler = StandardScaler()
        X_imp = scaler.fit_transform(df[imputer_cols].fillna(df[imputer_cols].median()))
        imputer = KNNImputer(n_neighbors=5)
        X_filled = imputer.fit_transform(X_imp)
        X_unscaled = scaler.inverse_transform(X_filled)
        for i, col in enumerate(imputer_cols):
            df[col] = X_unscaled[:, i]
        print(f"Imputed numeric columns: {numeric_missing}")
    else:
        print("No numeric missing values found.")

    # Categorical mode imputation
    categorical_missing = [c for c in categorical_cols if df[c].isna().any()]
    for col in categorical_missing:
        df[col] = df[col].fillna(df[col].mode().iloc[0])
    mlflow.log_param("missing_numeric_columns",     len(numeric_missing))
    mlflow.log_param("missing_categorical_columns", len(categorical_missing))

    # --------------------
    # 4) Date/time features
    # --------------------
    date_cols = ['deadline','created_at','launched_at']
    for col in date_cols:
        if col in df.columns:
            df[f'{col}_datetime'] = pd.to_datetime(df[col])

    df['campaign_duration'] = (
        df['deadline_datetime'] - df['launched_at_datetime']
    ).dt.total_seconds() / 86400
    df['preparation_time'] = (
        df['launched_at_datetime'] - df['created_at_datetime']
    ).dt.total_seconds() / 86400

    for col in date_cols:
        if f'{col}_datetime' in df:
            df[f'{col}_is_weekend'] = df[f'{col}_datetime'].dt.dayofweek >= 5

    # --------------------
    # 5) Cyclic encoding
    # --------------------
    for col in date_cols:
        dt = df[f'{col}_datetime']
        if dt is not None:
            month = dt.dt.month
            day   = dt.dt.day
            hour  = dt.dt.hour
            df[f'{col}_month_sin'] = np.sin(2*np.pi*month/12)
            df[f'{col}_month_cos'] = np.cos(2*np.pi*month/12)
            df[f'{col}_day_sin']   = np.sin(2*np.pi*day/31)
            df[f'{col}_day_cos']   = np.cos(2*np.pi*day/31)
            df[f'{col}_hour_sin']  = np.sin(2*np.pi*hour/24)
            df[f'{col}_hour_cos']  = np.cos(2*np.pi*hour/24)

    # --------------------
    # 6) Campaign & goal features
    # --------------------
    df['same_day_launch'] = (
        df['created_at_datetime'].dt.date ==
        df['launched_at_datetime'].dt.date
    ).astype(int)
    df['ideal_duration']  = (
        df['campaign_duration'].between(30,40)
    ).astype(int)

    df['goal_per_day']  = df['goal'] / df['campaign_duration']
    df['goal_log']      = np.log1p(df['goal'])
    bins = np.percentile(df['goal'], [0,10,25,50,75,90,95,99,100])
    df['goal_percentile_bin'] = pd.cut(df['goal'], bins=bins, labels=False, include_lowest=True)
    if 'static_usd_rate' in df:
        df['goal_usd_adjusted'] = df['goal'] * df['static_usd_rate']

    # --------------------
    # 7) Target-encoding on train split
    # --------------------
    np.random.seed(42)
    mask = np.random.rand(len(df)) < 0.8
    global_mean = df.loc[mask,'target'].mean()

    for col in ['main_category','category','country']:
        if col in df:
            rates = df.loc[mask].groupby(col)['target'].mean().to_dict()
            df[f'{col}_success_rate'] = df[col].map(rates).fillna(global_mean)
    if 'country' in df:
        counts = df['country'].value_counts().to_dict()
        df['country_project_count']     = df['country'].map(counts)
        df['country_project_count_log'] = np.log1p(df['country_project_count'])

    # --------------------
    # 8) Text-length & ratio features
    # --------------------
    if {'name_len','blurb_len'}.issubset(df.columns):
        df['name_blurb_ratio'] = df['name_len'] / df['blurb_len'].replace(0,1)
    if {'name_len_clean','name_len'}.issubset(df.columns):
        df['name_efficiency']  = df['name_len_clean'] / df['name_len'].replace(0,1)
    if {'blurb_len_clean','blurb_len'}.issubset(df.columns):
        df['blurb_efficiency'] = df['blurb_len_clean'] / df['blurb_len'].replace(0,1)

    # --------------------
    # 9) One-hot & label encoding
    # --------------------
    # Identify categoricals
    categorical = [
        c for c in df.columns
        if (df[c].dtype == 'object' or c.endswith('_weekday'))
        and c not in ['state','target']
    ]
    # Drop currency if duplicate
    if 'currency' in df:
        df.drop(columns=['currency'], inplace=True)
        categorical = [c for c in categorical if c!='currency']

    # Low-card: one-hot
    low_card  = [c for c in categorical if df[c].nunique()<10]
    for c in low_card:
        dummies = pd.get_dummies(df[c], prefix=c, drop_first=True)
        df = pd.concat([df, dummies], axis=1)

    # High-card: label then embeddings
    high_card = [c for c in categorical if df[c].nunique()>=10]
    df_label  = df.copy()
    le = LabelEncoder()
    label_cols = []
    for c in high_card:
        df[c+'_label'] = le.fit_transform(df[c])
        label_cols.append(c+'_label')

    # --------------------
    # 10) Entity embeddings via TF
    # --------------------
    emb_created = 0
    for col in label_cols:
        n_uniques = df[col].nunique()
        if n_uniques < 10:
            continue
        emb_dim = min(50, int(np.sqrt(n_uniques)))
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(1,)),
            tf.keras.layers.Embedding(input_dim=n_uniques+1, output_dim=emb_dim),
            tf.keras.layers.Flatten()
        ])
        emb = model.predict(df[col].values.reshape(-1,1), verbose=0)
        for i in range(emb_dim):
            df[f'{col}_emb_{i}'] = emb[:,i]
        emb_created += 1
    mlflow.log_param("entity_embeddings_created", emb_created)

    # --------------------
    # 11) PCA on numeric features
    # --------------------
    numeric = df.select_dtypes(include=['int64','float64']).drop(columns=['target'])
    if numeric.shape[1] > 20:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(numeric)
        pca = PCA(n_components=0.95, random_state=42)
        comps = pca.fit_transform(scaled)
        mlflow.log_param("pca_components_added", comps.shape[1])
        for i in range(comps.shape[1]):
            df[f'pca_comp_{i}'] = comps[:,i]
    else:
        mlflow.log_param("pca_components_added", 0)

    # --------------------
    # 12) Final cleanup & save
    # --------------------
    df = df.dropna().reset_index(drop=True)
    df.to_csv(raw_output_path, index=False)
    df.to_csv(final_output_path, index=False)

    # Log final shape
    mlflow.log_param("n_rows", df.shape[0])
    mlflow.log_param("n_cols", df.shape[1])
    mlflow.log_param("target_balance", df['target'].mean())

    mlflow.log_artifact(raw_output_path)
    mlflow.log_artifact(final_output_path)

    mlflow.end_run()
    print("Preprocessing complete. Files saved:" , raw_output_path, final_output_path)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
pipeline.py

End-to-end orchestration:
 1. (Optional) Run raw preprocessing to CSV
 2. Load & clean processed CSV
 3. Split, scale, feature‐select
 4. Train & evaluate baseline models
 5. Hyperparameter tuning
 6. Train & evaluate optimized models
 7. Compare & plot results
 8. Fairness analysis
 9. Explainability (feature importance & learning curves)
10. Save final pipeline + feature list
"""

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

from data_loader    import load_and_clean_data, split_data
from preprocessing  import main as run_preprocessing
from feature_selection import select_via_rfecv
from tuning         import tune_all
from modeling       import evaluate_model, log_model_to_mlflow
from evaluation     import make_comparison_df, combine_comparisons, plot_roc_auc_comparison, plot_best_model_metrics
from fairness       import evaluate_fairness_by_group
from explainability import plot_feature_importance, plot_learning_curves

def main():
    # 1) Preprocessing (if starting from raw Excel)
    # run_preprocessing()

    # 2) Load preprocessed data
    X, y = load_and_clean_data("kickstarter_final_processed.csv")
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 3) Scale
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # 4) Feature selection via RFECV
    Xtr_df = pd.DataFrame(X_train_s, columns=X_train.columns)
    selected_features = select_via_rfecv(Xtr_df, y_train, min_features=10)
    X_train_sel = Xtr_df[selected_features].values
    X_test_sel  = pd.DataFrame(X_test_s, columns=X_train.columns)[selected_features].values

    # 5) Baseline models
    baseline_models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest":       RandomForestClassifier(random_state=42),
        "XGBoost":             xgb.XGBClassifier(
                                   random_state=42,
                                   use_label_encoder=False,
                                   eval_metric="logloss",
                                   tree_method="hist",
                                   gpu_id=0
                              ),
        "MLPClassifier":       MLPClassifier(
                                   hidden_layer_sizes=(100,),
                                   activation="relu",
                                   solver="adam",
                                   max_iter=200,
                                   random_state=42
                              )
    }
    baseline_results = {}
    for name, model in baseline_models.items():
        res = evaluate_model(model, X_train_sel, X_test_sel, y_train, y_test, name)
        baseline_results[name] = res
        log_model_to_mlflow(model, res, f"{name} Baseline")

    baseline_df = make_comparison_df(baseline_results, optimized_prefix=False)

    # 6) Hyperparameter tuning
    optimized_models = tune_all(pd.DataFrame(X_train_sel, columns=selected_features), y_train)

    # 7) Evaluate optimized
    optimized_results = {}
    for name, model in optimized_models.items():
        run_name = f"Optimized {name}"
        res = evaluate_model(model, X_train_sel, X_test_sel, y_train, y_test, run_name)
        optimized_results[name] = res
        log_model_to_mlflow(model, res, run_name)

    optimized_df = make_comparison_df(optimized_results, optimized_prefix=True)

    # 8) Compare & plot
    combined_df = combine_comparisons(baseline_df, optimized_df)
    plot_roc_auc_comparison(baseline_df, optimized_df)
    plot_best_model_metrics(combined_df)

    # 9) Fairness on best optimized
    best_name = combined_df.iloc[0]["Model"].replace("Optimized ", "")
    best_model = optimized_models[best_name]
    y_pred = best_model.predict(X_test_sel)
    test_df_sel = pd.DataFrame(X_test_sel, columns=selected_features)
    fairness_df = evaluate_fairness_by_group(
        y_test, y_pred, test_df_sel,
        protected_vars=["staff_pick","goal_percentile_bin","main_category_encoded"],
        plot=True
    )

    # 10) Explainability
    plot_feature_importance(
        best_model, f"Optimized {best_name}",
        pd.DataFrame(X_train_sel, columns=selected_features),
        X_test_df=test_df_sel,
        feature_names=selected_features
    )
    plot_learning_curves(
        best_model,
        pd.DataFrame(X_train_sel, columns=selected_features),
        y_train
    )

    # 11) Save final pipeline & feature list
    pipeline = Pipeline([("scaler", scaler), ("model", best_model)])
    joblib.dump(pipeline, "models/best_pipeline.pkl")
    with open("selected_features.txt","w") as f:
        for feat in selected_features:
            f.write(feat + "\n")
    print("✅ Pipeline complete. Artifacts saved to models/best_pipeline.pkl and selected_features.txt")

if __name__ == "__main__":
    main()

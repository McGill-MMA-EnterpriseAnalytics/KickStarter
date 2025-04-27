"""tuning.py

Hyperparameter tuning routines using Optuna.
"""

import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier

def optimize_logistic_regression(X, y, n_trials=20, cv_splits=5):
    def objective(trial):
        C = trial.suggest_float('C', 0.01, 10.0, log=True)
        solver = trial.suggest_categorical('solver', ['liblinear','lbfgs'])
        penalty = 'l2'
        if solver == 'liblinear':
            penalty = trial.suggest_categorical('penalty',['l1','l2'])
        model = LogisticRegression(C=C, solver=solver, penalty=penalty,
                                   max_iter=1000, random_state=42, n_jobs=-1)
        return cross_val_score(
            model, X, y,
            cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42),
            scoring='roc_auc', n_jobs=-1
        ).mean()
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    return LogisticRegression(**study.best_params, max_iter=1000, random_state=42, n_jobs=-1)

def optimize_random_forest(X, y, n_trials=20, cv_splits=5):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators',50,300),
            'max_depth': trial.suggest_int('max_depth',3,20),
            'min_samples_split': trial.suggest_int('min_samples_split',2,20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf',1,10),
            'max_features': trial.suggest_categorical('max_features',['sqrt','log2',None])
        }
        model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
        return cross_val_score(model, X, y,
            cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42),
            scoring='roc_auc', n_jobs=-1).mean()
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    return RandomForestClassifier(random_state=42, n_jobs=-1, **study.best_params)

def optimize_xgboost(X, y, n_trials=20, cv_splits=5):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators',50,300),
            'max_depth': trial.suggest_int('max_depth',3,15),
            'learning_rate': trial.suggest_float('learning_rate',0.01,0.3,log=True),
            'subsample': trial.suggest_float('subsample',0.6,1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree',0.6,1.0),
            'min_child_weight': trial.suggest_int('min_child_weight',1,10),
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'gpu_id': 0,
            'random_state': 42,
            'verbosity': 0
        }
        model = xgb.XGBClassifier(**params)
        return cross_val_score(model, X, y,
            cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42),
            scoring='roc_auc', n_jobs=-1).mean()
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    best.update({'use_label_encoder': False, 'eval_metric': 'logloss',
                 'tree_method': 'hist', 'gpu_id': 0,
                 'random_state': 42, 'verbosity': 0})
    return xgb.XGBClassifier(**best)

def optimize_mlp(X, y, n_trials=20, cv_splits=5):
    def objective(trial):
        params = {
            'hidden_layer_sizes': (trial.suggest_int('hidden_layer_size',50,200),),
            'learning_rate_init': trial.suggest_loguniform('learning_rate_init',1e-4,1e-1),
            'alpha': trial.suggest_loguniform('alpha',1e-5,1e-1),
            'solver': 'adam',
            'max_iter': 200,
            'random_state': 42
        }
        model = MLPClassifier(**params)
        return cross_val_score(model, X, y,
            cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42),
            scoring='roc_auc', n_jobs=-1).mean()
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    return MLPClassifier(hidden_layer_sizes=(best['hidden_layer_size'],),
                         learning_rate_init=best['learning_rate_init'],
                         alpha=best['alpha'],
                         solver='adam',
                         max_iter=200,
                         random_state=42)

def tune_all(X, y, n_trials=20, cv_splits=5):
    """Run all tuning routines and return a dict of optimized estimators."""
    return {
        'logistic_regression': optimize_logistic_regression(X, y, n_trials, cv_splits),
        'random_forest':       optimize_random_forest(X, y, n_trials, cv_splits),
        'xgboost':             optimize_xgboost(X, y, n_trials, cv_splits),
        'mlp':                 optimize_mlp(X, y, n_trials, cv_splits)
    }

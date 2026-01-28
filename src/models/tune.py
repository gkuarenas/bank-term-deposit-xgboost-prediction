import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd

def tune_model(X: pd.Dataframe, y: pd.Series):
    '''
    
    Tunes an XGBoost model using Optuna

    Args:
        X (pd.Dataframe): Features
        y (pd.Series): Target
    
    '''
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            # "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            # "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss"
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv=3, scoring="recall")
        return scores.mean()
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best Params:", study.best_params)
    return study.best_params
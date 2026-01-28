import mlflow
import pandas as pd
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import time

def split_model(df: pd.DataFrame, target_col: str):
    """
    Trains an XGBoost model and logs with MLflow.

    Args:
        df (pd.DataFrame): Feature dataset.
        target_col (str): Name of the target column.
    """

    X = df.drop(columns=[target_col])
    y=df[target_col]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=0.1,
        stratify=y,
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=1/6,
        stratify=y_trainval,
        random_state=42
    )

    print("\nSplitting data into 75% Train, 15% Validation, and 10% Test")
    print(f"Train: {X_train.shape[0]} counts")
    print(f"Val: {X_val.shape[0]} counts")
    print(f"Test: {X_test.shape[0]} counts")

    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(df: pd.DataFrame, X_train, X_val, y_train, y_val):

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )

    # ===== TRAIN MODEL ===== #
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    print(f"Training time for unoptimized XGBoost Classifier: {train_time} seconds.")
    
    # ===== RUN PREDICTION ======#
    start_pred = time.time()
    pred = model.predict(X_val)
    pred_time = time.time() - start_pred
    print(f"Prediction time for XGBoost Classifier: {pred_time} seconds.")
    prec = precision_score(y_val, pred)
    rec = recall_score(y_val, pred)
    f1 = f1_score(y_val, pred)

    print(f"Model trained. Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")


    return model, {"precision": prec, "recall": rec, "f1": f1}

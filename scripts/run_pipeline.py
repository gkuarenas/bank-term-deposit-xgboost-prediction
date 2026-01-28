from pathlib import Path
import sys
import mlflow
import mlflow.sklearn
import argparse
from sklearn.metrics import classification_report

from xgboost import XGBClassifier
import time
from sklearn.metrics import precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
print(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# ===== Local modules ===== #
from src.data.load_data import load_data
from src.data.preprocessing import preprocess_data
from src.utils.validate_data import validate_data
from src.features.build_features import build_features
from src.models.train import split_model, train_model

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bank-full.csv"
TARGET_COL = 'y'

def main(args):

    # ===== MLFlow Setup for Experiment Tracking ===== #
    mlruns_dir = Path(PROJECT_ROOT) / "mlruns"
    mlruns_path = args.mlflow_uri or mlruns_dir.as_uri()
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name):

        # ===== 1. Load & Validate Dataset =====
        print("Loading data...")
        df = load_data(DATA_PATH)
        print(f"Data loaded.\nShape: {df.shape}")
        print(df.head(3))

        # Data Validation
        ok, failed = validate_data(df)

        if not ok:
            raise ValueError(f"Data validation failed. Failed expectations: {failed}")

        # ===== 2. Preprocessing data =====
        print("\nPreprocessing data...")
        df = preprocess_data(df)
        
        PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "bank-full_processed.csv"
        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_PATH, index=False)
        print(f"Dataset saved to {PROCESSED_PATH} |\nShape: {df.shape}")

        # ===== 3. Building Features ===== #
        print("\nBuilding Features...")
        df = build_features(df, TARGET_COL)
        print(df.head(3))
        
        for c in df.select_dtypes(include=["bool"]).columns:
            df[c] = df[c].astype(int)
        
        print(f"Feature engineering complete: {df.shape[1]} final features.")

        # ===== CRITICAL: Save Feature Metadata for Serving Consistency ===
        # This ensures serving pipeline uses exact same features in exact same order
        import json, joblib
        artifacts_dir = Path(PROJECT_ROOT) / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Get feature columns excluding target
        feature_cols = list(df.drop(columns=[TARGET_COL]).columns)

        # Save locally for development serving
        feature_json_path = artifacts_dir / "feature_columns.json"
        with feature_json_path.open("w", encoding="utf-8") as f:
            json.dump(feature_cols, f)
        
        # Log to MLflow for production serving
        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")

        # ESSENTIAL: Save preprocessing artifacts for serving pipeline
        # These artifacts ensure training and serving use identical transformation
        preprocessing_artifact = {
            "feature_columns": feature_cols,    # Exact feature order
            "target": TARGET_COL                # Target column name
        }
        
        pkl_path = artifacts_dir / "preprocessing.pkl"
        joblib.dump(preprocessing_artifact, str(pkl_path))
        mlflow.log_artifact(str(pkl_path))
        print(f"Saved {len(feature_cols)} feature columns for serving consistency.")

        # ===== 4. Split Model and Train (Optimized Parameters) ===== #
        print("\nTraining XGBoost model...")

        X_train, X_val, X_test, y_train, y_val, y_test = split_model(df, TARGET_COL)

        # === CRITICAL: Handle Class Imbalance ===
        # Calculate scale_pos_weight to handle imbalanced dataset
        # This tells XGBoost to give more weight to the minority class (subscribers)
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"Class imbalance ratio: {scale_pos_weight:.2f} (applied to positive class)")

        params = {
            # Tree structure parameters
            'n_estimators': 977,
            'learning_rate': 0.1300500912577778, 
            'max_depth': 10,

            # Regularization parameters
            'subsample': 0.9269698779148268,
            'colsample_bytree': 0.587735104054278,

            # Performance parameters
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss',

            'scale_pos_weight': scale_pos_weight
        }

        # Log all parameters
        mlflow.log_params(params)

        model = XGBClassifier(**params)
        

        # ===== TRAIN MODEL ===== #
        start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_train
        mlflow.log_metric("train_time", train_time)
        print(f"Training time for unoptimized XGBoost Classifier: {train_time:.2f} seconds.")
       
        # ===== 5. Evaluate model performance ======#
        print("Evaluating model performance...")

        start_pred = time.time()
        proba = model.predict_proba(X_test)[:,1] # Probability of Yes 

        pred = (proba >= args.threshold).astype(int)

        pred_time = time.time() - start_pred
        mlflow.log_metric("pred_time", pred_time)
        print(f"Prediction time for XGBoost Classifier: {pred_time:2f} seconds.")

        prec = precision_score(y_test, pred)
        rec = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)

        print(f"\nModel trained. Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        
        # Log all metrics for experiment tracking
        mlflow.log_metrics({
            "precision": prec,
            "recall": rec,
            "f1": f1
        })

        print("Model Performance")
        print(f"    Precision: {prec:.4f}   |   Recall: {rec:.4f}")
        print(f"    F1-score: {f1:.4f}")

        # ===== 7. Model Serialization and Logging =====
        print("\n Saving model to MLflow...")
        
        mlflow.sklearn.log_model(model, artifact_path="model")

        print("Model saved to MLflow for serving pipeline")

        '''# Logging dataset for MLflow UI
        train_ds = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_input(train_ds, context="training")'''

        # ===== FINAL PERFORMANCE SUMMARY ===== #
        print("\n===============================")
        print("===== PERFORMANCE SUMMARY =====")
        print("===============================")
        print(f"    Training time: {train_time:.2f}s")
        print(f"    Inference time: {pred_time:4f}s")

        print("===============================")
        print(" Detailed Classification Report")
        print("===============================")
        print(classification_report(y_test, pred, digits=4))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Run Bank Subscription Pipeline with XGBoost + MLflow')
    p.add_argument("--input", type=str, required=None,
                   help="path to CSV (e.g., data/raw/bank-full.csv)")
    p.add_argument("--target", type=str, default='y')
    p.add_argument("--threshold", type=float, default=0.45)
    p.add_argument("--mlflow_uri", type=str, default=None,
                   help="override MLflow tracking URI, else uses PROJECT_ROOT/mlruns")
    p.add_argument("--experiment", type=str, default="Bank Term Deposit")
    p.add_argument("--run_name", type=str, default=None)
    
    args = p.parse_args()
    main(args)


"""
Run this script as

python scripts/run_pipeline.py --mlflow_uri http://127.0.0.1:5000 --run_name run_n

Note: Make sure to change n in run_n to latest

"""
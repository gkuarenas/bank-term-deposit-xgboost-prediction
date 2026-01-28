import pathlib as Path
import pandas as pd
import mlflow

MODEL_DIR = "/app/model"

try:
    # Load the trained XGBoost model in MLflow pyfunc format
    # This ensures compatibility regardless of the underlying ML library
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"Model loaded successfuly from {MODEL_DIR}")
except Exception as e:
    print(f"Failed to load model from {MODEL_DIR}: {e}")
    # Try loading from local MLflow tracking
    try:
        import glob
        local_model_paths = list(Path("./mlruns").glob("*/*/artifacts.model"))
        if local_model_paths:
            latest_model = max(local_model_paths, key=lambda p: p.stat().st_mtime)
            model = mlflow.pyfunc.load_model(str(latest_model))
            MODEL_DIR = latest_model
            print(f"Fallback: Loaded model from {latest_model}")
        else:
            raise Exception("No model found in local mlruns")
    except Exception as fallback_error:
        raise Exception(f"Failed to load model: {e}. Fallback failed: {fallback_error}")

# ===== FEATURE SCHEMA LOADING ===== #
# CRITICAL: Load the exact feature column order used during training
# This ensures the model receives featured in the expected order
try:
    feature_file = Path(MODEL_DIR) / "feature_columns.txt"
    FEATURE_COLS = [ln.strip() for ln in feature_file.read_text().splitlines() if ln.strip()]
    print(f"Loaded {len(FEATURE_COLS)} feature columns from training")
except Exception as e:
    raise Exception(f"Failed to load feature columns: {e}")

# ===== FEATURE TRANSFORMATION CONSTANTS =====
# CRITICAL: These mappings must exactly match those used in training
# Any changes here will cause train/serve skew and degrade model performance

# Deterministic binary feature mappings (consistent with training)

BINARY_MAP = {
    "default": {"no": 0, "yes": 1},
    "housing": {"no": 0, "yes": 1},
    "loan": {"no": 0, "yes": 1},
}
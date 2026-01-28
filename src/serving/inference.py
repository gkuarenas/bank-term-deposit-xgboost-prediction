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

NUMERIC_COLS = {
    "age", "day", "duration", "campaign", 
    "pdays", "previous"
}

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply identical feature transformations as used during model training.
    
    This function is CRITICAL for production ML - it ensures that features are
    transformed exactly as they were during training to prevent train/serve skew.
    
    Transformation Pipeline:
    1. Clean column names and handle data types
    2. Apply deterministic binary encoding (using BINARY_MAP)
    3. One-hot encode remaining categorical features  
    4. Convert boolean columns to integers
    5. Align features with training schema and order
    
    Args:
        df: Single-row DataFrame with raw customer data
        
    Returns:
        DataFrame with features transformed and ordered for model input
        
    IMPORTANT: Any changes to this function must be reflected in training
    feature engineering to maintain consistency.
    """

    df = df.copy()

    # Clean column names
    df.columns = df.columns.str.strip()

    # ===== Step 1: Numeric Type Coercion ===== #
    # Ensure numeric columns are properly typed (handle string inputs)
    for c in NUMERIC_COLS:
        if c in df.columns:
            # Convert to numeric, replace invalid values with NaN
            df[c] = pd.to_numeric(df[c], errors='coerce')
            # Fill NaN with 0
            df[c] = df[c].fillna(0)

    # === STEP 2: Binary Feature Encoding ===
    # Apply deterministic mappings for binary features
    # CRITICAL: Must use exact same mappings as training
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)                    # Convert to string
                .str.strip()                    # Remove whitespace
                .map(mapping)                   # Apply binary mapping
                .astype("Int64")                # Handle NaN values
                .fillna(0)                      # Fill unknown values with 0
                .astype(int)                    # Final integer conversion
            )
    
    # === STEP 3: One-Hot Encoding for Remaining Categorical Features ===
    # Find remaining object/categorical columns (not in BINARY_MAP)
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        # Apply one-hot encoding with drop_first=True (same as training)
        # This prevents multicollinearity by dropping the first category
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    
    # === STEP 4: Boolean to Integer Conversion ===
    # Convert any boolean columns to integers (XGBoost compatibility)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
    
    # === STEP 5: Feature Alignment with Training Schema ===
    # CRITICAL: Ensure features are in exact same order as training
    # Missing features get filled with 0, extra features are dropped
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    
    return df

def predict(input_dict: dict) -> str:
    """
    Main prediction function for bank term deposit subscription inference.
    
    This function provides the complete inference pipeline from raw customer data
    to business-friendly prediction output. It's called by both the FastAPI endpoint
    and the Gradio interface to ensure consistent predictions.
    
    Pipeline:
    1. Convert input dictionary to DataFrame
    2. Apply feature transformations (identical to training)
    3. Generate model prediction using loaded XGBoost model
    4. Convert prediction to user-friendly string

    Args:
        input_dict: Dictionary containing raw customer data with keys matching
                   the CustomerData schema (16 features total)

    Returns:
        Human-readable prediction string:
        - "Likely to subscribe" model prediction = 1)
        - "Not likely to subscribe" (model prediction = 0)
    """

    # === STEP 1: Convert Input to DataFrame ===
    # Create single-row DataFrame for pandas transformations
    df = pd.DataFrame([input_dict])

    # === STEP 2: Apply Feature Transformations ===
    # Use the same transformation pipeline as training
    df_enc = _serve_transform(df)
    
    # === STEP 3: Generate Model Prediction ===
    # Call the loaded MLflow model for inference
    # The model returns predictions in various formats depending on the ML library
    try:
        preds = model.predict(df_enc)
        
        # Normalize prediction output to consistent format
        if hasattr(preds, "tolist"):
            preds = preds.tolist()  # Convert numpy array to list
            
        # Extract single prediction value (for single-row input)
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds
            
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")
    
    # === STEP 4: Convert to Business-Friendly Output ===
    # Convert binary prediction (0/1) to actionable business language
    if result == 1:
        return "Likely to subscribe"      # High risk - needs intervention
    else:
        return "Not likely to subscribe"  # Low risk - maintain normal service

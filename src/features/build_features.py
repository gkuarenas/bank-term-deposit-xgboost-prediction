import pandas as pd

def _map_binary_series(s: pd.Series) -> pd.Series:
    '''
    Deterministic binary encoding to 2-category features.

    This function implements the core binary encoding logic that converts
    categorical features with exactly 2 values into 0/1 integers. The mappings
    are deterministic and must be consistent between training and serving.
    '''

    # Get unique values and remove NaN, coerce to string then make set
    vals = (
        s.astype('string')
        .str.strip()
        .str.lower()
    )
    valset = set(vals.dropna().unique().tolist())

    # ===== DETERMINISTIC BINARY MAPPINGS ===== #

    # Yes/No mapping
    if valset == {"yes", "no"}:
        return s.map({
            "no": 0,
            "yes": 1
        }).astype("Int64")
    
    # ===== NON-BINARY FEATURES ===== #
    # No changes - one-hot encoding

    return s

def build_features(df: pd.DataFrame, target_col: str='y') -> pd.DataFrame:
    """
    Apply complete feature engineering pipeline for training data.
    
    This is the main feature engineering function that transforms raw customer data
    into ML-ready features. The transformations must be exactly replicated in the
    serving pipeline to ensure prediction accuracy.

    """
    df = df.copy()
    print(f"Starting feature engineering on {df.shape[1]} columns...")

    # ===== 1. Identify Feature Types ===== #
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print(f"Found {len(obj_cols)} categorical and {len(numeric_cols)} numeric columns.")

    # ===== 2. Split Categorical by Cardinality ===== #
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cols = [c for c in obj_cols if df[c]. dropna().nunique () > 2]

    # ===== 3. Apply Binary Encoding ===== #
    for c in binary_cols:
        df[c] = _map_binary_series(df[c])

    # ===== 4. Convert Boolean Columns ===== #
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)

    # ===== 5. One-hot Encoding ===== #
    if multi_cols:
        print(f"Applying one-hot encoding to {len(multi_cols)} multi-category columns...")
        original_shape = df.shape

        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

        new_features = df.shape[1] - original_shape[1] + len(multi_cols)
        print(f"Created {new_features} new features from {len(multi_cols)} categorical columns.")

    # ====== 6. Data Type Cleanup ===== #
    # Convert nullable integers (Int64) to standard integers for XGBoost
    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            # Fill any NaN values with 0 and convert to int
            df[c] = df[c].fillna(0).astype(int)


    return df
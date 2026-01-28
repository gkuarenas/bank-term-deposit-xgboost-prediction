import pandas as pd

def preprocess_data(df: pd.DataFrame, target_col: str='y') -> pd.DataFrame:
    '''
    Data cleaning done for this dataset:
    - binary (yes/no) encoding to 0/1

    No missing values
    '''

    # Convert target column to 0/1
    if target_col in df.columns and df[target_col].dtype == 'object':
        df[target_col] = (
            df[target_col]
            .astype("string")
            .str.strip()
            .str.lower()
            .map({"no":0, "yes":1})
        )
    return df
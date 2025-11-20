import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and null values."""
    if df is None:
        raise ValueError("Input DataFrame cannot be None")

    df = df.drop_duplicates()
    df = df.dropna()

    return df

import pandas as pd

def normalize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Normalize specified column using Min-Max scaling."""
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")

    col_min = df[column].min()
    col_max = df[column].max()

    df[column] = (df[column] - col_min) / (col_max - col_min)

    return df

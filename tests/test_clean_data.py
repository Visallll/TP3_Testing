import pandas as pd
from src.data_cleaning import clean_data

def test_remove_duplicates():
    df = pd.DataFrame({"A": [1, 1, 2]})
    cleaned = clean_data(df)
    assert len(cleaned) == 2

def test_remove_nulls():
    df = pd.DataFrame({"A": [1, None, 3]})
    cleaned = clean_data(df)
    assert cleaned.isnull().sum().sum() == 0

def test_rows_decrease():
    df = pd.DataFrame({"A": [1, 1, None]})
    cleaned = clean_data(df)
    assert len(cleaned) < len(df)

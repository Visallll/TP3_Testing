import pandas as pd
import pytest
from src.normalization import normalize_column

def test_normalized_range():
    df = pd.DataFrame({"value": [10, 20, 30]})
    result = normalize_column(df, "value")
    assert result["value"].between(0, 1).all()

def test_length_unchanged():
    df = pd.DataFrame({"value": [10, 20, 30]})
    result = normalize_column(df, "value")
    assert len(result) == 3

def test_invalid_column():
    df = pd.DataFrame({"value": [1, 2, 3]})
    with pytest.raises(KeyError):
        normalize_column(df, "wrong")

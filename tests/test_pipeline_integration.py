import pandas as pd
from src.pipeline import train_model, load_data, pipeline_evaluate

def test_load_data(tmp_path):
    # Create temporary CSV
    file = tmp_path / "data.csv"
    file.write_text("a,b,label\n1,2,0\n3,4,1")

    df = load_data(file)
    assert not df.empty
    assert "label" in df.columns

def test_train_model():
    df = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4],
        "label": [0, 1]
    })
    model = train_model(df)
    assert model is not None

def test_pipeline_accuracy(tmp_path):
    file = tmp_path / "data.csv"
    file.write_text("a,b,label\n1,5,0\n2,4,0\n3,3,1\n4,2,1")
    result = pipeline_evaluate(file)
    assert 0 <= result["accuracy"] <= 1

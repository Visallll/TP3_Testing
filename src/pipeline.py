import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.evaluation import evaluate_model


def load_data(filepath="D:\I5\I5_Mine\APDS\TP\TP3\data.csv"):
    """Load dataset from CSV file."""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        raise FileNotFoundError("CSV file not found. Please check the filepath.")

def train_model(df):
    """Train a Logistic Regression model."""
    if "label" not in df.columns:
        raise KeyError("Dataset must contain a 'label' column.")

    X = df.drop(columns=["label"])
    y = df["label"]

    model = LogisticRegression()
    model.fit(X, y)

    return model

def pipeline_evaluate(filepath="D:\I5\I5_Mine\APDS\TP\TP3\data.csv"):
    """End-to-end ML pipeline evaluation."""
    df = load_data(filepath)
    model = train_model(df)

    X = df.drop(columns=["label"])
    y_true = df["label"]
    y_pred = model.predict(X)

    return evaluate_model(y_true, y_pred)

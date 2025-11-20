from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(y_true, y_pred):
    """Return accuracy and F1 score in a dictionary."""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "accuracy": accuracy,
        "f1_score": f1
    }

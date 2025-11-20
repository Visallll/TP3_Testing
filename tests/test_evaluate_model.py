from src.evaluation import evaluate_model

def test_accuracy_perfect():
    assert evaluate_model([1, 0], [1, 0])["accuracy"] == 1.0

def test_f1_zero():
    assert evaluate_model([1, 1], [0, 0])["f1_score"] == 0.0

def test_keys_exist():
    result = evaluate_model([1], [1])
    assert "accuracy" in result
    assert "f1_score" in result

from train import model, X_test, y_test
from sklearn.metrics import accuracy_score

def test_accuracy():
    acc = accuracy_score(y_test, model.predict(X_test))
    assert acc > 0.8  # Test will fail if accuracy is under 80%

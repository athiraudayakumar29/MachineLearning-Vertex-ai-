"""
evaluate.py
Loads the trained model and evaluates it on the held-out test set,
printing a confusion matrix and accuracy score.
"""

import logging

import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from config import CONFIG
from preprocessing import run_preprocessing

logger = logging.getLogger(__name__)


def load_model(path: str = CONFIG.MODEL_PATH):
    return joblib.load(path)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print("Confusion matrix:")
    print(cm)
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return y_pred, cm, acc


if __name__ == "__main__":
    logging.basicConfig(level=CONFIG.LOG_LEVEL)

    X_train, X_test, y_train, y_test = run_preprocessing()

    classifier = load_model()
    evaluate_model(classifier, X_test, y_test)
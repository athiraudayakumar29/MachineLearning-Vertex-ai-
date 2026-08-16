"""
train.py
Trains a Logistic Regression classifier on the wine quality data
and saves the fitted model to disk (or GCS, once MODEL_PATH is a gs:// URI
— wired up on Day 4 when the container/Vertex job is built).
"""

import logging

import joblib
from sklearn.linear_model import LogisticRegression

from config import CONFIG
from preprocessing import run_preprocessing

logger = logging.getLogger(__name__)


def train_model(X_train, y_train, max_iter: int = CONFIG.MAX_ITER):
    """Fit a LogisticRegression classifier."""
    classifier = LogisticRegression(
        max_iter=max_iter, random_state=CONFIG.RANDOM_STATE
    )
    classifier.fit(X_train, y_train)
    return classifier


def save_model(model, path: str = CONFIG.MODEL_PATH):
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)


if __name__ == "__main__":
    logging.basicConfig(level=CONFIG.LOG_LEVEL)

    X_train, X_test, y_train, y_test = run_preprocessing()

    classifier = train_model(X_train, y_train)
    save_model(classifier)
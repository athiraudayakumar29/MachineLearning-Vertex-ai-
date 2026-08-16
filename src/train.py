"""
train.py
Trains a Logistic Regression classifier on the wine quality data
and saves the fitted model to disk (or GCS, once MODEL_PATH is a gs:// URI).

CLI usage:
    python train.py
    python train.py --max-iter 2000 --data-path ../data/winequality.csv
    python train.py --model-path model_v2.joblib --random-state 7
"""

import argparse
import json
import logging
from datetime import datetime, timezone

import joblib
from sklearn.linear_model import LogisticRegression

from config import CONFIG
from preprocessing import run_preprocessing

logger = logging.getLogger(__name__)

RUNS_LOG_PATH = "runs.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the wine-quality classifier.")
    parser.add_argument(
        "--data-path", default=CONFIG.DATA_PATH,
        help="Path to the training CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--model-path", default=CONFIG.MODEL_PATH,
        help="Where to save the trained model (default: %(default)s)",
    )
    parser.add_argument(
        "--max-iter", type=int, default=CONFIG.MAX_ITER,
        help="Max iterations for LogisticRegression solver (default: %(default)s)",
    )
    parser.add_argument(
        "--random-state", type=int, default=CONFIG.RANDOM_STATE,
        help="Random seed for reproducibility (default: %(default)s)",
    )
    parser.add_argument(
        "--test-size", type=float, default=CONFIG.TEST_SIZE,
        help="Fraction of data held out for testing (default: %(default)s)",
    )
    parser.add_argument(
        "--runs-log", default=RUNS_LOG_PATH,
        help="Where to append this run's params (default: %(default)s)",
    )
    return parser.parse_args()


def train_model(X_train, y_train, max_iter: int = CONFIG.MAX_ITER, random_state: int = CONFIG.RANDOM_STATE):
    """Fit a LogisticRegression classifier."""
    classifier = LogisticRegression(max_iter=max_iter, random_state=random_state)
    classifier.fit(X_train, y_train)
    return classifier


def save_model(model, path: str = CONFIG.MODEL_PATH):
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)


def log_run(params: dict, log_path: str = RUNS_LOG_PATH):
    """Append this run's params to a local JSONL file for basic reproducibility
    tracking. This is a lightweight stand-in for Vertex AI Experiments, wired
    up properly once the pipeline moves to GCP.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **params,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info("Run logged to %s", log_path)


if __name__ == "__main__":
    logging.basicConfig(level=CONFIG.LOG_LEVEL)
    args = parse_args()

    X_train, X_test, y_train, y_test = run_preprocessing(
        path=args.data_path, test_size=args.test_size, random_state=args.random_state
    )

    classifier = train_model(
        X_train, y_train, max_iter=args.max_iter, random_state=args.random_state
    )
    save_model(classifier, path=args.model_path)

    log_run(
        {
            "data_path": args.data_path,
            "model_path": args.model_path,
            "max_iter": args.max_iter,
            "random_state": args.random_state,
            "test_size": args.test_size,
            "n_features": X_train.shape[1],
            "n_train_rows": len(X_train),
            "n_test_rows": len(X_test),
        },
        log_path=args.runs_log,
    )
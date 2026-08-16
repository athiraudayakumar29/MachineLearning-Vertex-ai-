"""
evaluate.py
Loads a trained model and evaluates it on the held-out test set: computes
accuracy, F1, ROC-AUC (where applicable), and a confusion matrix. Checks
the results against a configurable quality gate (MIN_ACCURACY, MIN_F1) and
optionally compares against a previous "champion" model's metrics.

CLI usage:
    python evaluate.py
    python evaluate.py --model-path model.joblib --min-accuracy 0.55
    python evaluate.py --champion-metrics metrics_champion.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from config import CONFIG
from preprocessing import run_preprocessing

logger = logging.getLogger(__name__)

METRICS_OUT_PATH = "metrics.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the wine-quality classifier against a quality gate.")
    parser.add_argument("--data-path", default=CONFIG.DATA_PATH)
    parser.add_argument("--model-path", default=CONFIG.MODEL_PATH)
    parser.add_argument("--test-size", type=float, default=CONFIG.TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=CONFIG.RANDOM_STATE)
    parser.add_argument("--min-accuracy", type=float, default=CONFIG.MIN_ACCURACY,
                         help="Quality gate threshold for accuracy (default: %(default)s)")
    parser.add_argument("--min-f1", type=float, default=CONFIG.MIN_F1,
                         help="Quality gate threshold for weighted F1 (default: %(default)s)")
    parser.add_argument("--metrics-out", default=METRICS_OUT_PATH,
                         help="Where to write this run's metrics as JSON (default: %(default)s)")
    parser.add_argument("--champion-metrics", default=None,
                         help="Path to a previous metrics.json to compare against as the 'champion' model")
    parser.add_argument("--no-gate", action="store_true",
                         help="Skip the pass/fail gate check (still computes and prints metrics)")
    return parser.parse_args()


def load_model(path: str = CONFIG.MODEL_PATH):
    return joblib.load(path)


def compute_metrics(model, X_test, y_test) -> dict:
    """Compute accuracy, weighted F1, multi-class ROC-AUC, and confusion matrix."""
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    roc_auc = None
    try:
        if hasattr(model, "predict_proba"):
            classes = sorted(y_test.unique())
            if len(classes) > 2:
                y_test_bin = label_binarize(y_test, classes=classes)
                y_proba = model.predict_proba(X_test)
                roc_auc = roc_auc_score(y_test_bin, y_proba, average="weighted", multi_class="ovr")
            else:
                y_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba)
    except ValueError as e:
        logger.warning("Could not compute ROC-AUC: %s", e)

    return {
        "accuracy": acc,
        "f1_weighted": f1,
        "roc_auc_weighted": roc_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "n_test_rows": len(y_test),
    }


def print_report(metrics: dict):
    print("Confusion matrix:")
    for row in metrics["confusion_matrix"]:
        print(row)
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
    if metrics["roc_auc_weighted"] is not None:
        print(f"ROC-AUC (weighted): {metrics['roc_auc_weighted']:.4f}")
    else:
        print("ROC-AUC: not computed")


def check_quality_gate(metrics: dict, min_accuracy: float, min_f1: float) -> bool:
    """Return True if the model clears both thresholds."""
    passed = metrics["accuracy"] >= min_accuracy and metrics["f1_weighted"] >= min_f1
    print(f"\n--- Quality gate ---")
    print(f"Accuracy: {metrics['accuracy']:.4f} (min {min_accuracy}) -> {'PASS' if metrics['accuracy'] >= min_accuracy else 'FAIL'}")
    print(f"F1:       {metrics['f1_weighted']:.4f} (min {min_f1}) -> {'PASS' if metrics['f1_weighted'] >= min_f1 else 'FAIL'}")
    print(f"Overall: {'PASS' if passed else 'FAIL'}")
    return passed


def compare_to_champion(metrics: dict, champion_path: str):
    """Compare this run's metrics to a previously saved champion metrics.json."""
    try:
        with open(champion_path) as f:
            champion = json.load(f)
    except FileNotFoundError:
        logger.warning("Champion metrics file not found at %s; skipping comparison.", champion_path)
        return

    print(f"\n--- Challenger vs. champion ({champion_path}) ---")
    for key in ("accuracy", "f1_weighted"):
        challenger_val = metrics.get(key)
        champion_val = champion.get(key)
        if challenger_val is None or champion_val is None:
            continue
        delta = challenger_val - champion_val
        direction = "better" if delta > 0 else ("worse" if delta < 0 else "same")
        print(f"{key}: challenger={challenger_val:.4f} vs champion={champion_val:.4f} ({direction}, delta={delta:+.4f})")


def save_metrics(metrics: dict, path: str):
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **metrics}
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    logger.info("Metrics saved to %s", path)


if __name__ == "__main__":
    logging.basicConfig(level=CONFIG.LOG_LEVEL)
    args = parse_args()

    X_train, X_test, y_train, y_test = run_preprocessing(
        path=args.data_path, test_size=args.test_size, random_state=args.random_state
    )

    classifier = load_model(args.model_path)
    metrics = compute_metrics(classifier, X_test, y_test)
    print_report(metrics)
    save_metrics(metrics, args.metrics_out)

    if args.champion_metrics:
        compare_to_champion(metrics, args.champion_metrics)

    if not args.no_gate:
        passed = check_quality_gate(metrics, args.min_accuracy, args.min_f1)
        sys.exit(0 if passed else 1)
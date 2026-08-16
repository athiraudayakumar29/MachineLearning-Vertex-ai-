"""
Unit tests for train.py: train_model and save_model, using a small
synthetic dataset. Also keeps the original import-sanity check.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config  # noqa: E402
from train import train_model, save_model  # noqa: E402


def test_imports():
    import preprocessing  # noqa: F401
    import evaluate  # noqa: F401

    assert config.CONFIG.MAX_ITER == 1000
    assert config.CONFIG.TARGET_COL == "quality"


@pytest.fixture
def synthetic_train_data():
    rng = np.random.default_rng(1)
    n = 100
    X_train = pd.DataFrame(
        {
            "alcohol": rng.random(n) * 15,
            "acidity": rng.random(n) * 10,
        }
    )
    y_train = (X_train["alcohol"] > 7.5).astype(int)
    return X_train, y_train


def test_train_model_returns_fitted_classifier(synthetic_train_data):
    X_train, y_train = synthetic_train_data
    model = train_model(X_train, y_train, max_iter=200, random_state=42)
    # A fitted LogisticRegression exposes these attributes
    assert hasattr(model, "coef_")
    preds = model.predict(X_train)
    assert len(preds) == len(y_train)


def test_train_model_is_reproducible_with_same_seed(synthetic_train_data):
    X_train, y_train = synthetic_train_data
    model_a = train_model(X_train, y_train, max_iter=200, random_state=42)
    model_b = train_model(X_train, y_train, max_iter=200, random_state=42)
    assert np.allclose(model_a.coef_, model_b.coef_)


def test_save_model_writes_file(tmp_path, synthetic_train_data):
    X_train, y_train = synthetic_train_data
    model = train_model(X_train, y_train, max_iter=200, random_state=42)
    out_path = tmp_path / "model.joblib"
    save_model(model, path=str(out_path))
    assert out_path.exists()
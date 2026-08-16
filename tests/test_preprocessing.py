"""
Unit tests for preprocessing.py, using a small synthetic dataset so tests
run fast and don't depend on the real wine CSV being present.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing import get_low_corr_columns, preprocess, split_data  # noqa: E402


@pytest.fixture
def synthetic_df():
    n = 200
    # Deterministic, evenly-spaced alcohol values so correlation with quality
    # is exact and not subject to random sampling noise across test runs.
    alcohol = np.linspace(0, 15, n)
    quality = (alcohol > 7.5).astype(int)
    # noise_feature is a fixed repeating pattern uncorrelated with alcohol/quality
    noise = np.array([0, 1, 0, 1] * (n // 4))
    return pd.DataFrame(
        {
            "alcohol": alcohol,
            "noise_feature": noise,
            "quality": quality,
        }
    )


def test_get_low_corr_columns_drops_noise(synthetic_df):
    low_corr = get_low_corr_columns(synthetic_df, target_col="quality", threshold=0.10)
    assert "noise_feature" in low_corr
    assert "alcohol" not in low_corr


def test_preprocess_returns_X_y_without_target(synthetic_df):
    X, y = preprocess(synthetic_df, target_col="quality", drop_low_corr=False)
    assert "quality" not in X.columns
    assert len(X) == len(y) == len(synthetic_df)


def test_preprocess_drops_low_corr_when_enabled(synthetic_df):
    X, _ = preprocess(synthetic_df, target_col="quality", threshold=0.10, drop_low_corr=True)
    assert "noise_feature" not in X.columns
    assert "alcohol" in X.columns


def test_split_data_shapes(synthetic_df):
    X, y = preprocess(synthetic_df, target_col="quality", drop_low_corr=False)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_test) == len(synthetic_df)


def test_split_data_is_reproducible(synthetic_df):
    X, y = preprocess(synthetic_df, target_col="quality", drop_low_corr=False)
    split_a = split_data(X, y, test_size=0.2, random_state=42)
    split_b = split_data(X, y, test_size=0.2, random_state=42)
    pd.testing.assert_frame_equal(split_a[0], split_b[0])
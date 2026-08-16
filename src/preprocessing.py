"""
preprocessing.py
Loads the wine quality dataset, drops low-correlation features,
scales the remaining features, and splits it into train/test sets.
"""

import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import CONFIG

logger = logging.getLogger(__name__)


def load_data(path: str = CONFIG.DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame."""
    logger.info("Loading data from %s", path)
    data = pd.read_csv(path, sep=";")
    return data


def get_low_corr_columns(
    data: pd.DataFrame,
    target_col: str = CONFIG.TARGET_COL,
    threshold: float = CONFIG.LOW_CORR_THRESHOLD,
):
    """Return column names whose correlation with the target is below the threshold."""
    cormatrix = data.corr()[target_col]
    low_corr = cormatrix[abs(cormatrix) < threshold].index
    return low_corr


def preprocess(
    data: pd.DataFrame,
    target_col: str = CONFIG.TARGET_COL,
    threshold: float = CONFIG.LOW_CORR_THRESHOLD,
    drop_low_corr: bool = True,
):
    """
    Split into X (features) and y (target), optionally dropping
    low-correlation features. Returns (X, y).
    """
    y = data[target_col]
    X = data.drop(target_col, axis=1)

    if drop_low_corr:
        low_corr_cols = get_low_corr_columns(data, target_col, threshold)
        if len(low_corr_cols) > 0:
            logger.info("Dropping low-correlation columns: %s", list(low_corr_cols))
        X = X.drop(low_corr_cols, axis=1, errors="ignore")

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = CONFIG.TEST_SIZE,
    random_state: int = CONFIG.RANDOM_STATE,
):
    """Train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def run_preprocessing(path: str = CONFIG.DATA_PATH):
    """Convenience function: load -> preprocess -> split -> scale."""
    data = load_data(path)

    # Sanity checks (mirrors the original notebook exploration)
    assert data.isna().sum().sum() == 0, "Unexpected missing values in dataset"

    X, y = preprocess(data)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Scale features (fit on train only, to avoid leaking test info into train)
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    logger.info(
        "Split data: %d train rows, %d test rows, %d features",
        len(X_train), len(X_test), X_train.shape[1],
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    logging.basicConfig(level=CONFIG.LOG_LEVEL)
    X_train, X_test, y_train, y_test = run_preprocessing()
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("Features used:", list(X_train.columns))
"""
config.py
Centralized configuration for the wine-quality training pipeline.

All paths/params are overridable via environment variables so the same
code runs locally, in a Docker container, and as a Vertex AI Custom Job
without code changes. This replaces the hardcoded constants that used to
live at the top of preprocessing.py / train.py / test.py.
"""

import os
from dataclasses import dataclass


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass(frozen=True)
class Config:
    # --- Data ---
    # NOTE: original code used a Windows-style path (data\winequality.csv).
    # Using forward slashes here so this also works on Linux (Docker/Vertex).
    DATA_PATH: str = _env("DATA_PATH", "data/winequality.csv")
    TARGET_COL: str = _env("TARGET_COL", "quality")
    LOW_CORR_THRESHOLD: float = float(_env("LOW_CORR_THRESHOLD", "0.10"))

    # --- Model artifact ---
    # Local default keeps existing behavior; in GCP this becomes a GCS URI,
    # e.g. gs://my-bucket/models/wine-quality/model.joblib
    MODEL_PATH: str = _env("MODEL_PATH", "model.joblib")

    # --- Training hyperparameters ---
    MAX_ITER: int = int(_env("MAX_ITER", "1000"))
    RANDOM_STATE: int = int(_env("RANDOM_STATE", "42"))
    TEST_SIZE: float = float(_env("TEST_SIZE", "0.20"))

    # --- Quality gate (used by evaluate.py / Day 3) ---
    MIN_ACCURACY: float = float(_env("MIN_ACCURACY", "0.70"))
    MIN_F1: float = float(_env("MIN_F1", "0.65"))

    # --- GCP / Vertex settings ---
    PROJECT_ID: str = _env("GCP_PROJECT_ID", "")
    REGION: str = _env("GCP_REGION", "us-central1")
    GCS_BUCKET: str = _env("GCS_BUCKET", "")
    EXPERIMENT_NAME: str = _env("VERTEX_EXPERIMENT_NAME", "wine-quality-training")

    # --- Logging ---
    LOG_LEVEL: str = _env("LOG_LEVEL", "INFO")


CONFIG = Config()
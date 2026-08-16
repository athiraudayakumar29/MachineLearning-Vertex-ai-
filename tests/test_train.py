"""
Placeholder test — real unit tests for run_preprocessing/train_model
land on Day 2. This just proves the package imports cleanly and pytest
is wired up correctly in CI from Day 1.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_imports():
    import config  # noqa: F401
    import preprocessing  # noqa: F401
    import train  # noqa: F401

    assert config.CONFIG.MAX_ITER == 1000
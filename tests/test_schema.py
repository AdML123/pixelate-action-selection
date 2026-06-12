import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from paper34.schema import validate_accuracy_table


def test_accuracy_table_requires_core_columns():
    df = pd.DataFrame(
        {
            "corruption": ["gaussian_noise"],
            "severity": [5],
            "config": ["config_a"],
            "accuracy": [71.2],
            "transform": ["q4"],
        }
    )

    validate_accuracy_table(df)


def test_accuracy_table_rejects_missing_columns():
    df = pd.DataFrame({"corruption": ["fog"]})

    with pytest.raises(ValueError, match="missing columns"):
        validate_accuracy_table(df)

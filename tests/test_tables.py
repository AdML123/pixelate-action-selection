import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from pixelate_router.tables import compute_config_delta, make_value_map


def test_compute_config_delta_returns_a_minus_b():
    df = pd.DataFrame(
        [
            {"corruption": "gaussian_noise", "config": "config_a", "accuracy": 70.0},
            {"corruption": "gaussian_noise", "config": "config_b", "accuracy": 65.0},
        ]
    )

    result = compute_config_delta(df)

    assert result["gaussian_noise"] == 5.0


def test_make_value_map_formats_percent_values():
    values = make_value_map({"ACC_CLEAN": 93.456})

    assert values["[ACC_CLEAN]"] == "93.46"

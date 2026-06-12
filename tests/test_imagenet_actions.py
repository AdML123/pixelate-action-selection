import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from paper34.imagenet_actions import ACTION_NAMES, jpeg_roundtrip, oracle_summary_from_correctness


def test_action_names_are_stable():
    assert ACTION_NAMES == ["identity", "dncnn", "jpeg20", "jpeg10", "config_a", "config_b", "config_a10"]


def test_jpeg_roundtrip_preserves_shape_and_float_range():
    image = np.linspace(0.0, 1.0, 32 * 32 * 3, dtype=np.float32).reshape(32, 32, 3)

    out = jpeg_roundtrip(image, quality=20)

    assert out.shape == image.shape
    assert out.dtype == np.float32
    assert 0.0 <= float(out.min()) <= float(out.max()) <= 1.0


def test_oracle_summary_reports_best_fixed_and_order_gap():
    rows = [
        {
            "label": 0,
            "identity_pred": 1,
            "dncnn_pred": 1,
            "jpeg20_pred": 1,
            "jpeg10_pred": 1,
            "config_a_pred": 0,
            "config_b_pred": 1,
            "config_a10_pred": 1,
        },
        {
            "label": 1,
            "identity_pred": 0,
            "dncnn_pred": 0,
            "jpeg20_pred": 1,
            "jpeg10_pred": 0,
            "config_a_pred": 0,
            "config_b_pred": 1,
            "config_a10_pred": 0,
        },
    ]

    summary = oracle_summary_from_correctness(rows)

    assert summary["config_a_accuracy"] == 50.0
    assert summary["config_b_accuracy"] == 50.0
    assert summary["jpeg20_accuracy"] == 50.0
    assert summary["best_fixed_accuracy"] == 50.0
    assert summary["per_image_oracle_accuracy"] == 100.0
    assert summary["oracle_minus_best_fixed"] == 50.0
    assert summary["order_gap_abs"] == 0.0

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from oracle_ceiling import build_output_paths, decide_go_nogo, load_rgb_float, summarize_rows_by_corruption


def test_decide_go_nogo_go_for_large_pixelate_gap_and_oracle():
    summaries = {
        "pixelate": {"order_gap_abs": 2.5, "oracle_minus_best_fixed": 3.5},
        "jpeg_compression": {"oracle_minus_best_fixed": 1.0},
        "contrast": {"best_fixed_accuracy": 70.0, "config_a_accuracy": 69.9},
    }

    decision = decide_go_nogo(summaries)

    assert decision["status"] == "go"


def test_decide_go_nogo_no_go_for_tiny_pixelate_gap():
    summaries = {
        "pixelate": {"order_gap_abs": 0.4, "oracle_minus_best_fixed": 0.9},
        "jpeg_compression": {"oracle_minus_best_fixed": 0.8},
        "contrast": {"best_fixed_accuracy": 70.0, "config_a_accuracy": 70.0},
    }

    decision = decide_go_nogo(summaries)

    assert decision["status"] == "no_go"


def test_decide_go_nogo_borderline_when_gap_is_real_but_oracle_small():
    summaries = {
        "pixelate": {"order_gap_abs": 2.2, "oracle_minus_best_fixed": 1.5},
        "jpeg_compression": {"oracle_minus_best_fixed": 1.2},
        "contrast": {"best_fixed_accuracy": 70.0, "config_a_accuracy": 69.9},
    }

    decision = decide_go_nogo(summaries)

    assert decision["status"] == "borderline"


def test_build_output_paths_are_stable(tmp_path):
    paths = build_output_paths(tmp_path, severity=3, limit_images=500)

    assert paths["csv"].name == "oracle_ceiling_s3_500.csv"
    assert paths["json"].name == "oracle_ceiling_s3_500.json"


def test_load_rgb_float_resizes_and_normalizes(tmp_path):
    path = tmp_path / "image.JPEG"
    Image.fromarray(np.full((16, 20, 3), 127, dtype=np.uint8)).save(path)

    image = load_rgb_float(path, image_size=8)

    assert image.shape == (8, 8, 3)
    assert image.dtype == np.float32
    assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0


def test_summarize_rows_by_corruption_groups_metrics():
    rows = [
        {
            "corruption": "pixelate",
            "label": 0,
            "identity_pred": 0,
            "dncnn_pred": 1,
            "jpeg20_pred": 1,
            "jpeg10_pred": 1,
            "config_a_pred": 0,
            "config_b_pred": 1,
            "config_a10_pred": 1,
        },
        {
            "corruption": "contrast",
            "label": 1,
            "identity_pred": 0,
            "dncnn_pred": 1,
            "jpeg20_pred": 0,
            "jpeg10_pred": 0,
            "config_a_pred": 1,
            "config_b_pred": 1,
            "config_a10_pred": 0,
        },
    ]

    summaries = summarize_rows_by_corruption(rows)

    assert summaries["pixelate"]["identity_accuracy"] == 100.0
    assert summaries["contrast"]["dncnn_accuracy"] == 100.0

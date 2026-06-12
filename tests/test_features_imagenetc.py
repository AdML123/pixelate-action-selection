import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from pixelate_router.features_imagenetc import (
    FEATURE_NAMES,
    band_energies,
    confidence_stats,
    hfer,
    residual_hfer,
)


def test_hfer_is_higher_for_checkerboard_than_constant_image():
    constant = np.ones((32, 32, 3), dtype=np.float32) * 0.5
    checker = np.indices((32, 32)).sum(axis=0) % 2
    checker = np.repeat(checker[:, :, None], 3, axis=2).astype(np.float32)
    assert hfer(checker) > hfer(constant)


def test_band_energies_sum_to_one():
    image = np.random.default_rng(0).random((32, 32, 3), dtype=np.float32)
    bands = band_energies(image)
    assert set(bands) == {"low", "mid", "high"}
    assert abs(sum(bands.values()) - 1.0) < 1e-6


def test_confidence_stats_are_finite():
    logits = np.array([3.0, 1.0, -2.0], dtype=np.float32)
    stats = confidence_stats(logits)
    assert set(stats) == {"top1_prob", "margin", "entropy"}
    assert all(np.isfinite(value) for value in stats.values())
    assert stats["margin"] > 0.0


def test_residual_hfer_is_zero_for_zero_residual():
    residual = np.zeros((32, 32, 3), dtype=np.float32)
    assert residual_hfer(residual) == 0.0


def test_feature_names_are_approved_19_column_order():
    assert len(FEATURE_NAMES) == 19
    assert FEATURE_NAMES[:5] == [
        "hfer_input",
        "m_comm",
        "band_low",
        "band_mid",
        "band_high",
    ]
    assert FEATURE_NAMES[-3:] == [
        "jpeg20_top1_prob",
        "jpeg20_margin",
        "jpeg20_entropy",
    ]

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from pixelate_router.spectral import high_frequency_energy_ratio


def test_hfer_is_zero_for_constant_image():
    image = np.ones((32, 32, 3), dtype=np.float32)

    assert high_frequency_energy_ratio(image) == 0.0


def test_hfer_is_high_for_checkerboard():
    grid = (np.indices((32, 32)).sum(axis=0) % 2) * 2 - 1
    image = np.repeat(grid[:, :, None], 3, axis=2).astype(np.float32)

    assert high_frequency_energy_ratio(image) > 0.8

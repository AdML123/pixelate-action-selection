import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from pixelate_router.transforms import jpeg_roundtrip, uniform_quantize


def test_uniform_quantize_q4_preserves_shape_and_levels():
    image = np.array([[[0.0], [8.0], [18.0], [255.0]]], dtype=np.float32)

    result = uniform_quantize(image, bits=4)

    assert result.shape == image.shape
    assert set(result.flatten()).issubset(set(np.linspace(0.0, 255.0, 16)))


def test_jpeg_roundtrip_preserves_shape():
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    result = jpeg_roundtrip(image, quality=20)

    assert result.shape == image.shape

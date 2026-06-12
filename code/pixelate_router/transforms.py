from io import BytesIO

import numpy as np
from PIL import Image


def uniform_quantize(image: np.ndarray, bits: int) -> np.ndarray:
    if bits < 1:
        raise ValueError("bits must be positive")
    levels = (2**bits) - 1
    step = 255.0 / levels
    quantized = np.round(image.astype(np.float32) / step) * step
    return np.clip(quantized, 0.0, 255.0).astype(np.float32)


def jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    if not 1 <= quality <= 95:
        raise ValueError("quality must be in [1, 95]")
    pil = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    buffer = BytesIO()
    pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"))

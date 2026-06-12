import numpy as np


def high_frequency_energy_ratio(image: np.ndarray, cutoff_fraction: float = 0.25) -> float:
    if image.ndim != 3:
        raise ValueError("image must have shape H x W x C")
    h, w, _ = image.shape
    gray = image.astype(np.float64).mean(axis=2)
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    energy = np.abs(spectrum) ** 2
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask = radius > (w * cutoff_fraction)
    total = float(energy.sum())
    if total == 0.0:
        return 0.0
    return float(energy[mask].sum() / total)

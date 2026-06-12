from __future__ import annotations

import math

import numpy as np


FEATURE_NAMES = [
    "hfer_input",
    "m_comm",
    "band_low",
    "band_mid",
    "band_high",
    "comm_band_low",
    "comm_band_mid",
    "comm_band_high",
    "hfer_jpeg20_residual",
    "hfer_dncnn_residual",
    "identity_top1_prob",
    "identity_margin",
    "identity_entropy",
    "dncnn_top1_prob",
    "dncnn_margin",
    "dncnn_entropy",
    "jpeg20_top1_prob",
    "jpeg20_margin",
    "jpeg20_entropy",
]


def hfer(image: np.ndarray, cutoff: float = 0.25) -> float:
    energy, radius = _fft_energy_and_radius(image)
    mask = radius >= float(cutoff)
    return _safe_ratio(float(energy[mask].sum()), float(energy.sum()))


def residual_hfer(residual: np.ndarray, cutoff: float = 0.25) -> float:
    energy, radius = _fft_energy_and_radius(residual)
    total = float(energy.sum())
    if total <= 0.0:
        return 0.0
    mask = radius >= float(cutoff)
    return _safe_ratio(float(energy[mask].sum()), total)


def band_energies(image: np.ndarray) -> dict[str, float]:
    energy, radius = _fft_energy_and_radius(image)
    total = float(energy.sum())
    if total <= 0.0:
        return {"low": 0.0, "mid": 0.0, "high": 0.0}
    low = float(energy[radius < 0.15].sum())
    mid = float(energy[(radius >= 0.15) & (radius < 0.30)].sum())
    high = float(energy[radius >= 0.30].sum())
    return {"low": low / total, "mid": mid / total, "high": high / total}


def commutator_magnitude(config_a: np.ndarray, config_b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(config_a, dtype=np.float32) - np.asarray(config_b, dtype=np.float32))))


def confidence_stats(logits: np.ndarray) -> dict[str, float]:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError("confidence_stats requires at least two logits")
    shifted = values - float(values.max())
    exp_values = np.exp(shifted)
    probs = exp_values / float(exp_values.sum())
    top_two = np.partition(probs, -2)[-2:]
    top_two.sort()
    top1 = float(top_two[1])
    top2 = float(top_two[0])
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)))) / math.log(float(values.size))
    return {"top1_prob": top1, "margin": top1 - top2, "entropy": entropy}


def feature_row(
    image: np.ndarray,
    dncnn_image: np.ndarray,
    jpeg20_image: np.ndarray,
    config_a_image: np.ndarray,
    config_b_image: np.ndarray,
    identity_logits: np.ndarray,
    dncnn_logits: np.ndarray,
    jpeg20_logits: np.ndarray,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    dncnn_image = np.asarray(dncnn_image, dtype=np.float32)
    jpeg20_image = np.asarray(jpeg20_image, dtype=np.float32)
    config_a_image = np.asarray(config_a_image, dtype=np.float32)
    config_b_image = np.asarray(config_b_image, dtype=np.float32)

    input_bands = band_energies(image)
    comm_residual = config_a_image - config_b_image
    comm_bands = band_energies(comm_residual)
    identity_stats = confidence_stats(identity_logits)
    dncnn_stats = confidence_stats(dncnn_logits)
    jpeg20_stats = confidence_stats(jpeg20_logits)

    values = [
        hfer(image),
        commutator_magnitude(config_a_image, config_b_image),
        input_bands["low"],
        input_bands["mid"],
        input_bands["high"],
        comm_bands["low"],
        comm_bands["mid"],
        comm_bands["high"],
        residual_hfer(image - jpeg20_image),
        residual_hfer(image - dncnn_image),
        identity_stats["top1_prob"],
        identity_stats["margin"],
        identity_stats["entropy"],
        dncnn_stats["top1_prob"],
        dncnn_stats["margin"],
        dncnn_stats["entropy"],
        jpeg20_stats["top1_prob"],
        jpeg20_stats["margin"],
        jpeg20_stats["entropy"],
    ]
    return np.asarray(values, dtype=np.float32)


def _fft_energy_and_radius(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must have shape HxWx3")
    spectrum = np.fft.fft2(array, axes=(0, 1))
    energy = np.mean(np.abs(spectrum) ** 2, axis=2)
    energy[0, 0] = 0.0
    height, width = energy.shape
    fy = np.fft.fftfreq(height)
    fx = np.fft.fftfreq(width)
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    return energy.astype(np.float64), radius.astype(np.float64)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)

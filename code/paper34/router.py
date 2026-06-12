from __future__ import annotations

import numpy as np


def select_with_threshold(scores: np.ndarray, default_index: int, tau: float) -> int:
    scores = np.asarray(scores, dtype=np.float32)
    best = int(scores.argmax())
    if float(scores[best] - scores[default_index]) >= float(tau):
        return best
    return int(default_index)


def paired_lcb(router_correct: np.ndarray, default_correct: np.ndarray, z: float = 1.96) -> float:
    paired = np.asarray(router_correct, dtype=np.float64) - np.asarray(default_correct, dtype=np.float64)
    if paired.ndim != 1:
        raise ValueError("paired correctness arrays must be one-dimensional")
    if paired.size == 0:
        raise ValueError("paired correctness arrays must not be empty")
    mean = float(paired.mean())
    if paired.size == 1:
        return mean
    stderr = float(paired.std(ddof=1) / np.sqrt(paired.size))
    return mean - float(z) * stderr


def hfer_rule(hfer_value: float, theta: float, default_action: str) -> str:
    return "config_a10" if float(hfer_value) > float(theta) else str(default_action)


def two_threshold_rule(hfer_value: float, theta_low: float, theta_high: float) -> str:
    value = float(hfer_value)
    if value < float(theta_low):
        return "dncnn"
    if value > float(theta_high):
        return "config_a"
    return "dncnn"

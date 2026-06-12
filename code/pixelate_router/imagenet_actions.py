from __future__ import annotations

import io
from collections import Counter

import numpy as np
from PIL import Image


ACTION_NAMES = ["identity", "dncnn", "jpeg20", "jpeg10", "config_a", "config_b", "config_a10"]


def jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    if quality < 1 or quality > 95:
        raise ValueError("quality must be in [1, 95]")
    array = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must have shape HxWx3")
    uint8 = np.rint(array * 255.0).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(uint8).save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    decoded = Image.open(buffer).convert("RGB")
    return np.clip(np.asarray(decoded, dtype=np.float32) / 255.0, 0.0, 1.0)


def oracle_summary_from_correctness(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        raise ValueError("oracle summary requires at least one row")

    accuracies = {}
    for action in ACTION_NAMES:
        correct = [int(row[f"{action}_pred"]) == int(row["label"]) for row in rows]
        accuracies[f"{action}_accuracy"] = 100.0 * sum(correct) / n

    best_fixed = max(accuracies.values())
    oracle_correct = []
    winners = []
    for row in rows:
        correct_actions = [action for action in ACTION_NAMES if int(row[f"{action}_pred"]) == int(row["label"])]
        oracle_correct.append(bool(correct_actions))
        winners.append(correct_actions[0] if correct_actions else "none")

    per_image_oracle = 100.0 * sum(oracle_correct) / n
    order_gap_signed = accuracies["config_a_accuracy"] - accuracies["config_b_accuracy"]

    result = dict(accuracies)
    result.update(
        {
            "n": n,
            "best_fixed_accuracy": best_fixed,
            "per_image_oracle_accuracy": per_image_oracle,
            "oracle_minus_best_fixed": per_image_oracle - best_fixed,
            "order_gap_signed": order_gap_signed,
            "order_gap_abs": abs(order_gap_signed),
            "oracle_winner_counts": dict(Counter(winners)),
        }
    )
    return result

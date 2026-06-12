import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from evaluate_corruption_detector import (  # noqa: E402
    ACTIONS,
    CORRUPTIONS,
    _best_action_by_corruption,
    _evaluate_detector_policy,
)


def test_best_action_by_corruption_uses_validation_rewards():
    data = {
        "y": np.asarray(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
            ],
            dtype=np.float32,
        ),
        "corruption": np.asarray(["contrast", "contrast", "pixelate", "pixelate"]),
    }

    mapping = _best_action_by_corruption(data)

    assert mapping["contrast"] == ACTIONS.index("identity")
    assert mapping["pixelate"] == ACTIONS.index("config_a")


def test_detector_policy_uses_predicted_corruption_not_true_labels():
    mapping = {corr: ACTIONS.index("dncnn") for corr in CORRUPTIONS}
    mapping["pixelate"] = ACTIONS.index("config_a")
    data = {
        "y": np.zeros((2, len(ACTIONS)), dtype=np.float32),
        "corruption": np.asarray(["contrast", "pixelate"]),
    }
    data["y"][0, ACTIONS.index("config_a")] = 1.0
    data["y"][1, ACTIONS.index("jpeg20")] = 1.0
    predicted = np.asarray(["pixelate", "contrast"])

    report = _evaluate_detector_policy(data, predicted, mapping)

    assert report["accuracy"] == 50.0
    assert report["detector_accuracy"] == 0.0

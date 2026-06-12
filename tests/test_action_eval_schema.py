import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from full_action_eval import CellAccumulator, indices_for_split, split_indices


def test_split_indices_are_fixed_and_non_overlapping():
    splits = split_indices()

    assert splits["search"] == (0, 1999)
    assert splits["validation"] == (2000, 6999)
    assert splits["test"] == (7000, 16999)


def test_indices_for_split_can_limit_each_split_for_smoke_runs():
    assert indices_for_split("validation", limit=3) == [2000, 2001, 2002]


def test_cell_accumulator_reports_oracle_headroom_and_action_entropy():
    acc = CellAccumulator()
    acc.update(
        0,
        {
            "identity": 0,
            "dncnn": 1,
            "jpeg20": 1,
            "jpeg10": 1,
            "config_a": 1,
            "config_b": 1,
            "config_a10": 1,
        },
    )
    acc.update(
        1,
        {
            "identity": 0,
            "dncnn": 1,
            "jpeg20": 1,
            "jpeg10": 0,
            "config_a": 0,
            "config_b": 0,
            "config_a10": 0,
        },
    )

    summary = acc.summary()

    assert summary["n"] == 2
    assert summary["identity_accuracy"] == 50.0
    assert summary["dncnn_accuracy"] == 50.0
    assert summary["jpeg20_accuracy"] == 50.0
    assert summary["best_fixed_accuracy"] == 50.0
    assert summary["per_image_oracle_accuracy"] == 100.0
    assert summary["oracle_minus_best_fixed"] == 50.0
    assert summary["oracle_winner_counts"] == {"identity": 1, "dncnn": 1}
    assert math.isclose(summary["oracle_action_entropy_bits"], 1.0)

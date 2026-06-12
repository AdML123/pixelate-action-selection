import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from paper34.experiments import choose_adaptive_config, summarize_accuracy


def test_adaptive_gate_routes_high_hfer_to_config_a():
    assert choose_adaptive_config(hfer=0.62, threshold=0.50) == "config_a"


def test_adaptive_gate_routes_low_hfer_to_config_b():
    assert choose_adaptive_config(hfer=0.12, threshold=0.50) == "config_b"


def test_summarize_accuracy_counts_correct_predictions():
    rows = [
        {"corruption": "fog", "severity": 1, "config": "config_b", "correct": True},
        {"corruption": "fog", "severity": 1, "config": "config_b", "correct": False},
    ]

    summary = summarize_accuracy(rows)

    assert summary[0]["accuracy"] == 50.0

import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from make_imagenetc_tables import (
    format_accuracy,
    format_gain,
    table_ablation,
    table_main,
    table_action_distribution,
    table_oracle,
    table_pixelate_primary,
    table_pixelate_severity_from_rows,
)


def test_format_accuracy_uses_one_decimal_place():
    assert format_accuracy(58.234) == "58.2"


def test_format_gain_keeps_sign():
    assert format_gain(1.384) == "+1.4"
    assert format_gain(-0.156) == "-0.2"


def test_format_gain_suppresses_negative_zero():
    assert format_gain(-0.003) == "0.0"


def test_table_oracle_uses_confirmation_json():
    oracle = {"summaries": {}}
    action = {"summaries": {"test": {}}}
    for corr in ["contrast", "elastic_transform", "pixelate", "jpeg_compression"]:
        oracle["summaries"][corr] = {
            "dncnn_accuracy": 69.8,
            "best_fixed_accuracy": 70.5,
            "per_image_oracle_accuracy": 72.0,
            "oracle_winner_counts": {"identity": 10, "dncnn": 5, "none": 100},
        }
        action["summaries"]["test"][corr] = {
            "3": {
                "dncnn_accuracy": 1.0,
                "best_fixed_accuracy": 2.0,
                "per_image_oracle_accuracy": 3.0,
                "oracle_action_entropy_bits": 9.0,
            }
        }

    rendered = table_oracle(oracle, action)

    assert "Contrast & 69.8 & 70.5 & 72.0 & +1.5" in rendered
    assert "Contrast & 1.0 & 2.0 & 3.0" not in rendered


def _assert_no_empty_cells(rendered: str) -> None:
    assert "--" not in rendered
    assert " &  & " not in rendered


def test_table_pixelate_primary_makes_pixelate_headline():
    action = {
        "summaries": {
            "test": {
                "pixelate": {
                    "1": {
                        "n": 10000,
                        "dncnn_accuracy": 43.0,
                        "best_fixed_accuracy": 50.0,
                        "per_image_oracle_accuracy": 59.0,
                    }
                }
            }
        }
    }
    ablation = {
        "policies": {
            "logistic_all": {
                "test": {
                    "per_corruption": {
                        "pixelate": {
                            "accuracy": 48.762,
                            "default_accuracy": 43.578,
                            "gain": 5.184,
                            "paired_lcb": 4.923,
                        }
                    }
                }
            }
        }
    }
    rendered = table_pixelate_primary(action, ablation)

    assert "Pixelate test" in rendered
    assert "43.0" in rendered
    assert "48.8" in rendered
    assert "+5.2" in rendered
    assert "+4.9" in rendered
    assert "LOCO" not in rendered
    _assert_no_empty_cells(rendered)


def test_table_action_distribution_reports_pixelate_switching():
    ablation = {
        "policies": {
            "logistic_all": {
                "test": {
                    "per_corruption": {
                        "contrast": {
                            "gain": -0.158,
                            "action_distribution": {
                                "dncnn": {"share": 0.989},
                                "jpeg20": {"share": 0.002},
                                "config_a": {"share": 0.004},
                                "config_b": {"share": 0.003},
                            },
                        },
                        "elastic_transform": {
                            "gain": 0.506,
                            "action_distribution": {
                                "dncnn": {"share": 0.752},
                                "jpeg20": {"share": 0.061},
                                "config_a": {"share": 0.091},
                                "config_b": {"share": 0.078},
                            },
                        },
                        "pixelate": {
                            "gain": 5.184,
                            "action_distribution": {
                                "dncnn": {"share": 0.56662},
                                "jpeg20": {"share": 0.13066},
                                "config_a": {"share": 0.13992},
                                "config_b": {"share": 0.12384},
                                "identity": {"share": 0.00028},
                                "jpeg10": {"share": 0.01694},
                                "config_a10": {"share": 0.02174},
                            },
                        },
                        "jpeg_compression": {
                            "gain": 0.008,
                            "action_distribution": {
                                "dncnn": {"share": 0.925},
                                "jpeg20": {"share": 0.020},
                                "config_a": {"share": 0.025},
                                "config_b": {"share": 0.018},
                            },
                        },
                    }
                }
            }
        }
    }

    rendered = table_action_distribution(ablation)

    assert "Pixelate" in rendered
    assert "56.7" in rendered
    assert "13.1" in rendered
    assert "14.0" in rendered
    assert "12.4" in rendered
    assert "3.9" in rendered
    assert "+5.2" in rendered


def test_main_text_tables_have_no_empty_cells():
    action = {
        "summaries": {
            "test": {
                "contrast": {
                    "1": {
                        "n": 10,
                        "dncnn_accuracy": 60.0,
                        "best_fixed_accuracy": 61.0,
                        "per_image_oracle_accuracy": 63.0,
                    }
                },
                "elastic_transform": {
                    "1": {
                        "n": 10,
                        "dncnn_accuracy": 48.0,
                        "best_fixed_accuracy": 49.0,
                        "per_image_oracle_accuracy": 55.0,
                    }
                },
                "pixelate": {
                    "1": {
                        "n": 10,
                        "dncnn_accuracy": 43.578,
                        "best_fixed_accuracy": 50.592,
                        "per_image_oracle_accuracy": 59.416,
                    }
                },
                "jpeg_compression": {
                    "1": {
                        "n": 10,
                        "dncnn_accuracy": 62.0,
                        "best_fixed_accuracy": 62.1,
                        "per_image_oracle_accuracy": 64.0,
                    }
                },
            }
        }
    }
    per_corruption = {
        "contrast": {
            "accuracy": 61.508,
            "gain": -0.158,
            "paired_lcb": -0.23,
            "action_distribution": {"dncnn": {"share": 0.99}},
        },
        "elastic_transform": {
            "accuracy": 49.29,
            "gain": 0.506,
            "paired_lcb": 0.41,
            "action_distribution": {"dncnn": {"share": 0.75}},
        },
        "pixelate": {
            "accuracy": 48.762,
            "gain": 5.184,
            "paired_lcb": 4.924,
            "action_distribution": {"dncnn": {"share": 0.57}},
        },
        "jpeg_compression": {
            "accuracy": 61.974,
            "gain": 0.008,
            "paired_lcb": -0.02,
            "action_distribution": {"dncnn": {"share": 0.92}},
        },
    }
    ablation = {
        "policies": {
            "default_dncnn": {
                "validation": {"gain": 0.0},
                "test": {
                    "gain": 0.0,
                    "per_corruption": {
                        "pixelate": {"gain": 0.0},
                        "contrast": {"gain": 0.0},
                    },
                },
            },
            "best_fixed": {
                "validation": {"gain": 0.0},
                "test": {
                    "gain": 0.0,
                    "per_corruption": {
                        "pixelate": {"gain": 0.0},
                        "contrast": {"gain": 0.0},
                    },
                },
            },
            "hfer_rule": {
                "validation": {"gain": 0.0},
                "test": {
                    "gain": -0.007,
                    "per_corruption": {
                        "pixelate": {"gain": 0.0},
                        "contrast": {"gain": -0.022},
                    },
                },
            },
            "two_threshold": {
                "validation": {"gain": 0.0},
                "test": {
                    "gain": -0.003,
                    "per_corruption": {
                        "pixelate": {"gain": 0.002},
                        "contrast": {"gain": -0.014},
                    },
                },
            },
            "logistic_all": {
                "validation": {"gain": 1.301},
                "test": {
                    "gain": 1.385,
                    "accuracy": 55.384,
                    "action_distribution": {"dncnn": {"share": 0.8}},
                    "per_corruption": per_corruption,
                },
            },
            "logistic_spectral": {
                "validation": {"gain": 0.463},
                "test": {
                    "gain": 0.605,
                    "per_corruption": {
                        "pixelate": {"gain": 3.024},
                        "contrast": {"gain": -0.066},
                    },
                },
            },
            "logistic_confidence": {
                "validation": {"gain": 1.001},
                "test": {
                    "gain": 1.017,
                    "per_corruption": {
                        "pixelate": {"gain": 4.690},
                        "contrast": {"gain": -1.150},
                    },
                },
            },
            "logistic_no_commutator": {
                "validation": {"gain": 1.100},
                "test": {
                    "gain": 1.200,
                    "per_corruption": {
                        "pixelate": {"gain": 4.900},
                        "contrast": {"gain": -0.300},
                    },
                },
            },
            "detector_per_corruption": {
                "validation": {"gain": 0.25},
                "test": {
                    "gain": 0.40,
                    "per_corruption": {
                        "pixelate": {"gain": 2.20},
                        "contrast": {"gain": -0.10},
                    },
                },
            },
            "true_label_per_corruption_oracle": {
                "validation": {"gain": 0.25},
                "test": {
                    "gain": 0.50,
                    "per_corruption": {
                        "pixelate": {"gain": 2.50},
                        "contrast": {"gain": 0.0},
                    },
                },
            },
        }
    }

    for rendered in [
        table_pixelate_primary(action, ablation),
        table_main(action, ablation),
        table_ablation(ablation),
    ]:
        _assert_no_empty_cells(rendered)


def test_ablation_table_surfaces_simple_baselines():
    base_eval = {
        "gain": 0.0,
        "per_corruption": {
            "pixelate": {"gain": 0.0},
            "contrast": {"gain": 0.0},
        },
    }
    ablation = {
        "policies": {
            "default_dncnn": {"validation": {"gain": 0.0}, "test": base_eval},
            "best_fixed": {"validation": {"gain": 0.0}, "test": base_eval},
            "hfer_rule": {"validation": {"gain": 0.0}, "test": base_eval},
            "two_threshold": {"validation": {"gain": 0.0}, "test": base_eval},
            "logistic_confidence": {"validation": {"gain": 1.001}, "test": base_eval},
            "logistic_spectral": {"validation": {"gain": 0.463}, "test": base_eval},
            "logistic_no_commutator": {"validation": {"gain": 1.100}, "test": base_eval},
            "detector_per_corruption": {"validation": {"gain": 0.250}, "test": base_eval},
            "logistic_all": {"validation": {"gain": 1.301}, "test": base_eval},
            "true_label_per_corruption_oracle": {"validation": {"gain": 0.0}, "test": base_eval},
        }
    }

    rendered = table_ablation(ablation)

    for label in [
        "DnCNN only",
        "Best fixed",
        "HFER rule",
        "Two-thresh.",
        "Router & confidence",
        "Router & spectral",
        "Router & no comm.",
        "Detector & per-corr.",
        "Router & all",
        "Oracle & true corr.",
    ]:
        assert label in rendered
    _assert_no_empty_cells(rendered)


def test_pixelate_severity_table_uses_all_five_severities():
    rows = [
        {
            "severity": str(severity),
            "dncnn": str(40 + severity),
            "jpeg20": str(41 + severity),
            "config_a": str(42 + severity),
            "router": str(43 + severity),
            "gain": "1.2",
            "paired_lcb": "0.8",
        }
        for severity in [1, 2, 3, 4, 5]
    ]

    rendered = table_pixelate_severity_from_rows(rows)

    for severity in [1, 2, 3, 4, 5]:
        assert f"{severity} &" in rendered
    assert "Config-A" in rendered
    assert "Gain" in rendered
    assert "LCB" in rendered
    _assert_no_empty_cells(rendered)


def test_default_table_generator_outputs_only_manuscript_tables(tmp_path):
    outdir = tmp_path / "tables"

    completed = subprocess.run(
        [
            sys.executable,
            "code/make_tables.py",
            "--outdir",
            str(outdir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert {path.name for path in outdir.glob("*.tex")} == {
        "table_pixelate_primary.tex",
        "table_main.tex",
        "table_ablation.tex",
        "table_pixelate_severity.tex",
    }

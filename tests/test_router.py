import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from paper34.router import hfer_rule, paired_lcb, select_with_threshold, two_threshold_rule


def test_select_with_threshold_keeps_default_when_gain_is_small():
    scores = np.array([0.2, 0.8, 0.81], dtype=np.float32)
    assert select_with_threshold(scores, default_index=1, tau=0.05) == 1


def test_select_with_threshold_switches_when_gain_is_large():
    scores = np.array([0.2, 0.8, 0.91], dtype=np.float32)
    assert select_with_threshold(scores, default_index=1, tau=0.05) == 2


def test_paired_lcb_is_positive_when_switches_win():
    router = np.tile(np.array([1, 1, 0, 1, 1, 0], dtype=np.int64), 40)
    default = np.tile(np.array([0, 0, 0, 1, 1, 0], dtype=np.int64), 40)
    assert paired_lcb(router, default) > 0.0


def test_hfer_rule_switches_only_above_threshold():
    assert hfer_rule(0.81, theta=0.8, default_action="dncnn") == "config_a10"
    assert hfer_rule(0.80, theta=0.8, default_action="dncnn") == "dncnn"


def test_two_threshold_rule_uses_dncnn_outside_high_hfer():
    assert two_threshold_rule(0.1, theta_low=0.2, theta_high=0.8) == "dncnn"
    assert two_threshold_rule(0.5, theta_low=0.2, theta_high=0.8) == "dncnn"
    assert two_threshold_rule(0.9, theta_low=0.2, theta_high=0.8) == "config_a"


def test_no_commutator_feature_set_excludes_commutator_features():
    import train_router

    no_comm = train_router.FEATURE_SETS["no_commutator"]
    assert "m_comm" not in no_comm
    assert "comm_band_low" not in no_comm
    assert "comm_band_mid" not in no_comm
    assert "comm_band_high" not in no_comm
    assert "hfer_input" in no_comm
    assert "hfer_jpeg20_residual" in no_comm
    assert "identity_top1_prob" in no_comm
    assert "dncnn_top1_prob" in no_comm
    assert "jpeg20_top1_prob" in no_comm

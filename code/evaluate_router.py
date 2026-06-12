from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pixelate_router.features_imagenetc import FEATURE_NAMES
from pixelate_router.router import hfer_rule, paired_lcb, select_with_threshold, two_threshold_rule
from train_router import ACTIONS, CORRUPTIONS, DEFAULT_ACTION, DEFAULT_INDEX, FEATURE_SETS, SEVERITIES


RULE_POLICIES = ["hfer_rule", "two_threshold"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate router ablations and rule baselines.")
    parser.add_argument("--features-root", required=True)
    parser.add_argument("--action-csv", required=True)
    parser.add_argument("--router-dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--test-features-root", default=None)
    parser.add_argument("--device", default="auto")
    return parser


def run(args: argparse.Namespace) -> dict:
    import torch

    start_perf = time.perf_counter()
    start_time = datetime.now(timezone.utc)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device, torch)
    action_rows = _load_action_rows(Path(args.action_csv))
    val_root = Path(args.features_root)
    test_root = Path(args.test_features_root) if args.test_features_root else val_root

    validation = _load_split(val_root, action_rows, "validation", FEATURE_NAMES)
    test = _load_split(test_root, action_rows, "test", FEATURE_NAMES) if _has_split_features(test_root, "test") else None
    default_validation = _evaluate_fixed(validation, DEFAULT_INDEX)
    default_test = _evaluate_fixed(test, DEFAULT_INDEX) if test is not None else None

    policies = {}
    for feature_set in FEATURE_SETS:
        policies[f"logistic_{feature_set}"] = _evaluate_logistic_policy(
            Path(args.router_dir), feature_set, validation, test, torch, device
        )

    hfer_policy = _evaluate_hfer_rule(validation, test)
    policies["hfer_rule"] = hfer_policy
    policies["two_threshold"] = _evaluate_two_threshold_rule(validation, test)
    policies["best_fixed"] = _evaluate_best_fixed(validation, test)
    policies["default_dncnn"] = {
        "policy": "default_dncnn",
        "validation": default_validation,
        "test": default_test,
        "selected": {"action": DEFAULT_ACTION},
    }

    payload = {
        "schema_version": 1,
        "command": " ".join(sys.argv),
        "start_time_utc": start_time.isoformat(),
        "end_time_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - start_perf, 3),
        "environment": _environment_metadata(torch, device),
        "default_action": DEFAULT_ACTION,
        "feature_names": FEATURE_NAMES,
        "inputs": {
            "features_root": str(val_root.resolve()),
            "test_features_root": str(test_root.resolve()) if test is not None else None,
            "action_csv": str(Path(args.action_csv).resolve()),
            "router_dir": str(Path(args.router_dir).resolve()),
        },
        "policies": policies,
    }
    path = outdir / "ablation_report.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"ablation_report": str(path)}, indent=2), flush=True)
    return payload


def _evaluate_logistic_policy(router_dir: Path, feature_set: str, validation: dict, test: dict | None, torch, device) -> dict:
    checkpoint_path = router_dir / f"router_{feature_set}.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    feature_names = checkpoint["feature_names"]
    feature_indices = [FEATURE_NAMES.index(name) for name in feature_names]
    scaler = checkpoint["scaler"]
    tau = float(checkpoint["report"]["tau_selected"])
    state = checkpoint["model_state_dict"]
    model = torch.nn.Linear(len(feature_names), len(ACTIONS)).to(device)
    model.load_state_dict(state)
    model.eval()

    validation_eval = _evaluate_model_dataset(model, validation, feature_indices, scaler, tau, torch, device)
    test_eval = _evaluate_model_dataset(model, test, feature_indices, scaler, tau, torch, device) if test is not None else None
    return {
        "policy": f"logistic_{feature_set}",
        "feature_set": feature_set,
        "selected": {"tau": tau, "checkpoint": str(checkpoint_path)},
        "validation": validation_eval,
        "test": test_eval,
    }


def _evaluate_model_dataset(model, data: dict, feature_indices: list[int], scaler: dict, tau: float, torch, device) -> dict:
    x = data["x"][:, feature_indices].astype(np.float32)
    x = ((x - scaler["mean"]) / scaler["std"]).astype(np.float32)
    with torch.inference_mode():
        scores = torch.sigmoid(model(torch.from_numpy(x).float().to(device))).detach().cpu().numpy()
    selected = np.array([select_with_threshold(row, DEFAULT_INDEX, tau) for row in scores], dtype=np.int64)
    return _evaluate_selected(data, selected)


def _evaluate_hfer_rule(validation: dict, test: dict | None) -> dict:
    hfer_index = FEATURE_NAMES.index("hfer_input")
    best = None
    for theta in np.linspace(0.0, 1.0, 101):
        selected = np.array(
            [ACTIONS.index(hfer_rule(value, theta, DEFAULT_ACTION)) for value in validation["x"][:, hfer_index]],
            dtype=np.int64,
        )
        evaluation = _evaluate_selected(validation, selected)
        key = (evaluation["paired_lcb"], evaluation["gain"], -theta)
        if best is None or key > best[0]:
            best = (key, float(theta), evaluation)
    theta = best[1]
    test_eval = None
    if test is not None:
        selected_test = np.array(
            [ACTIONS.index(hfer_rule(value, theta, DEFAULT_ACTION)) for value in test["x"][:, hfer_index]],
            dtype=np.int64,
        )
        test_eval = _evaluate_selected(test, selected_test)
    return {
        "policy": "hfer_rule",
        "selected": {"theta": theta, "fallback": DEFAULT_ACTION, "switch_action": "config_a10"},
        "validation": best[2],
        "test": test_eval,
    }


def _evaluate_two_threshold_rule(validation: dict, test: dict | None) -> dict:
    hfer_index = FEATURE_NAMES.index("hfer_input")
    grid = np.linspace(0.0, 1.0, 51)
    best = None
    for theta_low in grid:
        for theta_high in grid:
            if theta_low >= theta_high:
                continue
            selected = np.array(
                [
                    ACTIONS.index(two_threshold_rule(value, theta_low, theta_high))
                    for value in validation["x"][:, hfer_index]
                ],
                dtype=np.int64,
            )
            evaluation = _evaluate_selected(validation, selected)
            key = (evaluation["paired_lcb"], evaluation["gain"], -theta_high, -theta_low)
            if best is None or key > best[0]:
                best = (key, float(theta_low), float(theta_high), evaluation)
    theta_low = best[1]
    theta_high = best[2]
    test_eval = None
    if test is not None:
        selected_test = np.array(
            [
                ACTIONS.index(two_threshold_rule(value, theta_low, theta_high))
                for value in test["x"][:, hfer_index]
            ],
            dtype=np.int64,
        )
        test_eval = _evaluate_selected(test, selected_test)
    return {
        "policy": "two_threshold",
        "selected": {"theta_low": theta_low, "theta_high": theta_high},
        "validation": best[3],
        "test": test_eval,
    }


def _evaluate_best_fixed(validation: dict, test: dict | None) -> dict:
    validation_scores = validation["y"].mean(axis=0)
    best_index = int(validation_scores.argmax())
    validation_eval = _evaluate_fixed(validation, best_index)
    test_eval = _evaluate_fixed(test, best_index) if test is not None else None
    return {
        "policy": "best_fixed",
        "selected": {"action": ACTIONS[best_index]},
        "validation": validation_eval,
        "test": test_eval,
    }


def _evaluate_fixed(data: dict, action_index: int) -> dict:
    selected = np.full(data["y"].shape[0], action_index, dtype=np.int64)
    return _evaluate_selected(data, selected)


def _evaluate_selected(data: dict, selected: np.ndarray) -> dict:
    row_indices = np.arange(data["y"].shape[0])
    correct = data["y"][row_indices, selected].astype(np.int64)
    default_correct = data["y"][:, DEFAULT_INDEX].astype(np.int64)
    return {
        "n": int(correct.size),
        "accuracy": float(correct.mean() * 100.0),
        "default_accuracy": float(default_correct.mean() * 100.0),
        "gain": float((correct.mean() - default_correct.mean()) * 100.0),
        "paired_lcb": float(paired_lcb(correct, default_correct) * 100.0),
        "action_distribution": _action_distribution(selected),
        "per_corruption": _per_corruption_report(data, selected, correct, default_correct),
    }


def _per_corruption_report(data: dict, selected: np.ndarray, correct: np.ndarray, default_correct: np.ndarray) -> dict:
    report = {}
    for corr in CORRUPTIONS:
        mask = data["corruption"] == corr
        report[corr] = {
            "n": int(mask.sum()),
            "accuracy": float(correct[mask].mean() * 100.0),
            "default_accuracy": float(default_correct[mask].mean() * 100.0),
            "gain": float((correct[mask].mean() - default_correct[mask].mean()) * 100.0),
            "paired_lcb": float(paired_lcb(correct[mask], default_correct[mask]) * 100.0),
            "action_distribution": _action_distribution(selected[mask]),
        }
    return report


def _action_distribution(selected: np.ndarray) -> dict:
    counts = Counter(int(item) for item in selected)
    total = int(selected.size)
    return {
        ACTIONS[index]: {
            "count": int(counts.get(index, 0)),
            "share": float(counts.get(index, 0) / total) if total else 0.0,
        }
        for index in range(len(ACTIONS))
    }


def _load_split(features_root: Path, action_rows: dict, split: str, feature_names: list[str]) -> dict:
    indices = [FEATURE_NAMES.index(name) for name in feature_names]
    x_parts = []
    y_parts = []
    corruption = []
    severity_values = []
    for corr in CORRUPTIONS:
        for severity in SEVERITIES:
            path = features_root / f"features_{corr}_{severity}_{split}.npy"
            if not path.exists():
                raise FileNotFoundError(path)
            features = np.load(path).astype(np.float32)[:, indices]
            rows = action_rows[(split, corr, severity)]
            if features.shape[0] != len(rows):
                raise RuntimeError(f"{path} has {features.shape[0]} rows but CSV has {len(rows)}")
            rewards = np.asarray(
                [[int(row[f"{action}_correct"]) for action in ACTIONS] for row in rows],
                dtype=np.float32,
            )
            x_parts.append(features)
            y_parts.append(rewards)
            corruption.extend([corr] * features.shape[0])
            severity_values.extend([severity] * features.shape[0])
    return {
        "x": np.vstack(x_parts),
        "y": np.vstack(y_parts),
        "corruption": np.asarray(corruption),
        "severity": np.asarray(severity_values, dtype=np.int64),
    }


def _has_split_features(features_root: Path, split: str) -> bool:
    return all(
        (features_root / f"features_{corr}_{severity}_{split}.npy").exists()
        for corr in CORRUPTIONS
        for severity in SEVERITIES
    )


def _load_action_rows(action_csv: Path) -> dict:
    rows = defaultdict(list)
    with action_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[(row["split"], row["corruption"], int(row["severity"]))].append(row)
    return rows


def _resolve_device(device: str, torch):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _environment_metadata(torch, device) -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

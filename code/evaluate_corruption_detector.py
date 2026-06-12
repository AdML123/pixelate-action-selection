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
from pixelate_router.io import open_text
from pixelate_router.router import paired_lcb
from train_router import ACTIONS, CORRUPTIONS, DEFAULT_INDEX, SEVERITIES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate corruption-detector action baseline.")
    parser.add_argument("--features-root", required=True)
    parser.add_argument("--test-features-root", required=True)
    parser.add_argument("--action-csv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--feature-set", default="all", choices=["all", "spectral", "confidence"])
    return parser


def run(args: argparse.Namespace) -> dict:
    start_perf = time.perf_counter()
    start_time = datetime.now(timezone.utc)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    action_rows = _load_action_rows(Path(args.action_csv))
    feature_names = _feature_names(args.feature_set)
    train = _load_split(Path(args.features_root), action_rows, "search", feature_names)
    validation = _load_split(Path(args.features_root), action_rows, "validation", feature_names)
    test = _load_split(Path(args.test_features_root), action_rows, "test", feature_names)

    detector = _NearestCentroidDetector().fit(train["x"], train["corruption"])
    predicted_validation = detector.predict(validation["x"])
    predicted_test = detector.predict(test["x"])
    mapping = _best_action_by_corruption(validation)
    validation_detector_policy = _evaluate_detector_policy(validation, predicted_validation, mapping)
    detector_policy = _evaluate_detector_policy(test, predicted_test, mapping)
    true_label_oracle = _evaluate_detector_policy(test, test["corruption"], mapping)
    payload = {
        "schema_version": 1,
        "command": " ".join(sys.argv),
        "start_time_utc": start_time.isoformat(),
        "end_time_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - start_perf, 3),
        "environment": _environment_metadata(),
        "feature_set": args.feature_set,
        "feature_names": feature_names,
        "selected": {
            "corruption_to_action": {corr: ACTIONS[index] for corr, index in mapping.items()},
        },
        "validation": validation_detector_policy,
        "test": detector_policy,
        "diagnostic_true_label_oracle": true_label_oracle,
    }
    path = outdir / "detector_baseline_report.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"detector_baseline_report": str(path)}, indent=2), flush=True)
    return payload


class _NearestCentroidDetector:
    def fit(self, x: np.ndarray, labels: np.ndarray):
        self.labels_ = np.asarray(CORRUPTIONS)
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ < 1e-6] = 1.0
        z = (x - self.mean_) / self.std_
        self.centroids_ = np.vstack([z[labels == corr].mean(axis=0) for corr in self.labels_])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean_) / self.std_
        distances = ((z[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        return self.labels_[distances.argmin(axis=1)]


def _best_action_by_corruption(data: dict) -> dict[str, int]:
    mapping = {}
    for corr in CORRUPTIONS:
        mask = data["corruption"] == corr
        if int(mask.sum()) == 0:
            continue
        scores = data["y"][mask].mean(axis=0)
        mapping[corr] = int(scores.argmax())
    return mapping


def _evaluate_detector_policy(data: dict, predicted_corruption: np.ndarray, mapping: dict[str, int]) -> dict:
    selected = np.asarray([mapping[str(corr)] for corr in predicted_corruption], dtype=np.int64)
    row_indices = np.arange(data["y"].shape[0])
    correct = data["y"][row_indices, selected].astype(np.int64)
    default_correct = data["y"][:, DEFAULT_INDEX].astype(np.int64)
    detector_correct = predicted_corruption == data["corruption"]
    return {
        "n": int(correct.size),
        "detector_accuracy": float(detector_correct.mean() * 100.0),
        "accuracy": float(correct.mean() * 100.0),
        "default_accuracy": float(default_correct.mean() * 100.0),
        "gain": float((correct.mean() - default_correct.mean()) * 100.0),
        "paired_lcb": float(paired_lcb(correct, default_correct) * 100.0),
        "action_distribution": _action_distribution(selected),
        "per_corruption": _per_corruption_report(data, correct, default_correct, selected, detector_correct),
    }


def _per_corruption_report(
    data: dict,
    correct: np.ndarray,
    default_correct: np.ndarray,
    selected: np.ndarray,
    detector_correct: np.ndarray,
) -> dict:
    report = {}
    for corr in CORRUPTIONS:
        mask = data["corruption"] == corr
        if int(mask.sum()) == 0:
            continue
        report[corr] = {
            "n": int(mask.sum()),
            "detector_accuracy": float(detector_correct[mask].mean() * 100.0),
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


def _feature_names(feature_set: str) -> list[str]:
    if feature_set == "all":
        return FEATURE_NAMES
    if feature_set == "spectral":
        return [
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
        ]
    return [
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


def _load_split(features_root: Path, action_rows: dict, split: str, feature_names: list[str]) -> dict:
    indices = [FEATURE_NAMES.index(name) for name in feature_names]
    x_parts = []
    y_parts = []
    corruption = []
    severity_values = []
    for corr in CORRUPTIONS:
        for severity in SEVERITIES:
            path = features_root / f"features_{corr}_{severity}_{split}.npy"
            features = np.load(path).astype(np.float32)[:, indices]
            rows = action_rows[(split, corr, severity)]
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


def _load_action_rows(action_csv: Path) -> dict:
    rows = defaultdict(list)
    with open_text(action_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[(row["split"], row["corruption"], int(row["severity"]))].append(row)
    return rows


def _environment_metadata() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform()}


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

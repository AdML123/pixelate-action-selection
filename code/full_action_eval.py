from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from oracle_ceiling import load_dncnn, load_resnet50, load_rgb_float, predict_action_batch, sha256_file
from paper34.imagenet_actions import ACTION_NAMES
from paper34.imagenetc_digital import digital_corruptions, iter_image_records


def split_indices() -> dict[str, tuple[int, int]]:
    return {"search": (0, 1999), "validation": (2000, 6999), "test": (7000, 16999)}


def indices_for_split(split: str, limit: int | None = None) -> list[int]:
    if split not in split_indices():
        raise ValueError(f"unknown split: {split}")
    start, end = split_indices()[split]
    count = end - start + 1 if limit is None else int(limit)
    if count < 1:
        raise ValueError("limit must be positive")
    if start + count - 1 > end:
        raise ValueError(f"limit {count} exceeds split {split} size")
    return list(range(start, start + count))


class CellAccumulator:
    def __init__(self) -> None:
        self.n = 0
        self.correct = Counter()
        self.oracle_correct = 0
        self.winner_counts = Counter()

    def update(self, label: int, predictions: dict[str, int]) -> str:
        self.n += 1
        correct_actions = []
        for action in ACTION_NAMES:
            if int(predictions[action]) == int(label):
                self.correct[action] += 1
                correct_actions.append(action)
        if correct_actions:
            self.oracle_correct += 1
            winner = correct_actions[0]
        else:
            winner = "none"
        self.winner_counts[winner] += 1
        return winner

    def summary(self) -> dict:
        if self.n < 1:
            raise ValueError("cannot summarize an empty cell")

        accuracies = {
            f"{action}_accuracy": 100.0 * float(self.correct[action]) / float(self.n)
            for action in ACTION_NAMES
        }
        best_action = max(ACTION_NAMES, key=lambda action: (accuracies[f"{action}_accuracy"], -ACTION_NAMES.index(action)))
        best_fixed = accuracies[f"{best_action}_accuracy"]
        oracle = 100.0 * float(self.oracle_correct) / float(self.n)
        order_gap_signed = accuracies["config_a_accuracy"] - accuracies["config_b_accuracy"]
        non_none_counts = {key: value for key, value in self.winner_counts.items() if key != "none"}
        non_none_total = sum(non_none_counts.values())
        entropy = _entropy_bits(non_none_counts.values()) if non_none_total else 0.0
        top_share = max(non_none_counts.values()) / non_none_total if non_none_total else 0.0

        result = dict(accuracies)
        result.update(
            {
                "n": self.n,
                "best_fixed_action": best_action,
                "best_fixed_accuracy": best_fixed,
                "per_image_oracle_accuracy": oracle,
                "oracle_minus_best_fixed": oracle - best_fixed,
                "order_gap_signed": order_gap_signed,
                "order_gap_abs": abs(order_gap_signed),
                "oracle_winner_counts": dict(sorted(self.winner_counts.items())),
                "oracle_action_entropy_bits": entropy,
                "oracle_action_top_share": top_share,
            }
        )
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full ImageNet-C digital action evaluation.")
    parser.add_argument("--digital-root", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--severities", default="1,2,3,4,5")
    parser.add_argument("--splits", default="search,validation,test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dncnn-checkpoint", required=True)
    parser.add_argument("--resnet50-checkpoint", required=True)
    parser.add_argument("--kair-root", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-per-split", type=int, default=None)
    return parser


def run(args: argparse.Namespace) -> dict:
    import torch
    import torchvision

    start_perf = time.perf_counter()
    start_time = datetime.now(timezone.utc)
    device = _resolve_device(args.device, torch)
    severities = _parse_int_list(args.severities)
    splits = _parse_str_list(args.splits)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "action_eval.csv"
    json_path = outdir / "action_eval.json"

    resnet = load_resnet50(args.resnet50_checkpoint, device)
    dncnn = load_dncnn(args.dncnn_checkpoint, args.kair_root, device)

    summaries: dict[str, dict] = {split: {} for split in splits}
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_csv_fieldnames())
        writer.writeheader()
        for split in splits:
            for corruption in digital_corruptions():
                summaries[split].setdefault(corruption, {})
                for severity in severities:
                    accumulator = CellAccumulator()
                    indices = indices_for_split(split, limit=args.limit_per_split)
                    records = iter_image_records(args.digital_root, corruption, severity, indices=indices)
                    if len(records) != len(indices):
                        raise RuntimeError(
                            f"{split} {corruption} severity {severity} returned {len(records)} records, "
                            f"expected {len(indices)}"
                        )
                    for start in range(0, len(records), args.batch_size):
                        batch_records = records[start : start + args.batch_size]
                        images = [load_rgb_float(record.path) for record in batch_records]
                        batch_predictions = predict_action_batch(images, resnet, dncnn, device, torch)
                        for offset, record in enumerate(batch_records):
                            predictions = {action: int(batch_predictions[action][offset]) for action in ACTION_NAMES}
                            winner = accumulator.update(record.label, predictions)
                            row = _row_from_record(split, record, predictions, winner)
                            writer.writerow(row)
                    summaries[split][corruption][str(severity)] = accumulator.summary()
                    print(
                        f"{split} {corruption} severity {severity}: {len(records)} images",
                        flush=True,
                    )

    end_time = datetime.now(timezone.utc)
    payload = {
        "schema_version": 1,
        "command": " ".join(sys.argv),
        "start_time_utc": start_time.isoformat(),
        "end_time_utc": end_time.isoformat(),
        "elapsed_seconds": round(time.perf_counter() - start_perf, 3),
        "parameters": {
            "corruptions": digital_corruptions(),
            "severities": severities,
            "splits": splits,
            "split_indices": split_indices(),
            "batch_size": args.batch_size,
            "limit_per_split": args.limit_per_split,
            "actions": ACTION_NAMES,
        },
        "resources": {
            "digital_root": str(Path(args.digital_root).resolve()),
            "dncnn_checkpoint": str(Path(args.dncnn_checkpoint).resolve()),
            "dncnn_checkpoint_sha256": sha256_file(args.dncnn_checkpoint),
            "resnet50_checkpoint": str(Path(args.resnet50_checkpoint).resolve()),
            "resnet50_checkpoint_sha256": sha256_file(args.resnet50_checkpoint),
            "kair_root": str(Path(args.kair_root).resolve()),
        },
        "environment": _environment_metadata(torch, torchvision, device),
        "summaries": summaries,
        "outputs": {"csv": str(csv_path), "json": str(json_path)},
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "csv": str(csv_path)}, indent=2), flush=True)
    return payload


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


def _row_from_record(split: str, record, predictions: dict[str, int], winner: str) -> dict:
    row = {
        "split": split,
        "corruption": record.corruption,
        "severity": record.severity,
        "image_index": record.image_index,
        "path": str(record.path),
        "wnid": record.wnid,
        "label": record.label,
        "oracle_action": winner,
        "correct_action_count": sum(1 for action in ACTION_NAMES if predictions[action] == record.label),
    }
    for action in ACTION_NAMES:
        row[f"{action}_pred"] = predictions[action]
        row[f"{action}_correct"] = int(predictions[action] == record.label)
    return row


def _csv_fieldnames() -> list[str]:
    fields = [
        "split",
        "corruption",
        "severity",
        "image_index",
        "path",
        "wnid",
        "label",
        "oracle_action",
        "correct_action_count",
    ]
    for action in ACTION_NAMES:
        fields.extend([f"{action}_pred", f"{action}_correct"])
    return fields


def _parse_int_list(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("expected at least one integer")
    return result


def _parse_str_list(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("expected at least one split")
    for split in result:
        if split not in split_indices():
            raise ValueError(f"unknown split: {split}")
    return result


def _entropy_bits(counts) -> float:
    values = [float(count) for count in counts if count > 0]
    total = sum(values)
    if total <= 0.0:
        return 0.0
    return -sum((value / total) * math.log2(value / total) for value in values)


def _resolve_device(device_arg: str, torch_module):
    if device_arg == "auto":
        device_arg = "cuda" if torch_module.cuda.is_available() else "cpu"
    return torch_module.device(device_arg)


def _environment_metadata(torch_module, torchvision_module, device) -> dict:
    cuda_available = bool(torch_module.cuda.is_available())
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch_module.__version__,
        "torchvision": torchvision_module.__version__,
        "cuda_available": cuda_available,
        "device": str(device),
        "gpu": torch_module.cuda.get_device_name(0) if cuda_available else None,
    }


if __name__ == "__main__":
    raise SystemExit(main())

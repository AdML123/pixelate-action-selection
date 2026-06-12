from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from full_action_eval import indices_for_split, split_indices
from oracle_ceiling import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_dncnn,
    load_resnet50,
    load_rgb_float,
    sha256_file,
)
from paper34.features_imagenetc import FEATURE_NAMES, feature_row
from paper34.imagenet_actions import jpeg_roundtrip
from paper34.imagenetc_digital import digital_corruptions, iter_image_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract ImageNet-C digital router features.")
    parser.add_argument("--digital-root", required=True)
    parser.add_argument("--action-csv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--splits", default="search,validation")
    parser.add_argument("--severities", default="1,2,3,4,5")
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
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    splits = _parse_str_list(args.splits)
    severities = _parse_int_list(args.severities)

    _validate_inputs(args)
    resnet = load_resnet50(args.resnet50_checkpoint, device)
    dncnn = load_dncnn(args.dncnn_checkpoint, args.kair_root, device)

    cells = []
    commutator_band_rows = []
    commutator_band_indices = [FEATURE_NAMES.index(name) for name in ["comm_band_low", "comm_band_mid", "comm_band_high"]]
    for split in splits:
        for corruption in digital_corruptions():
            for severity in severities:
                matrix, records = _extract_cell(args, split, corruption, severity, resnet, dncnn, device, torch)
                commutator_band_rows.append(matrix[:, commutator_band_indices])
                npy_path = outdir / f"features_{corruption}_{severity}_{split}.npy"
                json_path = outdir / f"features_{corruption}_{severity}_{split}.json"
                np.save(npy_path, matrix)
                cell_summary = _cell_summary(matrix, split, corruption, severity, records, npy_path)
                json_path.write_text(json.dumps(cell_summary, indent=2, sort_keys=True), encoding="utf-8")
                cells.append(cell_summary)
                print(f"{split} {corruption} severity {severity}: {matrix.shape[0]} rows", flush=True)

    end_time = datetime.now(timezone.utc)
    payload = {
        "schema_version": 1,
        "command": " ".join(sys.argv),
        "start_time_utc": start_time.isoformat(),
        "end_time_utc": end_time.isoformat(),
        "elapsed_seconds": round(time.perf_counter() - start_perf, 3),
        "feature_names": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "parameters": {
            "corruptions": digital_corruptions(),
            "severities": severities,
            "splits": splits,
            "split_indices": split_indices(),
            "batch_size": args.batch_size,
            "limit_per_split": args.limit_per_split,
        },
        "resources": {
            "digital_root": str(Path(args.digital_root).resolve()),
            "action_csv": str(Path(args.action_csv).resolve()),
            "dncnn_checkpoint": str(Path(args.dncnn_checkpoint).resolve()),
            "dncnn_checkpoint_sha256": sha256_file(args.dncnn_checkpoint),
            "resnet50_checkpoint": str(Path(args.resnet50_checkpoint).resolve()),
            "resnet50_checkpoint_sha256": sha256_file(args.resnet50_checkpoint),
            "kair_root": str(Path(args.kair_root).resolve()),
        },
        "environment": _environment_metadata(torch, torchvision, device),
        "cells": cells,
        "aggregate": _aggregate_summary(cells, commutator_band_rows),
    }
    summary_path = outdir / "feature_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path)}, indent=2), flush=True)
    return payload


def main() -> int:
    run(build_parser().parse_args())
    return 0


def _extract_cell(args, split: str, corruption: str, severity: int, resnet, dncnn, device, torch_module):
    indices = indices_for_split(split, limit=args.limit_per_split)
    records = iter_image_records(args.digital_root, corruption, severity, indices=indices)
    if len(records) != len(indices):
        raise RuntimeError(f"{split} {corruption} severity {severity} returned {len(records)} records")

    rows = []
    for start in range(0, len(records), args.batch_size):
        batch_records = records[start : start + args.batch_size]
        images = [load_rgb_float(record.path) for record in batch_records]
        batch_rows = _extract_batch(images, resnet, dncnn, device, torch_module)
        rows.extend(batch_rows)
    return np.stack(rows).astype(np.float32), records


def _extract_batch(images: list[np.ndarray], resnet, dncnn, device, torch_module) -> list[np.ndarray]:
    torch = torch_module
    identity_np = np.stack(images).astype(np.float32)
    jpeg20_np = np.stack([jpeg_roundtrip(image, quality=20) for image in identity_np]).astype(np.float32)

    with torch.inference_mode():
        identity = _nchw_tensor(identity_np, torch, device)
        jpeg20 = _nchw_tensor(jpeg20_np, torch, device)

        dncnn_identity = dncnn(identity).clamp(0.0, 1.0)
        config_a = dncnn(jpeg20).clamp(0.0, 1.0)
        config_b_np = np.stack([jpeg_roundtrip(image, quality=20) for image in _nhwc_array(dncnn_identity)]).astype(
            np.float32
        )
        config_b = _nchw_tensor(config_b_np, torch, device)

        identity_logits = _logits_tensor(resnet, identity, torch)
        dncnn_logits = _logits_tensor(resnet, dncnn_identity, torch)
        jpeg20_logits = _logits_tensor(resnet, jpeg20, torch)

        dncnn_np = _nhwc_array(dncnn_identity)
        config_a_np = _nhwc_array(config_a)
        config_b_np = _nhwc_array(config_b)
        identity_logits_np = identity_logits.detach().cpu().numpy().astype(np.float32)
        dncnn_logits_np = dncnn_logits.detach().cpu().numpy().astype(np.float32)
        jpeg20_logits_np = jpeg20_logits.detach().cpu().numpy().astype(np.float32)

    return [
        feature_row(
            identity_np[index],
            dncnn_np[index],
            jpeg20_np[index],
            config_a_np[index],
            config_b_np[index],
            identity_logits_np[index],
            dncnn_logits_np[index],
            jpeg20_logits_np[index],
        )
        for index in range(identity_np.shape[0])
    ]


def _cell_summary(matrix: np.ndarray, split: str, corruption: str, severity: int, records, npy_path: Path) -> dict:
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    return {
        "split": split,
        "corruption": corruption,
        "severity": severity,
        "n_rows": int(matrix.shape[0]),
        "feature_names": FEATURE_NAMES,
        "mean": {name: float(means[index]) for index, name in enumerate(FEATURE_NAMES)},
        "std": {name: float(stds[index]) for index, name in enumerate(FEATURE_NAMES)},
        "image_index_min": int(min(record.image_index for record in records)),
        "image_index_max": int(max(record.image_index for record in records)),
        "npy": str(npy_path),
    }


def _aggregate_summary(cells: list[dict], commutator_band_rows: list[np.ndarray]) -> dict:
    by_split: dict[str, dict] = {}
    for cell in cells:
        split = cell["split"]
        by_split.setdefault(split, {})
        by_split[split].setdefault(cell["corruption"], {})
        by_split[split][cell["corruption"]][str(cell["severity"])] = {
            "n_rows": cell["n_rows"],
            "hfer_input_mean": cell["mean"]["hfer_input"],
            "m_comm_mean": cell["mean"]["m_comm"],
            "hfer_jpeg20_residual_mean": cell["mean"]["hfer_jpeg20_residual"],
            "hfer_dncnn_residual_mean": cell["mean"]["hfer_dncnn_residual"],
            "identity_margin_mean": cell["mean"]["identity_margin"],
            "dncnn_margin_mean": cell["mean"]["dncnn_margin"],
            "jpeg20_margin_mean": cell["mean"]["jpeg20_margin"],
        }
    return {
        "by_split": by_split,
        "commutator_band_correlations": _commutator_band_correlations(commutator_band_rows),
    }


def _commutator_band_correlations(commutator_band_rows: list[np.ndarray]) -> dict:
    if not commutator_band_rows:
        return {"basis": "all_extracted_rows", "feature_names": ["comm_band_low", "comm_band_mid", "comm_band_high"], "matrix": []}
    rows = np.vstack(commutator_band_rows).astype(np.float64)
    if rows.shape[0] < 2:
        return {"basis": "all_extracted_rows", "feature_names": ["comm_band_low", "comm_band_mid", "comm_band_high"], "matrix": []}
    corr = np.corrcoef(rows, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return {
        "basis": "all_extracted_rows",
        "feature_names": ["comm_band_low", "comm_band_mid", "comm_band_high"],
        "n_rows": int(rows.shape[0]),
        "matrix": corr.tolist(),
    }


def _validate_inputs(args) -> None:
    for path in [args.digital_root, args.action_csv, args.dncnn_checkpoint, args.resnet50_checkpoint, args.kair_root]:
        if not Path(path).exists():
            raise FileNotFoundError(path)


def _nchw_tensor(images: np.ndarray, torch_module, device):
    return torch_module.from_numpy(images).permute(0, 3, 1, 2).contiguous().to(device)


def _nhwc_array(tensor) -> np.ndarray:
    return tensor.detach().clamp(0.0, 1.0).cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)


def _logits_tensor(resnet, tensor, torch_module):
    mean = torch_module.tensor(IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch_module.tensor(IMAGENET_STD, device=tensor.device).view(1, 3, 1, 1)
    return resnet((tensor - mean) / std)


def _parse_int_list(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("expected at least one severity")
    return result


def _parse_str_list(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("expected at least one split")
    for split in result:
        if split not in split_indices():
            raise ValueError(f"unknown split: {split}")
    return result


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

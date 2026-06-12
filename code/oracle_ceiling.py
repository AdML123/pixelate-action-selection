from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from paper34.imagenet_actions import ACTION_NAMES, jpeg_roundtrip, oracle_summary_from_correctness
from paper34.imagenetc_digital import digital_corruptions, discover_wnids, iter_image_records, select_indices


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def decide_go_nogo(summaries: dict[str, dict]) -> dict:
    pixelate = summaries["pixelate"]
    jpeg = summaries.get("jpeg_compression", {})
    contrast = summaries.get("contrast", {})

    pixel_gap = float(pixelate["order_gap_abs"])
    pixel_oracle = float(pixelate["oracle_minus_best_fixed"])
    jpeg_oracle = float(jpeg.get("oracle_minus_best_fixed", 0.0))
    contrast_safety = float(contrast.get("best_fixed_accuracy", 0.0)) - float(
        contrast.get("config_a_accuracy", 0.0)
    )

    if pixel_gap < 0.5 or all(float(row.get("oracle_minus_best_fixed", 0.0)) < 1.0 for row in summaries.values()):
        status = "no_go"
    elif pixel_gap >= 2.0 and (pixel_oracle >= 3.0 or jpeg_oracle >= 3.0) and contrast_safety <= 0.25:
        status = "go"
    else:
        status = "borderline"

    return {
        "status": status,
        "pixelate_order_gap_abs": pixel_gap,
        "pixelate_oracle_minus_best_fixed": pixel_oracle,
        "jpeg_oracle_minus_best_fixed": jpeg_oracle,
        "contrast_safety_deficit": contrast_safety,
    }


def build_output_paths(outdir: Path | str, severity: int, limit_images: int) -> dict[str, Path]:
    root = Path(outdir)
    stem = f"oracle_ceiling_s{severity}_{limit_images}"
    return {"csv": root / f"{stem}.csv", "json": root / f"{stem}.json"}


def load_rgb_float(path: Path | str, image_size: int = 224) -> np.ndarray:
    try:
        resample = Image.Resampling.BICUBIC
    except AttributeError:  # pragma: no cover - Pillow compatibility
        resample = Image.BICUBIC
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if rgb.size != (image_size, image_size):
            rgb = rgb.resize((image_size, image_size), resample=resample)
        return np.asarray(rgb, dtype=np.float32) / 255.0


def summarize_rows_by_corruption(rows: list[dict]) -> dict[str, dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["corruption"])].append(row)
    return {corruption: oracle_summary_from_correctness(group_rows) for corruption, group_rows in grouped.items()}


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_resnet50(checkpoint: Path | str, device):
    import torch
    from torchvision.models import resnet50

    model = resnet50(weights=None)
    state = torch.load(checkpoint, map_location="cpu")
    state = _unwrap_state_dict(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "ResNet-50 checkpoint is incompatible: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    return model.to(device).eval()


def load_dncnn(checkpoint: Path | str, kair_root: Path | str, device):
    import torch

    root = Path(kair_root)
    if not root.exists():
        raise FileNotFoundError(f"KAIR root not found: {root}")
    sys.path.insert(0, str(root))
    try:
        from models.network_dncnn import DnCNN
    except Exception as exc:  # pragma: no cover - exercised by integration run
        raise RuntimeError(f"failed to import KAIR DnCNN from {root}: {exc}") from exc

    model = DnCNN(in_nc=3, out_nc=3, nc=64, nb=20, act_mode="R")
    state = torch.load(checkpoint, map_location="cpu")
    state = _unwrap_state_dict(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "DnCNN checkpoint is incompatible with KAIR color DnCNN: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    return model.to(device).eval()


def run_oracle(args: argparse.Namespace) -> tuple[dict, dict[str, Path]]:
    import torch
    import torchvision

    start_perf = time.perf_counter()
    start_time = datetime.now(timezone.utc)
    device = _resolve_device(args.device, torch)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output_paths = build_output_paths(outdir, args.severity, args.limit_images)

    resnet = load_resnet50(args.resnet50_checkpoint, device)
    dncnn = load_dncnn(args.dncnn_checkpoint, args.kair_root, device)

    indices = select_indices(args.limit_images)
    rows: list[dict] = []
    for corruption in digital_corruptions():
        wnids = discover_wnids(args.digital_root, corruption, args.severity)
        if len(wnids) != 1000:
            raise RuntimeError(f"{corruption} severity {args.severity} exposes {len(wnids)} WNID dirs, expected 1000")
        records = iter_image_records(args.digital_root, corruption, args.severity, indices=indices)
        if len(records) != args.limit_images:
            raise RuntimeError(f"{corruption} returned {len(records)} records, expected {args.limit_images}")

        for start in range(0, len(records), args.batch_size):
            batch_records = records[start : start + args.batch_size]
            images = [load_rgb_float(record.path, args.image_size) for record in batch_records]
            predictions = predict_action_batch(images, resnet, dncnn, device, torch)
            for index, record in enumerate(batch_records):
                row = {
                    "corruption": record.corruption,
                    "severity": record.severity,
                    "image_index": record.image_index,
                    "path": str(record.path),
                    "wnid": record.wnid,
                    "label": record.label,
                }
                for action in ACTION_NAMES:
                    pred = int(predictions[action][index])
                    row[f"{action}_pred"] = pred
                    row[f"{action}_correct"] = int(pred == record.label)
                rows.append(row)
            done = min(start + args.batch_size, len(records))
            print(f"{corruption}: {done}/{len(records)}", flush=True)

    summaries = summarize_rows_by_corruption(rows)
    decision = decide_go_nogo(summaries)
    end_time = datetime.now(timezone.utc)
    payload = {
        "schema_version": 1,
        "command": " ".join(sys.argv),
        "start_time_utc": start_time.isoformat(),
        "end_time_utc": end_time.isoformat(),
        "elapsed_seconds": round(time.perf_counter() - start_perf, 3),
        "parameters": {
            "corruptions": digital_corruptions(),
            "severity": args.severity,
            "limit_images": args.limit_images,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "resize_policy": "bicubic only if source is not image_size x image_size",
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
        "decision": decision,
        "outputs": {name: str(path) for name, path in output_paths.items()},
    }

    write_rows_csv(output_paths["csv"], rows)
    output_paths["json"].write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)
    return payload, output_paths


def predict_action_batch(images: list[np.ndarray], resnet, dncnn, device, torch_module) -> dict[str, list[int]]:
    torch = torch_module
    with torch.inference_mode():
        identity_np = np.stack(images).astype(np.float32)
        jpeg20_np = _jpeg_batch(identity_np, quality=20)
        jpeg10_np = _jpeg_batch(identity_np, quality=10)

        identity = _nchw_tensor(identity_np, torch, device)
        jpeg20 = _nchw_tensor(jpeg20_np, torch, device)
        jpeg10 = _nchw_tensor(jpeg10_np, torch, device)

        dncnn_identity = dncnn(identity).clamp(0.0, 1.0)
        config_a = dncnn(jpeg20).clamp(0.0, 1.0)
        config_a10 = dncnn(jpeg10).clamp(0.0, 1.0)
        config_b_np = _jpeg_batch(_nhwc_array(dncnn_identity), quality=20)
        config_b = _nchw_tensor(config_b_np, torch, device)

        action_tensors = {
            "identity": identity,
            "dncnn": dncnn_identity,
            "jpeg20": jpeg20,
            "jpeg10": jpeg10,
            "config_a": config_a,
            "config_b": config_b,
            "config_a10": config_a10,
        }
        return {action: _predict_tensor(resnet, action_tensors[action], torch) for action in ACTION_NAMES}


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write oracle CSV without rows")
    fieldnames = [
        "corruption",
        "severity",
        "image_index",
        "path",
        "wnid",
        "label",
    ]
    for action in ACTION_NAMES:
        fieldnames.extend([f"{action}_pred", f"{action}_correct"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ImageNet-C digital oracle ceiling smoke.")
    parser.add_argument("--digital-root", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--severity", type=int, default=3)
    parser.add_argument("--limit-images", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dncnn-checkpoint", required=True)
    parser.add_argument("--resnet50-checkpoint", required=True)
    parser.add_argument("--kair-root", required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device string")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_oracle(args)
    return 0


def _unwrap_state_dict(state):
    if isinstance(state, dict):
        for key in ("state_dict", "model", "net"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


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


def _jpeg_batch(images: np.ndarray, quality: int) -> np.ndarray:
    return np.stack([jpeg_roundtrip(image, quality=quality) for image in images]).astype(np.float32)


def _nchw_tensor(images: np.ndarray, torch_module, device):
    return torch_module.from_numpy(images).permute(0, 3, 1, 2).contiguous().to(device)


def _nhwc_array(tensor) -> np.ndarray:
    return tensor.detach().clamp(0.0, 1.0).cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)


def _predict_tensor(resnet, tensor, torch_module) -> list[int]:
    mean = torch_module.tensor(IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch_module.tensor(IMAGENET_STD, device=tensor.device).view(1, 3, 1, 1)
    logits = resnet((tensor - mean) / std)
    return logits.argmax(dim=1).detach().cpu().tolist()


if __name__ == "__main__":
    raise SystemExit(main())

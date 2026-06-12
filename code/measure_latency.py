from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from oracle_ceiling import IMAGENET_MEAN, IMAGENET_STD, load_dncnn, load_resnet50, load_rgb_float, sha256_file
from pixelate_router.features_imagenetc import feature_row
from pixelate_router.imagenet_actions import jpeg_roundtrip
from pixelate_router.imagenetc_digital import iter_image_records
from pixelate_router.router import select_with_threshold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure batch-size-one preprocessing latency.")
    parser.add_argument("--digital-root", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--dncnn-checkpoint", required=True)
    parser.add_argument("--resnet50-checkpoint", required=True)
    parser.add_argument("--kair-root", required=True)
    parser.add_argument("--router-checkpoint", required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--measure", type=int, default=200)
    parser.add_argument("--device", default="auto")
    return parser


def run(args: argparse.Namespace) -> dict:
    import torch

    start_perf = time.perf_counter()
    start_time = datetime.now(timezone.utc)
    device = _resolve_device(args.device, torch)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    resnet = load_resnet50(args.resnet50_checkpoint, device)
    dncnn = load_dncnn(args.dncnn_checkpoint, args.kair_root, device)
    router_model, router_meta = _load_router(args.router_checkpoint, torch, device)
    image = _sample_image(args.digital_root)
    tensor = _nchw_tensor(image[None, ...], torch, device)

    modes = {
        "default_dncnn": lambda: _default_dncnn(tensor, dncnn, resnet, torch),
        "config_a": lambda: _config_a(image, dncnn, resnet, torch, device),
        "config_b": lambda: _config_b(image, dncnn, resnet, torch, device),
        "oracle_eval": lambda: _oracle_eval(image, dncnn, resnet, torch, device),
        "router": lambda: _router_path(image, dncnn, resnet, router_model, router_meta, torch, device),
    }
    results = {}
    for name, fn in modes.items():
        results[name] = _measure(fn, args.warmup, args.measure, torch, device)
        print(f"{name}: median {results[name]['median_ms']:.3f} ms", flush=True)

    payload = {
        "schema_version": 1,
        "command": " ".join(sys.argv),
        "start_time_utc": start_time.isoformat(),
        "end_time_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - start_perf, 3),
        "environment": _environment_metadata(torch, device),
        "parameters": {"warmup": args.warmup, "measure": args.measure, "batch_size": 1},
        "resources": {
            "digital_root": str(Path(args.digital_root).resolve()),
            "dncnn_checkpoint": str(Path(args.dncnn_checkpoint).resolve()),
            "dncnn_checkpoint_sha256": sha256_file(args.dncnn_checkpoint),
            "resnet50_checkpoint": str(Path(args.resnet50_checkpoint).resolve()),
            "resnet50_checkpoint_sha256": sha256_file(args.resnet50_checkpoint),
            "router_checkpoint": str(Path(args.router_checkpoint).resolve()),
        },
        "results": results,
    }
    path = outdir / "timing_report.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"timing_report": str(path)}, indent=2), flush=True)
    return payload


def _measure(fn, warmup: int, measure: int, torch, device) -> dict:
    for _ in range(warmup):
        fn()
    _synchronize(torch, device)
    values = []
    for _ in range(measure):
        _synchronize(torch, device)
        start = time.perf_counter()
        fn()
        _synchronize(torch, device)
        values.append((time.perf_counter() - start) * 1000.0)
    values_sorted = sorted(values)
    return {
        "median_ms": float(statistics.median(values_sorted)),
        "p25_ms": float(np.percentile(values_sorted, 25)),
        "p75_ms": float(np.percentile(values_sorted, 75)),
        "mean_ms": float(np.mean(values_sorted)),
        "n": int(measure),
    }


def _default_dncnn(tensor, dncnn, resnet, torch):
    with torch.inference_mode():
        out = dncnn(tensor).clamp(0.0, 1.0)
        return _logits_tensor(resnet, out, torch)


def _config_a(image, dncnn, resnet, torch, device):
    jpeg20 = jpeg_roundtrip(image, quality=20)
    with torch.inference_mode():
        out = dncnn(_nchw_tensor(jpeg20[None, ...], torch, device)).clamp(0.0, 1.0)
        return _logits_tensor(resnet, out, torch)


def _config_b(image, dncnn, resnet, torch, device):
    with torch.inference_mode():
        dncnn_out = dncnn(_nchw_tensor(image[None, ...], torch, device)).clamp(0.0, 1.0)
    jpeg20 = jpeg_roundtrip(_nhwc_array(dncnn_out)[0], quality=20)
    with torch.inference_mode():
        return _logits_tensor(resnet, _nchw_tensor(jpeg20[None, ...], torch, device), torch)


def _oracle_eval(image, dncnn, resnet, torch, device):
    actions = []
    jpeg20 = jpeg_roundtrip(image, quality=20)
    jpeg10 = jpeg_roundtrip(image, quality=10)
    with torch.inference_mode():
        identity = _nchw_tensor(image[None, ...], torch, device)
        dncnn_image = dncnn(identity).clamp(0.0, 1.0)
        config_a = dncnn(_nchw_tensor(jpeg20[None, ...], torch, device)).clamp(0.0, 1.0)
        config_a10 = dncnn(_nchw_tensor(jpeg10[None, ...], torch, device)).clamp(0.0, 1.0)
        config_b_np = jpeg_roundtrip(_nhwc_array(dncnn_image)[0], quality=20)
        identity_logits = _logits_tensor(resnet, identity, torch)
        dncnn_logits = _logits_tensor(resnet, dncnn_image, torch)
        jpeg20_tensor = _nchw_tensor(jpeg20[None, ...], torch, device)
        jpeg20_logits = _logits_tensor(resnet, jpeg20_tensor, torch)
        _ = feature_row(
            image,
            _nhwc_array(dncnn_image)[0],
            jpeg20,
            _nhwc_array(config_a)[0],
            config_b_np,
            identity_logits.detach().cpu().numpy()[0],
            dncnn_logits.detach().cpu().numpy()[0],
            jpeg20_logits.detach().cpu().numpy()[0],
        )
        tensors = [
            identity,
            dncnn_image,
            jpeg20_tensor,
            _nchw_tensor(jpeg10[None, ...], torch, device),
            config_a,
            _nchw_tensor(config_b_np[None, ...], torch, device),
            config_a10,
        ]
        for tensor in tensors:
            actions.append(_logits_tensor(resnet, tensor, torch))
    return actions


def _router_path(image, dncnn, resnet, router_model, router_meta, torch, device):
    jpeg20 = jpeg_roundtrip(image, quality=20)
    with torch.inference_mode():
        identity = _nchw_tensor(image[None, ...], torch, device)
        jpeg20_t = _nchw_tensor(jpeg20[None, ...], torch, device)
        dncnn_image = dncnn(identity).clamp(0.0, 1.0)
        config_a = dncnn(jpeg20_t).clamp(0.0, 1.0)
        config_b_np = jpeg_roundtrip(_nhwc_array(dncnn_image)[0], quality=20)
        config_b_t = _nchw_tensor(config_b_np[None, ...], torch, device)
        identity_logits = _logits_tensor(resnet, identity, torch)
        dncnn_logits = _logits_tensor(resnet, dncnn_image, torch)
        jpeg20_logits = _logits_tensor(resnet, jpeg20_t, torch)
        row = feature_row(
            image,
            _nhwc_array(dncnn_image)[0],
            jpeg20,
            _nhwc_array(config_a)[0],
            _nhwc_array(config_b_t)[0],
            identity_logits.detach().cpu().numpy()[0],
            dncnn_logits.detach().cpu().numpy()[0],
            jpeg20_logits.detach().cpu().numpy()[0],
        )
        x = ((row[router_meta["feature_indices"]] - router_meta["mean"]) / router_meta["std"]).astype(np.float32)
        scores = torch.sigmoid(router_model(torch.from_numpy(x[None, ...]).float().to(device)))
        selected = select_with_threshold(scores.detach().cpu().numpy()[0], router_meta["default_index"], router_meta["tau"])
        action = router_meta["actions"][selected]
        if action == "identity":
            return identity_logits
        if action == "dncnn":
            return dncnn_logits
        if action == "jpeg20":
            return jpeg20_logits
        if action == "jpeg10":
            jpeg10 = jpeg_roundtrip(image, quality=10)
            return _logits_tensor(resnet, _nchw_tensor(jpeg10[None, ...], torch, device), torch)
        if action == "config_a":
            return _logits_tensor(resnet, config_a, torch)
        if action == "config_b":
            return _logits_tensor(resnet, config_b_t, torch)
        jpeg10 = jpeg_roundtrip(image, quality=10)
        config_a10 = dncnn(_nchw_tensor(jpeg10[None, ...], torch, device)).clamp(0.0, 1.0)
        return _logits_tensor(resnet, config_a10, torch)


def _load_router(path, torch, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = torch.nn.Linear(len(checkpoint["feature_names"]), len(checkpoint["actions"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    from pixelate_router.features_imagenetc import FEATURE_NAMES

    meta = {
        "actions": checkpoint["actions"],
        "default_index": checkpoint["actions"].index(checkpoint["default_action"]),
        "feature_indices": [FEATURE_NAMES.index(name) for name in checkpoint["feature_names"]],
        "mean": checkpoint["scaler"]["mean"],
        "std": checkpoint["scaler"]["std"],
        "tau": float(checkpoint["report"]["tau_selected"]),
    }
    return model, meta


def _sample_image(digital_root: str) -> np.ndarray:
    record = iter_image_records(digital_root, "pixelate", 3, indices=[7000])[0]
    return load_rgb_float(record.path)


def _nchw_tensor(images: np.ndarray, torch, device):
    return torch.from_numpy(images.astype(np.float32)).permute(0, 3, 1, 2).contiguous().to(device)


def _nhwc_array(tensor) -> np.ndarray:
    return tensor.detach().clamp(0.0, 1.0).cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)


def _logits_tensor(resnet, tensor, torch):
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(1, 3, 1, 1)
    return resnet((tensor - mean) / std)


def _synchronize(torch, device) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


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

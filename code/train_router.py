from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pixelate_router.features_imagenetc import FEATURE_NAMES
from pixelate_router.router import paired_lcb, select_with_threshold


ACTIONS = ["identity", "dncnn", "jpeg20", "jpeg10", "config_a", "config_b", "config_a10"]
DEFAULT_ACTION = "dncnn"
DEFAULT_INDEX = ACTIONS.index(DEFAULT_ACTION)
CORRUPTIONS = ["contrast", "elastic_transform", "pixelate", "jpeg_compression"]
SEVERITIES = [1, 2, 3, 4, 5]
FEATURE_SETS = {
    "all": FEATURE_NAMES,
    "no_commutator": [
        name
        for name in FEATURE_NAMES
        if name not in {"m_comm", "comm_band_low", "comm_band_mid", "comm_band_high"}
    ],
    "spectral": [
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
    ],
    "confidence": [
        "identity_top1_prob",
        "identity_margin",
        "identity_entropy",
        "dncnn_top1_prob",
        "dncnn_margin",
        "dncnn_entropy",
        "jpeg20_top1_prob",
        "jpeg20_margin",
        "jpeg20_entropy",
    ],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ImageNet-C digital offline logistic action router.")
    parser.add_argument("--features-root", required=True)
    parser.add_argument("--action-csv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--taus", default="0.00,0.01,0.02,0.03,0.05,0.10,0.15")
    parser.add_argument("--device", default="auto")
    return parser


def run(args: argparse.Namespace) -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    start_perf = time.perf_counter()
    start_time = datetime.now(timezone.utc)
    _set_seed(args.seed, torch)
    device = _resolve_device(args.device, torch)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    features_root = Path(args.features_root)
    action_rows = _load_action_rows(Path(args.action_csv))
    taus = _parse_float_list(args.taus)

    reports = {}
    for feature_set_name, names in FEATURE_SETS.items():
        train_data = _load_split(features_root, action_rows, "search", names)
        val_data = _load_split(features_root, action_rows, "validation", names)
        scaler = _fit_scaler(train_data["x"])
        x_train = _apply_scaler(train_data["x"], scaler)
        x_val = _apply_scaler(val_data["x"], scaler)

        model = nn.Linear(x_train.shape[1], len(ACTIONS)).to(device)
        history = _train_model(model, x_train, train_data["y"], x_val, val_data["y"], args, torch, F, device)
        report = _validate_model(model, x_val, val_data, taus, feature_set_name, history, scaler, torch, device)
        report_path = outdir / f"validation_report_{feature_set_name}.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        checkpoint_path = outdir / f"router_{feature_set_name}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_set": feature_set_name,
                "feature_names": names,
                "actions": ACTIONS,
                "default_action": DEFAULT_ACTION,
                "scaler": scaler,
                "report": report,
            },
            checkpoint_path,
        )
        report["checkpoint"] = str(checkpoint_path)
        reports[feature_set_name] = report

    main_report = {
        "schema_version": 1,
        "command": " ".join(sys.argv),
        "start_time_utc": start_time.isoformat(),
        "end_time_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - start_perf, 3),
        "environment": _environment_metadata(torch, device),
        "parameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "seed": args.seed,
            "taus": taus,
            "feature_sets": list(FEATURE_SETS),
        },
        "default_action": DEFAULT_ACTION,
        "feature_set": "all",
        "selected_report": reports["all"],
        "reports": reports,
    }
    validation_report_path = outdir / "validation_report.json"
    validation_report_path.write_text(json.dumps(main_report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"validation_report": str(validation_report_path)}, indent=2), flush=True)
    return main_report


def _train_model(model, x_train, y_train, x_val, y_val, args, torch, F, device) -> list[dict]:
    x_train_t = torch.from_numpy(x_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    x_val_t = torch.from_numpy(x_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_loss = float("inf")
    stale = 0
    history = []
    n = x_train_t.shape[0]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        train_losses = []
        for start in range(0, n, args.batch_size):
            batch_idx = permutation[start : start + args.batch_size].to(device)
            logits = model(x_train_t[batch_idx])
            loss = F.binary_cross_entropy_with_logits(logits, y_train_t[batch_idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            val_loss = float(F.binary_cross_entropy_with_logits(model(x_val_t), y_val_t).detach().cpu())
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_loss": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def _validate_model(model, x_val, val_data, taus, feature_set_name, history, scaler, torch, device) -> dict:
    x_val_t = torch.from_numpy(x_val).float().to(device)
    model.eval()
    with torch.inference_mode():
        scores = torch.sigmoid(model(x_val_t)).detach().cpu().numpy().astype(np.float32)
    default_correct = val_data["y"][:, DEFAULT_INDEX].astype(np.int64)
    tau_reports = {}
    best_tau = None
    best_key = None
    for tau in taus:
        selected = np.array([select_with_threshold(row, DEFAULT_INDEX, tau) for row in scores], dtype=np.int64)
        router_correct = val_data["y"][np.arange(val_data["y"].shape[0]), selected].astype(np.int64)
        lcb = paired_lcb(router_correct, default_correct)
        gain = float((router_correct.mean() - default_correct.mean()) * 100.0)
        contrast_gain = _corruption_gain(router_correct, default_correct, val_data["corruption"], "contrast")
        tau_key = f"{tau:.2f}"
        tau_reports[tau_key] = {
            "tau": tau,
            "validation_accuracy_router": float(router_correct.mean() * 100.0),
            "validation_accuracy_default": float(default_correct.mean() * 100.0),
            "validation_gain": gain,
            "paired_lcb": float(lcb * 100.0),
            "contrast_gain": contrast_gain,
            "action_distribution": _action_distribution(selected),
            "per_corruption": _per_corruption_report(router_correct, default_correct, selected, val_data["corruption"]),
        }
        candidate = (lcb, gain, -tau)
        if best_key is None or candidate > best_key:
            best_key = candidate
            best_tau = tau
    selected_report = tau_reports[f"{best_tau:.2f}"].copy()
    selected_report.update(
        {
            "feature_set": feature_set_name,
            "default_action": DEFAULT_ACTION,
            "tau_selected": best_tau,
            "epochs_ran": len(history),
            "best_validation_loss": min(item["validation_loss"] for item in history),
            "history": history,
            "feature_names": list(FEATURE_SETS[feature_set_name]),
            "feature_mean": scaler["mean"].tolist(),
            "feature_std": scaler["std"].tolist(),
            "tau_sweep": tau_reports,
        }
    )
    return selected_report


def _load_split(features_root: Path, action_rows: dict, split: str, feature_names: list[str]) -> dict:
    indices = [FEATURE_NAMES.index(name) for name in feature_names]
    x_parts = []
    y_parts = []
    corruption = []
    severity_values = []
    image_index = []
    for corr in CORRUPTIONS:
        for severity in SEVERITIES:
            feature_path = features_root / f"features_{corr}_{severity}_{split}.npy"
            if not feature_path.exists():
                raise FileNotFoundError(feature_path)
            features = np.load(feature_path).astype(np.float32)[:, indices]
            rows = action_rows[(split, corr, severity)]
            if features.shape[0] != len(rows):
                raise RuntimeError(f"{feature_path} has {features.shape[0]} rows but CSV has {len(rows)}")
            rewards = np.asarray(
                [[int(row[f"{action}_correct"]) for action in ACTIONS] for row in rows],
                dtype=np.float32,
            )
            x_parts.append(features)
            y_parts.append(rewards)
            corruption.extend([corr] * features.shape[0])
            severity_values.extend([severity] * features.shape[0])
            image_index.extend([int(row["image_index"]) for row in rows])
    return {
        "x": np.vstack(x_parts),
        "y": np.vstack(y_parts),
        "corruption": np.asarray(corruption),
        "severity": np.asarray(severity_values, dtype=np.int64),
        "image_index": np.asarray(image_index, dtype=np.int64),
    }


def _load_action_rows(action_csv: Path) -> dict:
    rows = defaultdict(list)
    with action_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] in {"search", "validation"}:
                rows[(row["split"], row["corruption"], int(row["severity"]))].append(row)
    return rows


def _fit_scaler(x: np.ndarray) -> dict:
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return {"mean": mean, "std": std}


def _apply_scaler(x: np.ndarray, scaler: dict) -> np.ndarray:
    return ((x - scaler["mean"]) / scaler["std"]).astype(np.float32)


def _per_corruption_report(router_correct, default_correct, selected, corruptions) -> dict:
    report = {}
    for corr in CORRUPTIONS:
        mask = corruptions == corr
        report[corr] = {
            "n": int(mask.sum()),
            "router_accuracy": float(router_correct[mask].mean() * 100.0),
            "default_accuracy": float(default_correct[mask].mean() * 100.0),
            "gain": float((router_correct[mask].mean() - default_correct[mask].mean()) * 100.0),
            "paired_lcb": float(paired_lcb(router_correct[mask], default_correct[mask]) * 100.0),
            "action_distribution": _action_distribution(selected[mask]),
        }
    return report


def _corruption_gain(router_correct, default_correct, corruptions, corruption: str) -> float:
    mask = corruptions == corruption
    return float((router_correct[mask].mean() - default_correct[mask].mean()) * 100.0)


def _action_distribution(selected: np.ndarray) -> dict:
    counts = Counter(int(item) for item in selected)
    total = int(len(selected))
    return {
        ACTIONS[index]: {
            "count": int(counts.get(index, 0)),
            "share": float(counts.get(index, 0) / total) if total else 0.0,
        }
        for index in range(len(ACTIONS))
    }


def _parse_float_list(value: str) -> list[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("expected at least one tau")
    return result


def _resolve_device(device: str, torch):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _set_seed(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

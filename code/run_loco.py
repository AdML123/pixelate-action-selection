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
from train_router import ACTIONS, CORRUPTIONS, DEFAULT_ACTION, DEFAULT_INDEX, FEATURE_SETS, SEVERITIES
from train_router import _apply_scaler, _fit_scaler, _parse_float_list, _set_seed, _train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run leave-one-corruption-out router validation.")
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

    folds = {}
    for holdout in CORRUPTIONS:
        train_corruptions = [corr for corr in CORRUPTIONS if corr != holdout]
        train_data = _load_subset(features_root, action_rows, "search", train_corruptions, FEATURE_SETS["all"])
        eval_data = _load_subset(features_root, action_rows, "validation", [holdout], FEATURE_SETS["all"])
        scaler = _fit_scaler(train_data["x"])
        x_train = _apply_scaler(train_data["x"], scaler)
        x_eval = _apply_scaler(eval_data["x"], scaler)
        model = torch.nn.Linear(x_train.shape[1], len(ACTIONS)).to(device)
        history = _train_model(model, x_train, train_data["y"], x_eval, eval_data["y"], args, torch, F, device)
        fold = _evaluate_fold(model, x_eval, eval_data, taus, history, holdout, train_corruptions, torch, device)
        checkpoint_path = outdir / f"loco_router_holdout_{holdout}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_names": FEATURE_SETS["all"],
                "actions": ACTIONS,
                "default_action": DEFAULT_ACTION,
                "holdout": holdout,
                "train_corruptions": train_corruptions,
                "scaler": scaler,
                "report": fold,
            },
            checkpoint_path,
        )
        fold["checkpoint"] = str(checkpoint_path)
        folds[holdout] = fold

    payload = {
        "schema_version": 1,
        "command": " ".join(sys.argv),
        "start_time_utc": start_time.isoformat(),
        "end_time_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - start_perf, 3),
        "environment": _environment_metadata(torch, device),
        "default_action": DEFAULT_ACTION,
        "feature_set": "all",
        "parameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "seed": args.seed,
            "taus": taus,
        },
        "folds": folds,
    }
    path = outdir / "loco_report.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"loco_report": str(path)}, indent=2), flush=True)
    return payload


def _evaluate_fold(model, x_eval: np.ndarray, eval_data: dict, taus: list[float], history, holdout, train_corruptions, torch, device) -> dict:
    model.eval()
    with torch.inference_mode():
        scores = torch.sigmoid(model(torch.from_numpy(x_eval).float().to(device))).detach().cpu().numpy()
    default_correct = eval_data["y"][:, DEFAULT_INDEX].astype(np.int64)
    tau_reports = {}
    best_tau = None
    best_key = None
    for tau in taus:
        selected = np.array([select_with_threshold(row, DEFAULT_INDEX, tau) for row in scores], dtype=np.int64)
        correct = eval_data["y"][np.arange(eval_data["y"].shape[0]), selected].astype(np.int64)
        gain = float((correct.mean() - default_correct.mean()) * 100.0)
        lcb = float(paired_lcb(correct, default_correct) * 100.0)
        tau_key = f"{tau:.2f}"
        tau_reports[tau_key] = {
            "tau": tau,
            "accuracy": float(correct.mean() * 100.0),
            "default_accuracy": float(default_correct.mean() * 100.0),
            "gain": gain,
            "paired_lcb": lcb,
            "action_distribution": _action_distribution(selected),
            "jpeg_containing_share": _jpeg_containing_share(selected),
        }
        candidate = (lcb, gain, -tau)
        if best_key is None or candidate > best_key:
            best_key = candidate
            best_tau = tau
    selected_report = tau_reports[f"{best_tau:.2f}"].copy()
    selected_report.update(
        {
            "holdout": holdout,
            "train_corruptions": train_corruptions,
            "tau_selected": best_tau,
            "epochs_ran": len(history),
            "best_validation_loss": min(item["validation_loss"] for item in history),
            "history": history,
            "tau_sweep": tau_reports,
        }
    )
    return selected_report


def _load_subset(features_root: Path, action_rows: dict, split: str, corruptions: list[str], feature_names: list[str]) -> dict:
    indices = [FEATURE_NAMES.index(name) for name in feature_names]
    x_parts = []
    y_parts = []
    for corr in corruptions:
        for severity in SEVERITIES:
            path = features_root / f"features_{corr}_{severity}_{split}.npy"
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
    return {"x": np.vstack(x_parts), "y": np.vstack(y_parts)}


def _load_action_rows(action_csv: Path) -> dict:
    rows = defaultdict(list)
    with action_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] in {"search", "validation"}:
                rows[(row["split"], row["corruption"], int(row["severity"]))].append(row)
    return rows


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


def _jpeg_containing_share(selected: np.ndarray) -> float:
    jpeg_indices = {ACTIONS.index("jpeg20"), ACTIONS.index("jpeg10"), ACTIONS.index("config_a"), ACTIONS.index("config_b"), ACTIONS.index("config_a10")}
    return float(np.isin(selected, list(jpeg_indices)).mean())


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

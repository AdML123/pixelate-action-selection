from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from paper34.features_imagenetc import FEATURE_NAMES
from paper34.router import paired_lcb, select_with_threshold
from train_router import ACTIONS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate pixelate per-severity numeric table.")
    parser.add_argument("--action-csv", required=True)
    parser.add_argument("--features-root-test", required=True)
    parser.add_argument("--router-checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.action_csv, args.features_root_test, args.router_checkpoint)
    write_csv(outdir / "table_pixelate_severity.csv", rows)
    (outdir / "table_pixelate_severity.tex").write_text(render_table(rows), encoding="utf-8")
    return 0


def build_rows(action_csv: str, features_root_test: str, router_checkpoint: str) -> list[dict]:
    action_rows = _action_rows(action_csv)
    checkpoint = _load_router(router_checkpoint)
    rows = []
    for severity in [1, 2, 3, 4, 5]:
        cell = action_rows[severity]
        features = np.load(Path(features_root_test) / f"features_pixelate_{severity}_test.npy").astype(np.float32)
        if features.shape[0] != len(cell):
            raise RuntimeError(f"severity {severity} has {features.shape[0]} feature rows but {len(cell)} action rows")
        selected = _select_router_actions(features, checkpoint)
        dncnn = _accuracy(cell, "dncnn")
        jpeg20 = _accuracy(cell, "jpeg20")
        config_a = _accuracy(cell, "config_a")
        router_correct = np.asarray(
            [int(row[f"{ACTIONS[action_index]}_correct"]) for row, action_index in zip(cell, selected)],
            dtype=np.int64,
        )
        router_accuracy = float(router_correct.mean() * 100.0)
        dncnn_correct = np.asarray([int(row["dncnn_correct"]) for row in cell], dtype=np.int64)
        gain = router_accuracy - dncnn
        lcb = paired_lcb(router_correct, dncnn_correct) * 100.0
        rows.append(
            {
                "severity": severity,
                "dncnn": dncnn,
                "jpeg20": jpeg20,
                "config_a": config_a,
                "router": router_accuracy,
                "gain": gain,
                "paired_lcb": lcb,
            }
        )
    return rows


def render_table(rows: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        r"Sev. & DnCNN & J20 & Config-A & Router & Gain & LCB \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    str(row["severity"]),
                    f"{row['dncnn']:.1f}",
                    f"{row['jpeg20']:.1f}",
                    f"{row['config_a']:.1f}",
                    f"{row['router']:.1f}",
                    _format_gain(row["gain"]),
                    _format_gain(row["paired_lcb"]),
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["severity", "dncnn", "jpeg20", "config_a", "router", "gain", "paired_lcb"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _action_rows(action_csv: str) -> dict[int, list[dict]]:
    rows = {severity: [] for severity in [1, 2, 3, 4, 5]}
    with Path(action_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "test" and row["corruption"] == "pixelate":
                rows[int(row["severity"])].append(row)
    return rows


def _accuracy(rows: list[dict], action: str) -> float:
    return float(np.mean([int(row[f"{action}_correct"]) for row in rows]) * 100.0)


def _load_router(path: str) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _select_router_actions(features: np.ndarray, checkpoint: dict) -> np.ndarray:
    import torch

    feature_indices = [FEATURE_NAMES.index(name) for name in checkpoint["feature_names"]]
    x = features[:, feature_indices].astype(np.float32)
    x = ((x - checkpoint["scaler"]["mean"]) / checkpoint["scaler"]["std"]).astype(np.float32)
    model = torch.nn.Linear(len(checkpoint["feature_names"]), len(checkpoint["actions"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.inference_mode():
        scores = torch.sigmoid(model(torch.from_numpy(x))).numpy()
    default_index = checkpoint["actions"].index(checkpoint["default_action"])
    tau = float(checkpoint["report"]["tau_selected"])
    selected_checkpoint_indices = np.asarray(
        [select_with_threshold(row, default_index, tau) for row in scores],
        dtype=np.int64,
    )
    checkpoint_actions = checkpoint["actions"]
    return np.asarray([ACTIONS.index(checkpoint_actions[index]) for index in selected_checkpoint_indices], dtype=np.int64)


def _format_gain(value: float) -> str:
    rounded = round(float(value), 1)
    if rounded == 0.0:
        return "0.0"
    return f"{rounded:+.1f}"


if __name__ == "__main__":
    raise SystemExit(main())

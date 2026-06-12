from __future__ import annotations

import argparse
import csv
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from pixelate_router.features_imagenetc import FEATURE_NAMES
from oracle_ceiling import load_rgb_float
from pixelate_router.router import select_with_threshold
from pixelate_router.imagenetc_digital import iter_image_records


CORRUPTION_LABELS = {
    "contrast": "Contrast",
    "elastic_transform": "Elastic",
    "pixelate": "Pixelate",
    "jpeg_compression": "JPEG",
}
COLORS = {
    "contrast": "#4C78A8",
    "elastic_transform": "#F2A541",
    "pixelate": "#59A14F",
    "jpeg_compression": "#B07AA1",
}
ACTION_COLORS = {
    "dncnn": "#4C78A8",
    "jpeg20": "#F2A541",
    "config_a": "#59A14F",
    "config_b": "#E15759",
    "router": "#B07AA1",
    "other": "#8C8C8C",
}
ACTION_LABELS = {
    "dncnn": "DnCNN",
    "jpeg20": "JPEG20",
    "config_a": "Config-A",
    "config_b": "Config-B",
    "router": "Router",
    "other": "Other",
}
INK = "#2F2F2F"
GRID = "#E6E6E6"
PDF_METADATA = {
    "Creator": "pixelate-action-selection",
    "Producer": "Matplotlib",
    "CreationDate": None,
    "ModDate": None,
}


def _save_pdf(fig, path: Path, **kwargs) -> None:
    fig.savefig(path, metadata=PDF_METADATA, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate ImageNet-C manuscript figures.")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--curves-source", default="data/source/figure_pixelate_severity.csv")
    parser.add_argument("--case-digital-root")
    parser.add_argument(
        "--require-case-digital-root",
        action="store_true",
        help="Fail if the ImageNet-C case image cannot be rendered from a local authorized dataset copy.",
    )
    parser.add_argument("--case-corruption", default="pixelate")
    parser.add_argument("--case-severity", type=int, default=3)
    parser.add_argument("--case-index", type=int, default=7000)
    parser.add_argument("--action-json", default="data/derived/imagenetc/action_eval_summary.json")
    parser.add_argument("--action-csv")
    parser.add_argument("--feature-json", default="data/derived/imagenetc/feature_summary.json")
    parser.add_argument("--ablation-json", default="data/derived/imagenetc/ablation_report.json")
    parser.add_argument("--features-root-test")
    parser.add_argument("--router-checkpoint")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.4,
            "legend.frameon": False,
        }
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.case_digital_root:
        make_case_mechanism_figure(args, outdir, plt)
    elif args.require_case_digital_root:
        raise FileNotFoundError("--case-digital-root is required to render the ImageNet-C case figure")
    elif not (outdir / "figure_case_mechanism.pdf").exists():
        package_case = Path("figures/imagenetc/figure_case_mechanism.pdf")
        if package_case.exists():
            import shutil

            shutil.copyfile(package_case, outdir / "figure_case_mechanism.pdf")
        else:
            print("skipped case mechanism figure because --case-digital-root was not provided")
    make_residual_routing_figure(Path(args.feature_json), Path(args.ablation_json), outdir, plt)
    curves_source = Path(args.curves_source)
    if curves_source.exists():
        make_severity_curves_from_csv(curves_source, outdir, plt)
    else:
        missing = [
            name
            for name, value in {
                "--action-csv": args.action_csv,
                "--features-root-test": args.features_root_test,
                "--router-checkpoint": args.router_checkpoint,
            }.items()
            if not value
        ]
        if missing:
            joined = ", ".join(missing)
            raise FileNotFoundError(f"missing severity source and required full rerun inputs: {joined}")
        action = _load(args.action_json)
        make_severity_curves(action, args.action_csv, args.features_root_test, args.router_checkpoint, outdir, plt)
    print(f"wrote figures to {outdir}")
    return 0


def make_case_mechanism_figure(args: argparse.Namespace, outdir: Path, plt) -> None:
    from matplotlib.patches import Rectangle

    record = _case_record(
        args.case_digital_root,
        args.case_corruption,
        args.case_severity,
        args.case_index,
    )
    image = load_rgb_float(record.path)
    proxy = _jpeg_roundtrip(image, quality=20)
    residual = np.mean(np.abs(image - proxy), axis=2)
    r_j = float(np.mean(residual))

    height, width = residual.shape
    crop = min(72, height, width)
    y0 = max(0, height // 2 - crop // 2)
    x0 = max(0, width // 2 - crop // 2)
    y1 = y0 + crop
    x1 = x0 + crop

    fig = plt.figure(figsize=(3.45, 4.75))
    gs = fig.add_gridspec(
        4,
        2,
        height_ratios=[1.0, 0.55, 0.78, 0.90],
        width_ratios=[1.0, 1.0],
        wspace=0.16,
        hspace=0.46,
    )

    image_axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    crop_axes = [fig.add_subplot(gs[1, i]) for i in range(2)]
    residual_ax = fig.add_subplot(gs[2, 0])
    bar_ax = fig.add_subplot(gs[2, 1])
    profile_ax = fig.add_subplot(gs[3, :])
    panels = [
        (image_axes[0], image, "Pixelated sample\n$x$"),
        (image_axes[1], proxy, "JPEG proxy\n$J_{20}(x)$"),
    ]
    for ax, data, title in panels:
        ax.imshow(np.clip(data, 0.0, 1.0))
        ax.add_patch(Rectangle((x0, y0), crop, crop, fill=False, edgecolor="#FFFFFF", linewidth=1.0))
        ax.set_title(title, pad=2.0)
        ax.set_xticks([])
        ax.set_yticks([])

    crops = [
        image[y0:y1, x0:x1],
        proxy[y0:y1, x0:x1],
    ]
    for ax, data in zip(crop_axes, crops):
        ax.imshow(np.clip(data, 0.0, 1.0))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#FFFFFF")
            spine.set_linewidth(0.8)

    residual_ax.imshow(residual, cmap="magma", vmin=0.0, vmax=max(float(np.percentile(residual, 99)), 1e-4))
    residual_ax.add_patch(Rectangle((x0, y0), crop, crop, fill=False, edgecolor="#FFFFFF", linewidth=1.0))
    residual_ax.set_title("JPEG residual\n$|x-J_{20}(x)|$", pad=2.0)
    residual_ax.set_xticks([])
    residual_ax.set_yticks([])

    bar_ax.bar(
        [0, 1, 2],
        [43.6, 48.8, 59.4],
        color=[ACTION_COLORS["dncnn"], ACTION_COLORS["router"], INK],
        width=0.60,
    )
    bar_ax.set_xticks([0, 1, 2])
    bar_ax.set_xticklabels(["D", "R", "O"])
    bar_ax.set_ylim(35, 62)
    bar_ax.set_yticks([40, 50, 60])
    bar_ax.set_ylabel("Accuracy (%)", labelpad=1)
    bar_ax.set_title("Pixelate test", pad=2.0)
    bar_ax.grid(axis="y", color=GRID, lw=0.45)

    row = crop // 2
    x_axis = np.arange(crop)
    profile = crops[0][row].mean(axis=1)
    proxy_profile = crops[1][row].mean(axis=1)
    profile_ax.plot(x_axis, profile, color=INK, lw=1.35, label="$x$")
    profile_ax.plot(x_axis, proxy_profile, color=ACTION_COLORS["jpeg20"], lw=1.35, label="$J_{20}(x)$")
    profile_ax.fill_between(
        x_axis,
        profile,
        proxy_profile,
        color=ACTION_COLORS["router"],
        alpha=0.25,
        linewidth=0.0,
        label="$|x-J_{20}(x)|$",
    )
    profile_ax.set_title("Local block-edge proxy", pad=2.0)
    profile_ax.set_xlabel("Crop column")
    profile_ax.set_ylabel("Mean luminance")
    profile_ax.set_xlim(0, crop - 1)
    profile_ax.set_ylim(-0.02, 1.02)
    profile_ax.grid(axis="y", color=GRID, lw=0.45)
    profile_ax.text(
        0.03,
        0.10,
        f"$r_J(x)$ = {r_j:.3f}",
        transform=profile_ax.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#DDDDDD", "boxstyle": "square,pad=0.20"},
    )
    handles, labels = profile_ax.get_legend_handles_labels()
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.06, top=0.90, wspace=0.18, hspace=0.52)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        handlelength=1.15,
        columnspacing=0.55,
        borderaxespad=0.0,
    )
    _save_pdf(fig, outdir / "figure_case_mechanism.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def make_residual_routing_figure(feature_json: Path, ablation_json: Path, outdir: Path, plt) -> None:
    features = _load(feature_json)
    ablation = _load(ablation_json)
    feature_cells = features["aggregate"]["by_split"]["validation"]
    policy = ablation["policies"]["logistic_all"]["test"]["per_corruption"]

    fig = plt.figure(figsize=(3.45, 4.45))
    gs = fig.add_gridspec(4, 1, height_ratios=[0.18, 1.0, 0.30, 1.08], hspace=0.34)
    residual_legend_ax = fig.add_subplot(gs[0, 0])
    residual_ax = fig.add_subplot(gs[1, 0])
    action_legend_ax = fig.add_subplot(gs[2, 0])
    routing_ax = fig.add_subplot(gs[3, 0])
    residual_legend_ax.set_axis_off()
    action_legend_ax.set_axis_off()
    severities = [1, 2, 3, 4, 5]
    for corr, label in CORRUPTION_LABELS.items():
        values = [feature_cells[corr][str(severity)]["hfer_jpeg20_residual_mean"] for severity in severities]
        residual_ax.plot(severities, values, marker="o", ms=3.0, lw=1.35, color=COLORS[corr], label=label)
    residual_ax.set_ylabel("$r_J(x)$ mean")
    residual_ax.set_xticks(severities)
    residual_ax.set_title("JPEG residual is highest for pixelate")
    residual_ax.grid(axis="y", color=GRID, lw=0.45)
    handles, labels = residual_ax.get_legend_handles_labels()
    residual_legend_ax.legend(
        handles,
        labels,
        ncol=4,
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        handlelength=1.0,
        columnspacing=0.5,
        borderaxespad=0.0,
    )

    x = np.arange(len(CORRUPTION_LABELS))
    bottom = np.zeros(len(CORRUPTION_LABELS), dtype=np.float64)
    action_order = ["dncnn", "jpeg20", "config_a", "config_b", "other"]
    for action in action_order:
        shares = []
        for corr in CORRUPTION_LABELS:
            dist = policy[corr]["action_distribution"]
            if action == "other":
                share = sum(value["share"] for key, value in dist.items() if key not in {"dncnn", "jpeg20", "config_a", "config_b"})
            else:
                share = dist[action]["share"]
            shares.append(100.0 * share)
        routing_ax.bar(x, shares, bottom=bottom, color=ACTION_COLORS[action], width=0.68, label=ACTION_LABELS[action])
        bottom += np.asarray(shares)
    routing_ax.set_xticks(x)
    routing_ax.set_xticklabels([CORRUPTION_LABELS[corr] for corr in CORRUPTION_LABELS], rotation=18, ha="right")
    routing_ax.set_ylabel("Selected action (%)")
    routing_ax.set_ylim(0, 100)
    routing_ax.set_title("Routing concentrates on pixelate", pad=2.0)
    routing_ax.grid(axis="y", color=GRID, lw=0.45)
    handles, labels = routing_ax.get_legend_handles_labels()
    action_legend_ax.legend(
        handles,
        labels,
        ncol=3,
        loc="center",
        bbox_to_anchor=(0.50, 0.5),
        handlelength=0.9,
        columnspacing=0.55,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.08, top=0.97, hspace=0.32)
    _save_pdf(fig, outdir / "figure_residual_routing.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def make_severity_curves(action: dict, action_csv: str, features_root: str, router_checkpoint: str, outdir: Path, plt) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.55), sharex=True)
    legend_handles, legend_labels = _plot_severity_panel(
        axes[0],
        action["summaries"]["test"]["pixelate"],
        severities=[1, 2, 3, 4, 5],
        router_curve=_router_curve("pixelate", action_csv, features_root, router_checkpoint),
        title="Pixelate",
    )
    _plot_severity_panel(
        axes[1],
        action["summaries"]["test"]["elastic_transform"],
        severities=[1, 2, 3, 4, 5],
        router_curve=_router_curve("elastic_transform", action_csv, features_root, router_checkpoint),
        title="Elastic control",
    )
    axes[0].set_ylabel("Top-1 accuracy (%)")
    axes[1].set_ylabel("Top-1 accuracy (%)")
    axes[1].set_xlabel("Severity")
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.995),
        handlelength=1.1,
        columnspacing=0.7,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), pad=0.45, h_pad=0.55)
    _save_pdf(fig, outdir / "figure_pixelate_severity.pdf", bbox_inches="tight")
    plt.close(fig)


def make_severity_curves_from_csv(source_csv: Path, outdir: Path, plt) -> None:
    rows = _read_csv_rows(source_csv)
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.55), sharex=True)
    legend_handles, legend_labels = _plot_severity_rows(axes[0], rows, "pixelate", "Pixelate")
    _plot_severity_rows(axes[1], rows, "elastic_transform", "Elastic control")
    axes[0].set_ylabel("Top-1 accuracy (%)")
    axes[1].set_ylabel("Top-1 accuracy (%)")
    axes[1].set_xlabel("Severity")
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.995),
        handlelength=1.1,
        columnspacing=0.7,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), pad=0.45, h_pad=0.55)
    _save_pdf(fig, outdir / "figure_pixelate_severity.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_severity_panel(
    ax,
    summaries: dict,
    severities: list[int],
    router_curve: list[float],
    title: str,
    compact: bool = False,
):
    dncnn = [summaries[str(severity)]["dncnn_accuracy"] for severity in severities]
    jpeg20 = [summaries[str(severity)]["jpeg20_accuracy"] for severity in severities]
    config_a = [summaries[str(severity)]["config_a_accuracy"] for severity in severities]
    curves = [
        ("dncnn", dncnn, "o"),
        ("jpeg20", jpeg20, "s"),
        ("config_a", config_a, "^"),
        ("router", router_curve, "D"),
    ]
    for action, values, marker in curves:
        width = 1.55 if action == "router" else 1.35
        ax.plot(
            severities,
            values,
            marker=marker,
            ms=2.6 if compact else 3.2,
            lw=1.0 if compact else width,
            color=ACTION_COLORS[action],
            label=ACTION_LABELS[action],
            alpha=0.85 if compact else 1.0,
        )
    ax.set_title(title, pad=2.0)
    ax.set_xticks(severities)
    if compact:
        ax.set_yticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=6, pad=1)
    ax.grid(axis="y", color=GRID, lw=0.45)
    return ax.get_legend_handles_labels()


def _plot_severity_rows(ax, rows: list[dict], corr: str, title: str, compact: bool = False):
    subset = [row for row in rows if row["corruption"] == corr]
    for action in ["dncnn", "jpeg20", "config_a", "router"]:
        curve = [row for row in subset if row["action"] == action]
        curve.sort(key=lambda row: int(row["severity"]))
        severities = [int(row["severity"]) for row in curve]
        accuracy = [float(row["accuracy"]) for row in curve]
        marker = {"dncnn": "o", "jpeg20": "s", "config_a": "^", "router": "D"}[action]
        width = 1.55 if action == "router" else 1.35
        ax.plot(
            severities,
            accuracy,
            marker=marker,
            ms=2.6 if compact else 3.2,
            lw=1.0 if compact else width,
            color=ACTION_COLORS[action],
            label=ACTION_LABELS[action],
            alpha=0.85 if compact else 1.0,
        )
    ax.set_title(title, pad=2.0)
    ax.set_xticks([1, 2, 3, 4, 5])
    if compact:
        ax.set_yticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=6, pad=1)
    ax.grid(axis="y", color=GRID, lw=0.45)
    return ax.get_legend_handles_labels()


def _case_record(
    digital_root: str | None,
    corruption: str,
    severity: int,
    image_index: int,
):
    if not digital_root:
        raise FileNotFoundError("--case-digital-root is required to render the ImageNet-C case figure")
    records = iter_image_records(digital_root, corruption, severity, indices=[image_index])
    if not records:
        raise FileNotFoundError(
            f"no case image found for corruption={corruption}, severity={severity}, index={image_index}"
        )
    return records[0]


def _jpeg_roundtrip(image: np.ndarray, quality: int = 20) -> np.ndarray:
    from PIL import Image

    array = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(array, mode="RGB")
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.float32) / 255.0


def _router_curve(corr: str, action_csv: str, features_root: str, router_checkpoint: str) -> list[float]:
    import torch

    checkpoint = torch.load(router_checkpoint, map_location="cpu", weights_only=False)
    model = torch.nn.Linear(len(checkpoint["feature_names"]), len(checkpoint["actions"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    feature_indices = [FEATURE_NAMES.index(name) for name in checkpoint["feature_names"]]
    mean = checkpoint["scaler"]["mean"]
    std = checkpoint["scaler"]["std"]
    tau = float(checkpoint["report"]["tau_selected"])
    default_index = checkpoint["actions"].index(checkpoint["default_action"])
    rows = _action_rows(action_csv, corr)
    result = []
    for severity in range(1, 6):
        matrix = np.load(Path(features_root) / f"features_{corr}_{severity}_test.npy").astype(np.float32)
        x = ((matrix[:, feature_indices] - mean) / std).astype(np.float32)
        with torch.inference_mode():
            scores = torch.sigmoid(model(torch.from_numpy(x))).numpy()
        selected = np.array([select_with_threshold(row, default_index, tau) for row in scores], dtype=np.int64)
        cell_rows = rows[severity]
        correct = [
            int(cell_rows[index][f"{checkpoint['actions'][action_index]}_correct"])
            for index, action_index in enumerate(selected)
        ]
        result.append(float(np.mean(correct) * 100.0))
    return result


def _action_rows(action_csv: str, corr: str) -> dict[int, list[dict]]:
    rows = {severity: [] for severity in range(1, 6)}
    with Path(action_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "test" and row["corruption"] == corr:
                rows[int(row["severity"])].append(row)
    return rows


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _read_csv_rows(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    raise SystemExit(main())

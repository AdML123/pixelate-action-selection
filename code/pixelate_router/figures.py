from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def _save_publication_figure(fig, prefix: Path, dpi: int = 600) -> None:
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def export_radial_spectra(spectra: pd.DataFrame, prefix: Path) -> None:
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    colors = ["#3B6EA8", "#D08C2F", "#4E8F5D", "#8B5A9F"]
    for idx, (name, group) in enumerate(spectra.groupby("corruption", sort=False)):
        group = group.sort_values("radius")
        ax.plot(
            group["radius"],
            group["energy"],
            linewidth=1.4,
            color=colors[idx % len(colors)],
            label=str(name),
        )
    ax.axvline(0.25, color="#666666", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Normalized spatial frequency")
    ax.set_ylabel("Radial energy")
    ax.legend(loc="best")
    _save_publication_figure(fig, Path(prefix))


def export_pipeline_diagram(prefix: Path) -> None:
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(4.0, 2.2))
    ax.set_axis_off()

    def box(x: float, y: float, text: str) -> None:
        rect = Rectangle((x, y), 0.86, 0.34, linewidth=0.8, edgecolor="#333333", facecolor="#F4F6F8")
        ax.add_patch(rect)
        ax.text(x + 0.43, y + 0.17, text, ha="center", va="center")

    def arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.8,
                color="#333333",
            )
        )

    box(0.05, 0.68, "Input")
    box(1.15, 1.02, "FFT gate")
    box(2.25, 1.02, "Config-A\nC -> D -> F")
    box(2.25, 0.34, "Config-B\nD -> C -> F")
    box(3.35, 0.68, "Prediction")
    arrow(0.91, 0.85, 1.15, 1.15)
    arrow(0.91, 0.85, 2.25, 0.51)
    arrow(2.01, 1.19, 2.25, 1.19)
    arrow(3.11, 1.19, 3.35, 0.85)
    arrow(3.11, 0.51, 3.35, 0.85)
    ax.set_xlim(0.0, 4.25)
    ax.set_ylim(0.2, 1.45)
    _save_publication_figure(fig, Path(prefix))


def export_severity_curves(curves: pd.DataFrame, prefix: Path) -> None:
    _configure_matplotlib()
    corruptions = list(curves["corruption"].drop_duplicates())
    ncols = min(2, max(1, len(corruptions)))
    nrows = (len(corruptions) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.6 * ncols, 2.1 * nrows), squeeze=False)
    colors = {"config_a": "#3B6EA8", "config_b": "#D08C2F", "router": "#4E8F5D"}
    labels = {"config_a": "Config-A", "config_b": "Config-B", "router": "router"}
    for ax, corruption in zip(axes.flat, corruptions):
        subset = curves[curves["corruption"] == corruption]
        for config, group in subset.groupby("config", sort=False):
            group = group.sort_values("severity")
            ax.plot(
                group["severity"],
                group["accuracy"],
                marker="o",
                markersize=3,
                linewidth=1.2,
                color=colors.get(config, "#555555"),
                label=labels.get(config, config),
            )
        ax.set_title(str(corruption), fontsize=7, fontweight="bold")
        ax.set_xlabel("Severity")
        ax.set_ylabel("Top-1 accuracy (%)")
        ax.set_xticks(sorted(subset["severity"].unique()))
    for ax in axes.flat[len(corruptions) :]:
        ax.set_axis_off()
    axes.flat[0].legend(loc="best")
    _save_publication_figure(fig, Path(prefix))

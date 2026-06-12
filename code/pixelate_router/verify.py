from pathlib import Path

REQUIRED_RESULT_FILES = (
    "oracle_ceiling_s3_2000.json",
    "action_eval_summary.json",
    "feature_summary.json",
    "validation_report.json",
    "ablation_report.json",
    "detector_baseline_report.json",
    "loco_report.json",
    "timing_report.json",
)

REQUIRED_TABLE_FILES = (
    "table_pixelate_primary.tex",
    "table_main.tex",
    "table_pixelate_severity.tex",
    "table_ablation.tex",
)

REQUIRED_FIGURE_FILES = (
    "figure_case_mechanism.pdf",
    "figure_residual_routing.pdf",
    "figure_pixelate_severity.pdf",
)


def verify_required_results(results_dir: Path) -> None:
    results_dir = Path(results_dir)
    missing = [name for name in REQUIRED_RESULT_FILES if not (results_dir / name).exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"missing required result artifacts: {joined}")


def verify_required_outputs(tables_dir: Path, figures_dir: Path) -> None:
    tables_dir = Path(tables_dir)
    figures_dir = Path(figures_dir)
    missing_tables = [name for name in REQUIRED_TABLE_FILES if not (tables_dir / name).exists()]
    missing_figures = [name for name in REQUIRED_FIGURE_FILES if not (figures_dir / name).exists()]
    if missing_tables or missing_figures:
        missing = [*(f"table:{name}" for name in missing_tables), *(f"figure:{name}" for name in missing_figures)]
        raise FileNotFoundError(f"missing generated manuscript artifacts: {', '.join(missing)}")

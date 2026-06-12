import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from pixelate_router.config import ResourceConfig


def _command(script: str, *args: str) -> list[str]:
    return [sys.executable, str(Path("code") / script), *args]


def build_commands(config: ResourceConfig, limit_per_split: int | None = None) -> list[list[str]]:
    shared_model_args = [
        "--digital-root",
        str(config.digital_root),
        "--dncnn-checkpoint",
        str(config.dncnn_checkpoint),
        "--resnet50-checkpoint",
        str(config.resnet50_checkpoint),
        "--kair-root",
        str(config.kair_root),
    ]
    action_dir = config.root / "data" / "action_eval"
    feature_dir = config.root / "data" / "features"
    test_feature_dir = config.root / "data" / "features_test"
    router_dir = config.root / "data" / "router"
    action_csv = action_dir / "action_eval.csv"
    router_checkpoint = router_dir / "router_all.pt"

    optional_limit = [] if limit_per_split is None else ["--limit-per-split", str(limit_per_split)]

    return [
        _command(
            "oracle_ceiling.py",
            *shared_model_args,
            "--outdir",
            str(config.derived_dir),
            "--severity",
            "3",
            "--limit-images",
            "2000",
        ),
        _command(
            "full_action_eval.py",
            *shared_model_args,
            "--outdir",
            str(action_dir),
            *optional_limit,
        ),
        _command(
            "extract_imagenetc_features.py",
            *shared_model_args,
            "--action-csv",
            str(action_csv),
            "--outdir",
            str(feature_dir),
            "--splits",
            "search,validation",
            *optional_limit,
        ),
        _command(
            "extract_imagenetc_features.py",
            *shared_model_args,
            "--action-csv",
            str(action_csv),
            "--outdir",
            str(test_feature_dir),
            "--splits",
            "test",
            *optional_limit,
        ),
        _command("train_router.py", "--features-root", str(feature_dir), "--action-csv", str(action_csv), "--outdir", str(router_dir)),
        _command(
            "evaluate_router.py",
            "--features-root",
            str(feature_dir),
            "--test-features-root",
            str(test_feature_dir),
            "--action-csv",
            str(action_csv),
            "--router-dir",
            str(router_dir),
            "--outdir",
            str(config.derived_dir),
        ),
        _command(
            "evaluate_corruption_detector.py",
            "--features-root",
            str(feature_dir),
            "--test-features-root",
            str(test_feature_dir),
            "--action-csv",
            str(action_csv),
            "--outdir",
            str(config.derived_dir),
        ),
        _command(
            "make_pixelate_severity_table.py",
            "--action-csv",
            str(action_csv),
            "--features-root-test",
            str(test_feature_dir),
            "--router-checkpoint",
            str(router_checkpoint),
            "--outdir",
            str(config.root / "data" / "source"),
        ),
        _command("run_loco.py", "--features-root", str(feature_dir), "--action-csv", str(action_csv), "--outdir", str(config.derived_dir)),
        _command(
            "measure_latency.py",
            *shared_model_args,
            "--router-checkpoint",
            str(router_checkpoint),
            "--outdir",
            str(config.derived_dir),
        ),
        _command("make_tables.py", "--outdir", str(config.table_dir)),
        _command(
            "make_figures.py",
            "--outdir",
            str(config.figure_dir),
            "--case-digital-root",
            str(config.digital_root),
            "--case-severity",
            "3",
            "--case-index",
            "7000",
        ),
        _command(
            "verify_results.py",
            "--results",
            str(config.derived_dir),
            "--tables",
            str(config.table_dir),
            "--figures",
            str(config.figure_dir),
        ),
    ]


def load_config(path: Path) -> ResourceConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ResourceConfig.from_dict(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full local ImageNet-C reproduction pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without executing them.")
    parser.add_argument("--limit-per-split", type=int, default=None, help="Optional small local smoke limit.")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    commands = build_commands(load_config(config_path), limit_per_split=args.limit_per_split)
    for command in commands:
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

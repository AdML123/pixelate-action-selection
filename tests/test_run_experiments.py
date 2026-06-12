import subprocess
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_run_experiments_dry_run_lists_full_pipeline(tmp_path):
    config = tmp_path / "local.yaml"
    config.write_text(
        "\n".join(
            [
                f"root: {tmp_path.as_posix()}",
                "digital_root: data/digital",
                "kair_root: external/KAIR",
                "dncnn_checkpoint: model_zoo/dncnn_color_blind.pth",
                "resnet50_checkpoint: model_zoo/resnet50-11ad3fa6.pth",
                "derived_dir: data/derived/imagenetc",
                "table_dir: tables/imagenetc",
                "figure_dir: figures/imagenetc",
                "",
            ]
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, "code/run_experiments.py", "--config", str(config), "--dry-run", "--limit-per-split", "2"],
        cwd=PACKAGE_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "oracle_ceiling.py" in completed.stdout
    assert "full_action_eval.py" in completed.stdout
    assert "extract_imagenetc_features.py" in completed.stdout
    assert "train_router.py" in completed.stdout
    assert "evaluate_router.py" in completed.stdout
    assert "make_tables.py" in completed.stdout
    assert "make_figures.py" in completed.stdout
    assert "verify_results.py" in completed.stdout
    assert "implemented " + "after" not in completed.stdout.lower()


def test_example_config_supports_dry_run():
    completed = subprocess.run(
        [sys.executable, "code/run_experiments.py", "--config", "configs/local.example.yaml", "--dry-run"],
        cwd=PACKAGE_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "code" in completed.stdout
    assert "run_experiments.py" not in completed.stdout
    assert "verify_results.py" in completed.stdout

import sys
import subprocess
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from paper34.figures import (
    export_pipeline_diagram,
    export_radial_spectra,
    export_severity_curves,
)


def _assert_exports_exist(prefix: Path):
    assert prefix.with_suffix(".svg").exists()
    assert prefix.with_suffix(".pdf").exists()
    assert prefix.with_suffix(".png").exists()


def test_export_radial_spectra_creates_publication_files(tmp_path):
    spectra = pd.DataFrame(
        {
            "radius": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            "energy": [0.1, 0.3, 0.6, 0.6, 0.3, 0.1],
            "corruption": ["Gaussian noise"] * 3 + ["Fog"] * 3,
        }
    )

    prefix = tmp_path / "figure1_radial_spectra"
    export_radial_spectra(spectra, prefix)

    _assert_exports_exist(prefix)


def test_export_pipeline_diagram_creates_publication_files(tmp_path):
    prefix = tmp_path / "figure2_pipeline"

    export_pipeline_diagram(prefix)

    _assert_exports_exist(prefix)


def test_export_severity_curves_creates_publication_files(tmp_path):
    curves = pd.DataFrame(
        {
            "corruption": ["Gaussian noise"] * 6,
            "severity": [1, 2, 3, 1, 2, 3],
            "config": ["config_a", "config_a", "config_a", "config_b", "config_b", "config_b"],
            "accuracy": [62.0, 66.0, 70.0, 68.0, 65.0, 61.0],
        }
    )

    prefix = tmp_path / "figure3_severity_curves"
    export_severity_curves(curves, prefix)

    _assert_exports_exist(prefix)


def test_imagenetc_figures_generate_without_real_case_data(tmp_path):
    outdir = tmp_path / "figures"

    completed = subprocess.run(
        [
            sys.executable,
            "code/make_figures.py",
            "--outdir",
            str(outdir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert (outdir / "figure_case_mechanism.pdf").exists()
    assert (outdir / "figure_residual_routing.pdf").exists()
    assert (outdir / "figure_pixelate_severity.pdf").exists()


def test_imagenetc_figures_can_require_real_case_data(tmp_path):
    outdir = tmp_path / "figures"

    completed = subprocess.run(
        [
            sys.executable,
            "code/make_figures.py",
            "--outdir",
            str(outdir),
            "--require-case-digital-root",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "--case-digital-root" in completed.stderr


def test_imagenetc_figures_generate_redesigned_outputs(tmp_path):
    local_root = Path(__file__).resolve().parents[4] / "data" / "digital"
    if not local_root.exists():
        pytest.skip("local ImageNet-C digital subset is not available")
    outdir = tmp_path / "figures"

    completed = subprocess.run(
        [
            sys.executable,
            "code/make_figures.py",
            "--outdir",
            str(outdir),
            "--case-digital-root",
            str(local_root),
            "--case-severity",
            "3",
            "--case-index",
            "7000",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    )

    assert completed.returncode == 0
    assert (outdir / "figure_case_mechanism.pdf").exists()
    assert (outdir / "figure_residual_routing.pdf").exists()
    assert (outdir / "figure_pixelate_severity.pdf").exists()

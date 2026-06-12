import sys
import subprocess
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

EXPECTED_MANUSCRIPT_FIGURES = {
    "figure_case_mechanism.pdf",
    "figure_residual_routing.pdf",
    "figure_pixelate_severity.pdf",
}


def _generated_pdfs(outdir: Path) -> set[str]:
    return {path.name for path in outdir.glob("*.pdf")}


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
    assert _generated_pdfs(outdir) == EXPECTED_MANUSCRIPT_FIGURES


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
    assert _generated_pdfs(outdir) == EXPECTED_MANUSCRIPT_FIGURES


def test_figure_generator_has_no_legacy_pipeline_or_radial_exports():
    source = (Path(__file__).resolve().parents[1] / "code" / "make_imagenetc_figures.py").read_text(
        encoding="utf-8"
    )

    forbidden = [
        "figure_" + "pipeline.pdf",
        "figure_" + "radial_spectra.pdf",
        "def make_" + "pipeline_figure",
        "def make_" + "radial_spectra",
        "def " + "radial_spectrum",
        "inset" + "_axes",
    ]
    for phrase in forbidden:
        assert phrase not in source

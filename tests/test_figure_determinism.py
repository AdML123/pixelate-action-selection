import hashlib
import subprocess
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_aggregate_pdf_figures_are_byte_stable(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"

    for outdir in [first, second]:
        subprocess.run(
            [sys.executable, "code/make_figures.py", "--outdir", str(outdir)],
            cwd=PACKAGE_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    for name in ["figure_residual_routing.pdf", "figure_pixelate_severity.pdf"]:
        assert _hash(first / name) == _hash(second / name)

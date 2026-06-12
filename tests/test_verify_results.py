import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from pixelate_router.verify import REQUIRED_FIGURE_FILES, REQUIRED_RESULT_FILES, REQUIRED_TABLE_FILES, verify_required_outputs, verify_required_results


def test_verify_required_results_accepts_present_files(tmp_path):
    for name in REQUIRED_RESULT_FILES:
        (tmp_path / name).write_text("{}\n", encoding="utf-8")

    verify_required_results(tmp_path)


def test_verify_required_results_rejects_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing required result artifacts"):
        verify_required_results(tmp_path)


def test_verify_required_outputs_accepts_present_files(tmp_path):
    tables = tmp_path / "tables"
    figures = tmp_path / "figures"
    tables.mkdir()
    figures.mkdir()
    for name in REQUIRED_TABLE_FILES:
        (tables / name).write_text("% table\n", encoding="utf-8")
    for name in REQUIRED_FIGURE_FILES:
        (figures / name).write_bytes(b"%PDF-1.7\n")

    verify_required_outputs(tables, figures)

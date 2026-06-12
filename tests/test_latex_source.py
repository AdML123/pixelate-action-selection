import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from pixelate_router.draft_audit import find_unresolved_markers


def _optional_latex_source() -> str | None:
    path = Path("latex_source/main.tex")
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def test_latex_has_no_template_warning_text():
    text = _optional_latex_source()
    if text is None:
        return

    assert "This document is a model and instructions" not in text
    assert "Failure to remove the template text" not in text


def test_latex_has_no_unresolved_manuscript_markers():
    text = _optional_latex_source()
    if text is None:
        return

    assert find_unresolved_markers(text) == []

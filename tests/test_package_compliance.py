from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _text_files():
    ignored_dirs = {".git", "__pycache__", ".pytest_cache"}
    ignored_suffixes = {".pdf", ".png", ".pyc", ".zip", ".pt", ".pth", ".npy"}
    ignored_names = {"MANIFEST.sha256"}
    for path in PACKAGE_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if ignored_dirs.intersection(path.parts):
            continue
        if path.name in ignored_names or path.name in {Path(__file__).name, "test_draft_audit.py"}:
            continue
        if path.suffix.lower() in ignored_suffixes:
            continue
        yield path


def test_public_package_has_no_internal_or_legacy_identifiers():
    forbidden = [
        "paper" + "34",
        "choose_" + "adapt" + "ive_config",
        "summarize_" + "accuracy",
        "band" + "it",
    ]
    offenders = []
    for path in _text_files():
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for term in forbidden:
            if term in text:
                offenders.append(f"{path.relative_to(PACKAGE_ROOT)}: {term}")

    assert offenders == []

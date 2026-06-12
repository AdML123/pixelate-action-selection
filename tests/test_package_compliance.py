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


def test_public_package_has_no_runtime_cache_artifacts():
    forbidden_parts = {"__pycache__", ".pytest_cache"}
    forbidden_suffixes = {".pyc", ".pyo", ".aux", ".log", ".out", ".toc", ".fls", ".fdb_latexmk"}
    offenders = []
    manifest = PACKAGE_ROOT / "MANIFEST.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        relative = Path(line.split("  ", 1)[1])
        if forbidden_parts.intersection(relative.parts) or relative.suffix.lower() in forbidden_suffixes:
            offenders.append(str(relative))

    assert offenders == []


def test_derived_artifacts_do_not_expose_private_work_paths():
    slash = chr(92)
    forbidden = ["local_" + "review", "D:" + slash, "C:" + slash, "/" + "Users/", slash + "Users" + slash]
    offenders = []
    for path in (PACKAGE_ROOT / "data" / "derived").rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for term in forbidden:
            if term in text:
                offenders.append(f"{path.relative_to(PACKAGE_ROOT)}: {term}")

    assert offenders == []


def test_full_rerun_entrypoint_is_executable():
    text = (PACKAGE_ROOT / "code" / "run_experiments.py").read_text(encoding="utf-8").lower()
    forbidden = ["implemented " + "after", "smoke checks " + "pass", "place" + "holder", "to" + "do", "st" + "ub"]

    for phrase in forbidden:
        assert phrase not in text

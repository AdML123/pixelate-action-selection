import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from pixelate_router.security import scan_text
from secret_scan import iter_text_files


def test_secret_scan_flags_local_absolute_path():
    slash = chr(92)
    findings = scan_text("data lives at " + "C:" + slash + "Users" + slash + "name" + slash + "secret")

    assert findings


def test_secret_scan_allows_relative_path():
    findings = scan_text("data lives at data/derived/table2.csv")

    assert findings == []


def test_secret_scan_skips_git_directory(tmp_path):
    git_object = tmp_path / ".git" / "objects" / "aa"
    git_object.mkdir(parents=True)
    slash = chr(92)
    (git_object / "bb").write_text("packed " + "D:" + slash + "local" + slash + "path", encoding="utf-8")
    visible = tmp_path / "README.md"
    visible.write_text("relative data/derived/table.csv", encoding="utf-8")

    files = list(iter_text_files(tmp_path))

    assert files == [visible]


def test_secret_scan_skips_binary_feature_matrices(tmp_path):
    matrix = tmp_path / "features.npy"
    slash = chr(92)
    matrix.write_bytes(("binary-ish D:" + slash + "local").encode("utf-8"))
    visible = tmp_path / "README.md"
    visible.write_text("relative data/features/features.npy", encoding="utf-8")

    files = list(iter_text_files(tmp_path))

    assert files == [visible]

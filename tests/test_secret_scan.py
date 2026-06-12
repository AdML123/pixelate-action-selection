import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from paper34.security import scan_text
from secret_scan import iter_text_files


def test_secret_scan_flags_local_absolute_path():
    findings = scan_text("data lives at C:\\Users\\name\\secret")

    assert findings


def test_secret_scan_allows_relative_path():
    findings = scan_text("data lives at data/derived/table2.csv")

    assert findings == []


def test_secret_scan_skips_git_directory(tmp_path):
    git_object = tmp_path / ".git" / "objects" / "aa"
    git_object.mkdir(parents=True)
    (git_object / "bb").write_text("packed D:\\local\\path", encoding="utf-8")
    visible = tmp_path / "README.md"
    visible.write_text("relative data/derived/table.csv", encoding="utf-8")

    files = list(iter_text_files(tmp_path))

    assert files == [visible]

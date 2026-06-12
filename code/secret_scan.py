import argparse
from pathlib import Path

from pixelate_router.security import scan_text


def iter_text_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        parts = set(path.parts)
        if ".git" in parts or "__pycache__" in parts or "tests" in parts:
            continue
        if path.name == "security.py":
            continue
        if path.suffix.lower() in {".png", ".pdf", ".pt", ".pth", ".tar", ".gz", ".zip", ".npy", ".npz"}:
            continue
        yield path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    root = Path(args.root)
    findings = []
    for path in iter_text_files(root):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in scan_text(text):
            findings.append(f"{path}: {pattern}")
    if findings:
        for finding in findings:
            print(finding)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import re

PATTERNS = [
    re.compile(r"ghp_[A-Za-z0-9_]+"),
    re.compile(r"github_pat_[A-Za-z0-9_]+"),
    re.compile(r"BEGIN .*PRIVATE KEY"),
    re.compile(r"[A-Z]:\\"),
    re.compile(r"/Users/"),
    re.compile(r"password\s*[:=]", re.IGNORECASE),
]


def scan_text(text: str) -> list[str]:
    return [pattern.pattern for pattern in PATTERNS if pattern.search(text)]

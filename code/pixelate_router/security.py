import re

PATTERNS = [
    re.compile(r"ghp" + r"_[A-Za-z0-9_]+"),
    re.compile(r"github" + r"_pat_[A-Za-z0-9_]+"),
    re.compile(r"BEGIN .*PRIVATE KEY"),
    re.compile(r"[A-Z]:\\"),
    re.compile(r"/" + r"Users/"),
    re.compile(r"pass" + r"word\s*[:=]", re.IGNORECASE),
]


def scan_text(text: str) -> list[str]:
    return [pattern.pattern for pattern in PATTERNS if pattern.search(text)]

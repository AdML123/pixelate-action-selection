from __future__ import annotations

import gzip
from pathlib import Path
from typing import IO


def open_text(path: str | Path, *, encoding: str = "utf-8", newline: str | None = None) -> IO[str]:
    resolved = Path(path)
    if resolved.suffix == ".gz":
        return gzip.open(resolved, mode="rt", encoding=encoding, newline=newline)
    return resolved.open(mode="r", encoding=encoding, newline=newline)

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_VAL_RE = re.compile(r"ILSVRC2012_val_(\d{8})\.(?:JPEG|jpg|jpeg)$", re.IGNORECASE)


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    corruption: str
    severity: int
    image_index: int
    wnid: str
    label: int


def digital_corruptions() -> list[str]:
    return ["contrast", "elastic_transform", "pixelate", "jpeg_compression"]


def select_indices(count: int, start: int = 0) -> list[int]:
    if count < 1:
        raise ValueError("count must be positive")
    if start < 0:
        raise ValueError("start must be non-negative")
    return list(range(start, start + count))


def parse_val_index(path: Path | str) -> int:
    match = _VAL_RE.search(Path(path).name)
    if not match:
        raise ValueError(f"not an ImageNet validation filename: {path}")
    index = int(match.group(1)) - 1
    if index < 0:
        raise ValueError(f"ImageNet validation index must be positive: {path}")
    return index


def wnid_to_class_index(wnids: Iterable[str]) -> dict[str, int]:
    unique = sorted(set(wnids))
    return {wnid: index for index, wnid in enumerate(unique)}


def discover_wnids(root: Path | str, corruption: str, severity: int) -> list[str]:
    cell = _cell_dir(root, corruption, severity)
    if not cell.exists():
        raise FileNotFoundError(f"ImageNet-C digital cell not found: {cell}")
    return sorted(path.name for path in cell.iterdir() if path.is_dir())


def iter_image_records(
    root: Path | str,
    corruption: str,
    severity: int,
    indices: Iterable[int] | None = None,
) -> list[ImageRecord]:
    cell = _cell_dir(root, corruption, severity)
    wnids = discover_wnids(root, corruption, severity)
    class_index = wnid_to_class_index(wnids)
    selected = None if indices is None else {int(index) for index in indices}

    records = []
    for path in cell.glob("*/*.JPEG"):
        image_index = parse_val_index(path)
        if selected is not None and image_index not in selected:
            continue
        wnid = path.parent.name
        records.append(
            ImageRecord(
                path=path,
                corruption=corruption,
                severity=severity,
                image_index=image_index,
                wnid=wnid,
                label=class_index[wnid],
            )
        )

    records.sort(key=lambda record: record.image_index)
    if selected is not None:
        found = {record.image_index for record in records}
        missing = sorted(selected - found)
        if missing:
            preview = ", ".join(str(index) for index in missing[:10])
            raise FileNotFoundError(f"missing requested ImageNet indices under {cell}: {preview}")
    return records


def _cell_dir(root: Path | str, corruption: str, severity: int) -> Path:
    if severity < 1 or severity > 5:
        raise ValueError("severity must be in [1, 5]")
    return Path(root) / corruption / str(severity)

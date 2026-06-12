import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from pixelate_router.imagenetc_digital import (
    digital_corruptions,
    iter_image_records,
    parse_val_index,
    select_indices,
    wnid_to_class_index,
)


def test_digital_corruptions_are_manuscript_order():
    assert digital_corruptions() == ["contrast", "elastic_transform", "pixelate", "jpeg_compression"]


def test_select_indices_is_deterministic_prefix():
    assert select_indices(count=5, start=10) == [10, 11, 12, 13, 14]


def test_parse_val_index_uses_zero_based_imagenet_val_index():
    assert parse_val_index(Path("ILSVRC2012_val_00020106.JPEG")) == 20105


def test_wnid_to_class_index_sorts_like_imagefolder():
    mapping = wnid_to_class_index(["n01443537", "n01440764", "n01484850"])

    assert mapping == {"n01440764": 0, "n01443537": 1, "n01484850": 2}


def test_iter_image_records_maps_jpeg_tree_to_labels(tmp_path):
    cell = tmp_path / "pixelate" / "3"
    first = cell / "n01440764" / "ILSVRC2012_val_00000001.JPEG"
    third = cell / "n01443537" / "ILSVRC2012_val_00000003.JPEG"
    first.parent.mkdir(parents=True)
    third.parent.mkdir(parents=True)
    first.write_bytes(b"fake")
    third.write_bytes(b"fake")

    records = list(iter_image_records(tmp_path, "pixelate", 3, indices=[2, 0]))

    assert [(r.image_index, r.wnid, r.label) for r in records] == [
        (0, "n01440764", 0),
        (2, "n01443537", 1),
    ]


def test_iter_image_records_fails_on_missing_requested_index(tmp_path):
    cell = tmp_path / "pixelate" / "3" / "n01440764"
    cell.mkdir(parents=True)
    (cell / "ILSVRC2012_val_00000001.JPEG").write_bytes(b"fake")

    with pytest.raises(FileNotFoundError, match="missing requested ImageNet indices"):
        list(iter_image_records(tmp_path, "pixelate", 3, indices=[0, 1]))

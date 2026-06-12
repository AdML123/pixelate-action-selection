import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from pixelate_router.config import ResourceConfig


def test_resource_config_resolves_relative_paths(tmp_path):
    cfg = ResourceConfig.from_dict(
        {
            "root": str(tmp_path),
            "digital_root": "data/digital",
            "kair_root": "external/KAIR",
            "dncnn_checkpoint": "model_zoo/dncnn_color_blind.pth",
            "resnet50_checkpoint": "model_zoo/resnet50-11ad3fa6.pth",
            "derived_dir": "data/derived/imagenetc",
            "table_dir": "tables/imagenetc",
            "figure_dir": "figures/imagenetc",
        }
    )

    assert cfg.digital_root == tmp_path / "data/digital"
    assert cfg.figure_dir == tmp_path / "figures/imagenetc"

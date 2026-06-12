from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResourceConfig:
    root: Path
    digital_root: Path
    kair_root: Path
    dncnn_checkpoint: Path
    resnet50_checkpoint: Path
    derived_dir: Path
    table_dir: Path
    figure_dir: Path

    @classmethod
    def from_dict(cls, raw: dict) -> "ResourceConfig":
        root = Path(raw["root"]).expanduser().resolve()

        def resolve(key: str) -> Path:
            value = Path(raw[key]).expanduser()
            return value if value.is_absolute() else root / value

        return cls(
            root=root,
            digital_root=resolve("digital_root"),
            kair_root=resolve("kair_root"),
            dncnn_checkpoint=resolve("dncnn_checkpoint"),
            resnet50_checkpoint=resolve("resnet50_checkpoint"),
            derived_dir=resolve("derived_dir"),
            table_dir=resolve("table_dir"),
            figure_dir=resolve("figure_dir"),
        )

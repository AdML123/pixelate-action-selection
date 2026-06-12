import csv
import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from pixelate_router.io import open_text


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_packaged_intermediate_artifacts_support_fast_reproduction():
    required_files = [
        "data/action_eval/action_eval.csv.gz",
        "data/action_eval/action_eval.json",
        "data/features/feature_summary.json",
        "data/features_test/feature_summary.json",
        "data/router/router_all.pt",
        "data/router/router_no_commutator.pt",
        "data/router/router_confidence.pt",
        "data/router/router_spectral.pt",
        "data/router/validation_report.json",
        "data/router/ablation_report.json",
    ]
    for relative in required_files:
        assert (PACKAGE_ROOT / relative).exists(), relative

    assert len(list((PACKAGE_ROOT / "data" / "features").glob("features_*_search.npy"))) == 20
    assert len(list((PACKAGE_ROOT / "data" / "features").glob("features_*_validation.npy"))) == 20
    assert len(list((PACKAGE_ROOT / "data" / "features_test").glob("features_*_test.npy"))) == 20


def test_gzipped_action_eval_is_readable_with_package_io():
    action_csv = PACKAGE_ROOT / "data" / "action_eval" / "action_eval.csv.gz"

    with open_text(action_csv, newline="", encoding="utf-8") as handle:
        first = next(csv.DictReader(handle))

    assert first["split"] in {"search", "validation", "test"}
    assert first["corruption"] in {"contrast", "elastic_transform", "pixelate", "jpeg_compression"}
    assert first["path"] == ""


def test_intermediate_artifacts_do_not_expose_private_paths():
    forbidden = ["D:" + "\\", "C:" + "\\Users", "/" + "Users/", "\\" + "Users" + "\\"]
    offenders = []
    for path in [
        *PACKAGE_ROOT.joinpath("data", "action_eval").rglob("*"),
        *PACKAGE_ROOT.joinpath("data", "router").rglob("*"),
        *PACKAGE_ROOT.joinpath("data", "features").rglob("*.json"),
        *PACKAGE_ROOT.joinpath("data", "features_test").rglob("*.json"),
    ]:
        if not path.is_file() or path.suffix in {".pt", ".npy"}:
            continue
        if path.suffix == ".gz":
            text = gzip.open(path, mode="rt", encoding="utf-8").read()
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
        for term in forbidden:
            if term in text:
                offenders.append(f"{path.relative_to(PACKAGE_ROOT)}: {term}")

    assert offenders == []

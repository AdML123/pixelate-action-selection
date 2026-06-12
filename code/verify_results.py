import argparse
from pathlib import Path

from pixelate_router.verify import verify_required_outputs, verify_required_results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="data/derived/imagenetc")
    parser.add_argument("--tables", default="tables/imagenetc")
    parser.add_argument("--figures", default="figures/imagenetc")
    args = parser.parse_args()
    verify_required_results(Path(args.results))
    verify_required_outputs(Path(args.tables), Path(args.figures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    raise SystemExit("full model/data runner is implemented after resource smoke checks pass")


if __name__ == "__main__":
    raise SystemExit(main())

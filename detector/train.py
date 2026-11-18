"""Training entry point for the shared pothole + sign detector.

Stub: dataset and model choice are pending the first baseline (see docs/decisions.md).
"""

import sys
from pathlib import Path

import yaml

CONFIG = Path(__file__).parent / "configs" / "classes.yaml"


def load_classes() -> dict:
    with open(CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    classes = load_classes()
    print("Loaded classes:")
    for group, names in classes.items():
        print(f"  {group}: {len(names)}")
    print("Training not yet implemented. Dataset and model pending.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Load configs/default.yaml."""

from pathlib import Path

import yaml

DEFAULT = Path(__file__).parents[1] / "configs" / "default.yaml"


def load_config(path: str | Path = DEFAULT) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def all_drives(cfg: dict) -> list[str]:
    return cfg["drives"]["dev"] + cfg["drives"]["test"]

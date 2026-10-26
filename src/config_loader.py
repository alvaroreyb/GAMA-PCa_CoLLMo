from __future__ import annotations

from pathlib import Path
import os
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent / "resources"


def output_directory(root: Path = PROJECT_ROOT) -> Path:
    configured = os.environ.get("GAMA_PCA_OUTPUT_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.cwd() / "outputs").resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must contain a mapping: {p}")
    return data


def load_app_config(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    return load_yaml(root / "configs" / "app_config.yaml")


def load_ollama_config(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    return load_yaml(root / "configs" / "ollama_config.yaml")


def load_dataset_plan(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    return load_yaml(root / "configs" / "dataset_plan.yaml")

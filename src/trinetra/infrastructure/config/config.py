"""Configuration management for Trinetra-AI."""

from pathlib import Path
from typing import Any

import yaml

# Resolve the configs directory relative to this file:
# src/trinetra/infrastructure/config/config.py -> ../../../../../configs
_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "configs"


def load_config(file_name: str) -> dict[str, Any]:
    """Load a specific YAML configuration file.

    Args:
        file_name: The name of the file (e.g., "dataset.yaml" or just "dataset").

    Returns:
        A dictionary containing the configuration data.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    if not file_name.endswith((".yaml", ".yml")):
        file_name += ".yaml"

    file_path = _CONFIG_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config if config is not None else {}


def get_config(section: str) -> dict[str, Any]:
    """Load configuration for a specific section.

    Args:
        section: The name of the configuration section/file without extension.

    Returns:
        A dictionary containing the configuration data.
    """
    return load_config(section)

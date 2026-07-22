"""Tests for the configuration management system."""

import pytest

from trinetra_core.utils.config import get_config, load_config


def test_load_existing_config():
    """Verify that an existing YAML file loads successfully as a dictionary."""
    # We expect default.yaml to always exist based on the project structure
    config = load_config("default.yaml")

    assert isinstance(config, dict)
    assert "project_name" in config
    assert config["project_name"] == "Trinetra-AI"


def test_get_config_adds_extension():
    """Verify that get_config correctly appends the .yaml extension."""
    config = get_config("default")
    assert isinstance(config, dict)
    assert config["project_name"] == "Trinetra-AI"


def test_missing_config_raises_error():
    """Verify that attempting to load a non-existent config raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_config("nonexistent_config_file_123.yaml")

    assert "Configuration file not found" in str(exc_info.value)

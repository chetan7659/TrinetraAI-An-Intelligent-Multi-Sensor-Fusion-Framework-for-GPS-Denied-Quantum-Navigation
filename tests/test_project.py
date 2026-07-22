"""Test: Verify that main.py executes successfully."""

from __future__ import annotations

import subprocess
import sys


def test_main_executes_successfully() -> None:
    """main.py should exit with code 0 and print the expected message."""
    result = subprocess.run(
        [sys.executable, "main.py"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, f"main.py failed with: {result.stderr}"
    assert "Trinetra-AI initialized successfully." in result.stdout

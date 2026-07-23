"""Pytest configuration for unit tests."""

import matplotlib

# Enforce headless plotting for all unit tests to avoid Tkinter issues on Windows/CI
matplotlib.use("Agg")

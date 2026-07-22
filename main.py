"""Trinetra-AI: Entry point."""

import os
import sys

# Ensure the src directory is in the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from trinetra.__main__ import main

if __name__ == "__main__":
    main()

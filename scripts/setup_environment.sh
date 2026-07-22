#!/usr/bin/env bash
# ==============================================================================
# Trinetra-AI — Environment Setup (Linux / macOS)
# ==============================================================================
# Creates a virtual environment, activates it, and installs dependencies.
#
# Usage:
#   chmod +x scripts/setup_environment.sh
#   source scripts/setup_environment.sh
# ==============================================================================

set -euo pipefail

VENV_DIR=".venv"
REQUIREMENTS="requirements.txt"

echo "============================================================"
echo "  Trinetra-AI — Environment Setup"
echo "============================================================"

# ── 1. Check Python version ──────────────────────────
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON_CMD="$cmd"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python 3.11+ not found. Please install Python >= 3.11."
    exit 1
fi

PYTHON_VERSION=$("$PYTHON_CMD" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Using: $PYTHON_CMD ($PYTHON_VERSION)"

# ── 2. Create virtual environment ────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "  Virtual environment already exists at $VENV_DIR"
else
    echo "  Creating virtual environment at $VENV_DIR ..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
fi

# ── 3. Activate virtual environment ──────────────────
echo "  Activating virtual environment ..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ── 4. Upgrade pip ───────────────────────────────────
echo "  Upgrading pip ..."
pip install --upgrade pip --quiet

# ── 5. Install dependencies ─────────────────────────
if [ -f "$REQUIREMENTS" ]; then
    echo "  Installing dependencies from $REQUIREMENTS ..."
    pip install -r "$REQUIREMENTS" --quiet
else
    echo "  WARNING: $REQUIREMENTS not found. Skipping dependency install."
fi

# ── 6. Summary ───────────────────────────────────────
echo "============================================================"
echo "  ✅ Environment ready."
echo ""
echo "  Activate later with:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "  Verify with:"
echo "    python scripts/check_environment.py"
echo "============================================================"

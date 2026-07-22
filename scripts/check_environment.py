#!/usr/bin/env python3
"""Trinetra-AI: Environment verification script.

Checks that the Python version and all required dependencies are installed
and importable. Prints a summary table and exits with code 0 on success,
or code 1 if any check fails.
"""

from __future__ import annotations

import sys
import io


def _check_python_version(minimum: tuple[int, int] = (3, 11)) -> tuple[bool, str]:
    """Verify the running Python version meets the minimum requirement."""
    current = sys.version_info[:2]
    ok = current >= minimum
    label = f"{current[0]}.{current[1]}"
    req = f">={minimum[0]}.{minimum[1]}"
    return ok, f"Python {label} (requires {req})"


def _check_import(package: str) -> tuple[bool, str]:
    """Try to import *package* and return (success, description)."""
    try:
        mod = __import__(package)
        version = getattr(mod, "__version__", "unknown")
        return True, f"{package} {version}"
    except ImportError:
        return False, f"{package} — NOT FOUND"


def main() -> int:
    """Run all environment checks and print results."""
    checks: list[tuple[str, tuple[bool, str]]] = [
        ("Python version", _check_python_version()),
        ("NumPy", _check_import("numpy")),
        ("Pandas", _check_import("pandas")),
        ("SciPy", _check_import("scipy")),
        ("Matplotlib", _check_import("matplotlib")),
    ]

    # Ensure UTF-8 output on Windows consoles.
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )

    width = 60
    print("=" * width)
    print("  Trinetra-AI -- Environment Check")
    print("=" * width)

    all_ok = True
    for name, (ok, detail) in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name:<20s} {detail}")
        if not ok:
            all_ok = False

    print("-" * width)
    if all_ok:
        print("  All checks passed. Environment is ready.")
    else:
        print("  Some checks failed. Please install missing dependencies.")
        print("     Run: pip install -r requirements.txt")
    print("=" * width)

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

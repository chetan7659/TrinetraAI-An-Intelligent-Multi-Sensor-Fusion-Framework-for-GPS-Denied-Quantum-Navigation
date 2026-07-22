# Trinetra-AI

**An Intelligent Multi-Sensor Fusion Framework for GPS-Denied Quantum Navigation**

> M.Tech Research Project — Milestone M0: Environment & Project Setup

---

## Overview

Trinetra-AI is a research framework for intelligent navigation in GPS-denied environments. It combines multi-sensor fusion techniques with AI-driven state estimation to achieve robust positioning using quantum-enhanced inertial sensors, vision, LiDAR, and other complementary modalities.

---

## Project Structure

```
Trinetra-AI/
│
├── src/
│   └── trinetra_core/           # Core Python package
│       ├── acquisition/         # Sensor data acquisition interfaces
│       ├── preprocessing/       # Signal conditioning & cleaning
│       ├── features/            # Feature extraction & engineering
│       ├── sensor_fusion/       # Multi-sensor fusion algorithms
│       ├── navigation/          # State estimation & trajectory
│       ├── ai_modules/          # Deep learning / AI models
│       ├── evaluation/          # Metrics & benchmarking
│       └── utils/               # Shared utilities
│
├── tests/                       # Unit & integration tests
├── configs/                     # YAML / JSON configuration files
├── datasets/
│   ├── raw/                     # Unmodified sensor recordings
│   ├── processed/               # Cleaned / transformed data
│   └── external/                # Third-party benchmark datasets
│
├── models/                      # Trained model checkpoints
├── scripts/                     # CLI utilities & setup scripts
├── notebooks/                   # Jupyter exploration notebooks
├── docs/                        # Documentation & architecture notes
├── docker/                      # Dockerfiles & compose configs
├── research/                    # Papers, literature notes, references
├── results/                     # Experiment outputs & logs
│
├── main.py                      # Entry point
├── pyproject.toml               # PEP 621 project metadata
├── requirements.txt             # Pip dependencies
├── LICENSE                      # MIT License
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

---

## Getting Started

### Prerequisites

| Tool   | Version  | Check command        |
|--------|----------|----------------------|
| Python | ≥ 3.11   | `python --version`   |
| pip    | latest   | `pip --version`      |
| git    | any      | `git --version`      |

### Installation

#### Option A — Automated Setup (Recommended)

**Windows (PowerShell):**
```powershell
.\scripts\setup_environment.ps1
```

**Linux / macOS:**
```bash
chmod +x scripts/setup_environment.sh
source scripts/setup_environment.sh
```

#### Option B — Manual Setup

**1. Create a virtual environment**

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

**2. Upgrade pip**

```bash
pip install --upgrade pip
```

**3. Install dependencies**

```bash
# Core dependencies
pip install -r requirements.txt

# Development tools (optional)
pip install -e ".[dev]"
```

### Verify Installation

```bash
python scripts/check_environment.py
```

Expected output:

```
============================================================
  Trinetra-AI — Environment Check
============================================================
  [✓] Python version       Python 3.11 (requires >=3.11)
  [✓] NumPy                numpy 1.26.x
  [✓] Pandas               pandas 2.1.x
  [✓] SciPy                scipy 1.11.x
  [✓] Matplotlib           matplotlib 3.8.x
------------------------------------------------------------
  ✅ All checks passed. Environment is ready.
============================================================
```

### Run the Application

```bash
python main.py
# → Trinetra-AI initialized successfully.
```

---

## Development Status

| Milestone | Description                  | Status      |
|-----------|------------------------------|-------------|
| **M0**    | Environment & Project Setup  | 🔧 Active  |
| M1        | Data Acquisition Pipeline    | 🔲 Planned  |
| M2        | Preprocessing & Features     | 🔲 Planned  |
| M3        | Sensor Fusion Engine         | 🔲 Planned  |
| M4        | AI Module Integration        | 🔲 Planned  |
| M5        | Evaluation & Benchmarks      | 🔲 Planned  |

---

## Development Workflow

### Install Development Dependencies

```bash
# Activate your virtual environment first, then:
pip install -r requirements.txt

# Or install as editable with all dev tools:
pip install -e ".[dev]"
```

### Code Formatting — Black

[Black](https://black.readthedocs.io/) is an opinionated code formatter that enforces a consistent style across the entire codebase. It eliminates style debates in code reviews.

```bash
# Format all source files
black src/ tests/ main.py

# Check without modifying (dry run)
black --check src/ tests/ main.py

# Show what would change
black --diff src/ tests/ main.py
```

### Linting — Ruff

[Ruff](https://docs.astral.sh/ruff/) is an extremely fast Python linter that catches bugs, enforces best practices, and auto-sorts imports. It replaces flake8, isort, and several other tools in a single binary.

```bash
# Lint all source files
ruff check src/ tests/ main.py

# Lint and auto-fix issues
ruff check --fix src/ tests/ main.py
```

### Testing — Pytest

[Pytest](https://docs.pytest.org/) discovers and runs tests with minimal boilerplate. It integrates with coverage reporting to measure how much of the code is exercised by tests.

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src/trinetra_core --cov-report=term-missing

# Run a specific test file
pytest tests/test_project.py -v
```

### Pre-commit Hooks

[Pre-commit](https://pre-commit.com/) runs formatting and linting checks automatically before every `git commit`. This ensures no unformatted or broken code enters the repository.

```bash
# Install hooks (one-time setup)
pre-commit install

# Run all hooks on every file (manual)
pre-commit run --all-files

# Update hook versions
pre-commit autoupdate
```

### EditorConfig

The `.editorconfig` file ensures consistent editor settings (UTF-8, LF line endings, 4-space indentation) across VS Code, JetBrains, Vim, and other editors that support the [EditorConfig](https://editorconfig.org/) standard. No manual configuration needed.

### VS Code Integration

The `.vscode/settings.json` file pre-configures:

- **Black** as the default Python formatter with format-on-save
- **Ruff** for real-time linting with auto-fix
- **Pytest** for test discovery and the testing sidebar
- **100-character ruler** matching the project line-length standard

Recommended VS Code extensions:

- `ms-python.python`
- `ms-python.black-formatter`
- `charliermarsh.ruff`

---

## Configuration Management

Trinetra-AI uses a centralized configuration system powered by YAML files located in the `configs/` directory.

### Why Centralized Configuration?

- **Single Source of Truth** — No magic numbers hidden in source code. All hyperparameters and paths are centralized.
- **Reproducibility** — Research experiments can be fully reproduced by saving the configuration state.
- **Modularity** — Configurations are logically separated by domain (e.g., `dataset.yaml`, `model.yaml`, `navigation.yaml`).

### Why YAML?

YAML was chosen because it is human-readable, supports comments, and clearly represents nested hierarchical data better than JSON or INI.

### Usage

Future modules will consume configuration via the clean helper functions in `src/trinetra_core/utils/config.py`:

```python
from trinetra_core.utils.config import get_config

# Load dataset configuration
dataset_cfg = get_config("dataset")
print(dataset_cfg["sampling_rate"])
```

## Logging & Error Handling

Trinetra-AI uses a centralized logging and exception system to ensure consistent monitoring and debugging across all research experiments.

### Logging

The logger is built on Python's standard `logging` library. By centralizing it, we guarantee:
- Consistent log formatting (Timestamp, Level, Module, Message)
- Simultaneous console output and file persistence
- Dynamic log level control via `configs/logging.yaml`

Log files are automatically saved to the `logs/` directory. They are generated daily (e.g., `trinetra_2026-07-23.log`).

#### Usage

Every future module should instantiate its logger like this:

```python
from trinetra_core.utils.logger import setup_logger
from trinetra_core.utils.exceptions import DatasetError

# Initialize logger for the current module
logger = setup_logger(__name__)

logger.info("Dataset preprocessing started.")

try:
    # Some logic
    raise DatasetError("Dataset directory not found.")
except DatasetError as e:
    logger.error(f"Failed to load dataset: {e}")
```

### Exception Hierarchy

Custom exceptions provide granular error tracking without leaking implementation details. All exceptions inherit from `TrinetraError`.
Available exceptions include: `ConfigurationError`, `DatasetError`, `PreprocessingError`, `SensorFusionError`, `NavigationError`, `ModelError`, and `EvaluationError`.

---

## Why Virtual Environments?

Virtual environments are **essential** for research software:

- **Isolation** — Each project gets its own dependency tree. Installing SciPy 1.11 for Trinetra-AI won't break another project that requires SciPy 1.9.
- **Reproducibility** — A collaborator (or your future self) can recreate the exact same environment from `requirements.txt`, ensuring experiments produce identical results.
- **Clean system** — Your system Python stays untouched. No risk of corrupting OS-level packages.

## Why Dependency Pinning?

Pinning versions (e.g., `numpy>=1.26.0`) prevents silent breakage:

- **Deterministic builds** — `pip install -r requirements.txt` produces the same environment today and six months from now.
- **Debugging confidence** — When a regression appears, you know it's in *your* code, not a transitive dependency upgrade.
- **Publication integrity** — Research results must be tied to a specific software stack. Reviewers and readers should be able to reproduce your experiments exactly.

## Why Reproducibility Matters in Research Software

Scientific software has a higher bar than typical application code:

- **Peer review** — Reviewers must be able to run your code and replicate your results.
- **Long-term archival** — Your M.Tech thesis results should be reproducible years after submission.
- **Collaboration** — Team members across different operating systems (Windows, Linux, macOS) must get identical behaviour.
- **Credibility** — Irreproducible results undermine the scientific contribution, regardless of how novel the algorithm is.

This project uses virtual environments, pinned dependencies, and cross-platform setup scripts to ensure full reproducibility from day one.

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

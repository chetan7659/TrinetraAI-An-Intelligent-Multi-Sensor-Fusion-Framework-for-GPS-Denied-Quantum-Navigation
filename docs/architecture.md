# Trinetra-AI: Clean Architecture Design

This document details the architectural layout of the Trinetra-AI multi-sensor fusion framework, based on Clean Architecture (Hexagonal Architecture) principles.

---

## The Dependency Rule

The core principle of this architecture is **The Dependency Rule**:
> Dependencies must always point inward. Source code dependencies can only point from outer layers toward inner layers.

Components in the inner circles (e.g., Domain Core) cannot know anything about components in the outer circles (e.g., Infrastructure, Adapters). In practice, this means that:
- Business entities and interfaces never import anything from adapters, configuration utilities, or databases.
- The Application layer only imports from the Domain layer.
- Adapters and Infrastructure can import from the Application and Domain layers.

```text
    +-----------------------------------------------+
    |                INFRASTRUCTURE                 |
    |  (config, logging, persistence, utils)        |
    |      +---------------------------------+      |
    |      |            ADAPTERS             |      |
    |      |  (sensors, datasets, storage)   |      |
    |      |      +-------------------+      |      |
    |      |      |    APPLICATION    |      |      |
    |      |      |    (use cases)    |      |      |
    |      |      |      +-----+      |      |      |
    |      |      |      | DOM |      |      |      |
    |      |      |      +-----+      |      |      |
    |      |      +-------------------+      |      |
    |      +---------------------------------+      |
    +-----------------------------------------------+
                 Inward Dependency Flow (--->)
```

---

## Architectural Layers

### 1. Domain Core (`src/trinetra/domain/`)
The domain layer represents the core business rules, algorithms, and logical abstractions. It is strictly stateless and has no external dependencies.
*   **`fusion/`** — Core sensor fusion algorithms (e.g. Kalman Filter mathematics, kinematics equations).
*   **`navigation/`** — Trajectory models and inertial dead reckoning mathematics.
*   **`state/`** — Definitions of physical states (e.g. Position, Velocity, Attitude).
*   **`ai/`** — Deep learning module abstractions and estimation boundaries.
*   **`interfaces/`** — Boundary ports defining abstract class models (e.g. `AbstractSensorReader`).
*   **`common/`** — Standard math operations, units, and models.

*Allowed imports:* None (except internal domain packages and Python builtins).

### 2. Application (`src/trinetra/application/`)
Defines the software-specific use cases and coordinates the flow of data to and from the domain entities.
*   **`use_cases/`** — Single-focus application tasks (e.g. `run_calibration`).
*   **`orchestration/`** — Managing step-by-step fusion executions and pipeline runs.
*   **`services/`** — Helpers and orchestration abstractions.

*Allowed imports:* `domain/`, Python builtins.

### 3. Interface Adapters (`src/trinetra/adapters/`)
Translates data between the format convenient for the use cases/domain, and the format required by external agencies (e.g. hardware, database).
*   **`sensors/`** — Physical sensor drivers and interfaces (LiDAR, GNSS, IMU readers).
*   **`datasets/`** — Parsers and loaders for raw training data.
*   **`storage/`** — Disk serializers, ONNX inference loaders, CSV file writers.
*   **`visualization/`** — Trajectory mapping and performance plotting.
*   **`ros2/`** — ROS2 node publishers, subscribers, and message schemas.

*Allowed imports:* `application/`, `domain/`, standard libraries.

### 4. Infrastructure (`src/trinetra/infrastructure/`)
Contains all the concrete framework implementations, utility modules, and system configurations.
*   **`config/`** — YAML parser and file loader logic.
*   **`logging/`** — Central logging engine.
*   **`persistence/`** — Database connectors, disk caching.
*   **`utils/`** — Technical helpers.

*Allowed imports:* `domain/`, `application/`, `adapters/`.

### 5. Shared (`src/trinetra/shared/`)
Contains cross-cutting, lightweight concerns used across all other layers without violating clean architecture boundaries.
*   **`exceptions/`** — Global exception classes (e.g., `TrinetraError`).
*   **`types/`** — Broadly reused type definitions.
*   **`constants/`** — Global constant variables.

*Allowed imports:* None (strictly static definitions only).

---

## Component Mapping (C1 - C12)

The architecture maps to the project components in the following structural units:

| Component Code | Clean Architecture Directory | Responsibility |
|:---:|:---|:---|
| **C1** | `adapters/sensors` | Physical sensor readers and drivers (IMU, GNSS, LiDAR). |
| **C2** | `application/services` | Coordination services for processing sensor inputs. |
| **C3** | `domain/common` | Baseline physical constants, coordinate systems, math helpers. |
| **C4** | `domain/ai` | Neural estimator interfaces, model definitions. |
| **C5** | `domain/fusion` | Multi-sensor fusion mathematical implementations (EKF/UKF). |
| **C6** | `domain/navigation` | Kinematic trajectory and state transition models. |
| **C7** | `domain/state` | Definitions of navigational state structures. |
| **C8** | `application/orchestration` | Core navigation execution loop coordinator. |
| **C9** | `adapters/storage` | Database wrappers and disk cache serializers. |
| **C10** | `adapters/visualization` | Live plotting, Matplotlib trajectory maps. |
| **C11** | `adapters/ros2` | ROS2 node bindings and sensor topic integration. |
| **C12** | `infrastructure/config` | YAML configurations parsing and loader interface. |

---

## Example Import Direction

### Allowed Import Flow:
An adapter reading IMU data (`C1`) needs to log details and raise exceptions if parameters are out of range:
```python
# File: src/trinetra/adapters/sensors/imu_reader.py
from trinetra.infrastructure.logging import setup_logger   # OK: Outward importing helper
from trinetra.shared.exceptions import SensorFusionError  # OK: Shared error import
```

### Prohibited Import Flow (Architectural Break):
A core Kalman Filter solver in `domain/fusion` must **never** reference configurations, log files, or databases directly:
```python
# File: src/trinetra/domain/fusion/kalman_filter.py
from trinetra.infrastructure.config import get_config  # LINT ERROR: Inward dependency break!
```
Instead, configurations should be loaded at the Infrastructure/Application layer and passed down into the domain algorithms as primitives or simple parameters.

---

## Dataset Architecture (C1 Sensor Ingestion)

The Dataset Ingestion component establishes a clear boundary between raw external data and the internal domain logic. The following section documents the architecture in its implemented state, using the RoNIN adapter as the reference implementation.

### Why Adapters Exist
Research datasets are delivered in wildly different formats (HDF5, ROS bags, CSVs, custom binary files) with inconsistent naming conventions, coordinate frames, and timestamp strategies. Interface adapters isolate this variance. Each adapter translates one specific dataset's filesystem layout into a standardized structure (e.g., `Recording`, `SensorRecord`) that the rest of the system understands. The Domain layer is never exposed to format-specific concerns.

### How Datasets Plug Into the System
Every concrete dataset adapter must:
1. Inherit from `DatasetInterface` (`src/trinetra/domain/interfaces/dataset_interface.py`).
2. Implement all eight abstract methods of that interface.
3. Call `registry.register_dataset(name, AdapterClass)` at import time.

The Application layer requests a dataset by name from the registry (`registry.get_dataset("ronin_dataset")`), receives the *class*, instantiates it, and calls its interface methods. No part of the Application or Domain layer needs to know which concrete adapter is in use.

### RoNIN Adapter: Responsibilities

**Module**: `src/trinetra/adapters/datasets/ronin/`

| Component | Responsibility |
|:---|:---|
| `models.py` | Defines `Recording`, a frozen (immutable) dataclass representing one recording session. Pure data; no methods. |
| `validator.py` | Encapsulates all filesystem checks. Returns structured `DatasetValidationReport` objects. Does NOT open any file. |
| `adapter.py` | Implements `DatasetInterface`. Resolves paths from `configs/dataset.yaml`. Discovers splits and recordings dynamically from the filesystem. Delegates all validation to `RoninValidator`. |
| `__init__.py` | Exports the public API of the subpackage. Triggers auto-registration in the global registry. |

### Separation of Concerns: Three Independent Components

The RoNIN dataset pipeline is deliberately split into three independent components. Each can be developed, tested, and replaced without affecting the others.

| Concern | Component | Module | Status |
|:---|:---|:---|:---:|
| **Dataset Discovery** — find splits and recording dirs | `RoninAdapter` | `adapter.py` | ✅ Done (M1.2) |
| **Metadata Parsing** — read and validate `info.json` | `RoninMetadataLoader` | `metadata_loader.py` | ✅ Done (M1.3) |
| **Sensor Decoding** — read raw arrays from `data.hdf5` | HDF5 Reader | `hdf5_reader.py` | 🔜 M1.4 |
| **Preprocessing** — calibration, resampling, windowing | Preprocessing Pipeline | `preprocessing/` | 🔜 M2 |

### RoNIN Data Pipeline (M1.2 + M1.3)

```
datasets/raw/ronin/
        │
        ▼
  RoninAdapter                 ← M1.2: filesystem discovery
  (adapter.py)
        │  returns
        ▼
  Recording                    ← immutable value object (models.py)
  (recording_id, split,
   recording_path,
   hdf5_path, info_path)
        │  consumed by
        ▼
  RoninMetadataLoader          ← M1.3: opens info.json only
  (metadata_loader.py)
        │  returns
        ▼
  RoninRecordingMetadata       ← immutable value object (metadata_models.py)
  (device, length, date,
   calibration, time_sync,
   orientation, errors)
        │
        ▼
  [HDF5 Reader — M1.4]         ← will open data.hdf5 only
        │
        ▼
  SensorRecord stream          ← domain-level abstraction
        │
        ▼
  Application layer → Domain
```

### Component Responsibility Boundaries

**RoninAdapter** (`adapter.py`):
- Discovers split directories by scanning the filesystem (no filenames hardcoded).
- Discovers valid recordings (presence of `data.hdf5` + `info.json` only).
- Returns `Recording` objects (paths only; no file content).
- Does **not** open any file.

**RoninMetadataLoader** (`metadata_loader.py`):
- Accepts a `Recording` and opens **only** `recording.info_path`.
- Parses and validates all required JSON fields against the real schema.
- Raises `DatasetError` with precise context on any failure.
- Returns an immutable `RoninRecordingMetadata` value object.
- Does **not** open `data.hdf5`.

**HDF5 Reader** (future, M1.4):
- Will accept a `Recording` and open **only** `recording.hdf5_path`.
- Will decode raw sensor arrays (gyro, acce, etc.) from the HDF5 groups.
- Will yield `SensorRecord` objects consumed by the Application layer.
- Does **not** re-parse `info.json` (uses `RoninRecordingMetadata` if calibration is needed).

### Why the Domain Layer Never Imports Dataset-Specific Code

The Domain layer (where Kalman filters, kinematics, and state models live) must remain completely ignorant of data sources. If the Domain layer imported an HDF5 parsing routine or referenced a RoNIN-specific field name, it would become tightly coupled to one dataset's format — a direct violation of the Dependency Rule.

The Domain only ever receives abstract `SensorRecord` values, ensuring that swapping one dataset for another requires only a change of adapter — zero changes to core algorithms.

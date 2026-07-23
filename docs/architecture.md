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

### Separation of Concerns: Five Independent Components

The RoNIN dataset pipeline is split into five strictly independent components. Each component has exactly one responsibility and zero knowledge of its neighbours' internals.

| Concern | Component | Module | Status |
|:---|:---|:---|:---:|
| **Dataset Discovery** — find splits and recording dirs | `RoninAdapter` | `adapter.py` | ✅ Done (M1.2) |
| **Metadata Parsing** — read and validate `info.json` | `RoninMetadataLoader` | `metadata_loader.py` | ✅ Done (M1.3) |
| **Sensor Decoding** — read raw arrays from `data.hdf5` → `RoninRawFrame` | `RoninHDF5Reader` | `hdf5_reader.py` | ✅ Done (M1.4) |
| **Canonical Mapping** — `RoninRawFrame` → domain `SensorRecord` | `RoninCanonicalMapper` | `canonical_mapper.py` | ✅ Done (M1.5) |
| **Dataset Utilities** — iteration helpers, split loaders, batching | Utilities | `utils.py` | 🔜 M1.6 |
| **EDA** — visualisation, statistics, distribution analysis | Notebooks / scripts | `reports/eda/` | 🔜 M1.7 |

### RoNIN Data Pipeline (Target Architecture)

```
datasets/raw/ronin/
        │
        ▼
  RoninAdapter                 ← M1.2: filesystem discovery only
  (adapter.py)                   Does NOT open any file.
        │  returns
        ▼
  Recording                    ← immutable value object (models.py)
  (recording_id, split,
   recording_path,
   hdf5_path, info_path)
        │
        ├─────────────────────────────────────┐
        ▼                                     ▼
  RoninMetadataLoader          ← M1.3    RoninHDF5Reader           ← M1.4
  (metadata_loader.py)                   (hdf5_reader.py)
  Opens info.json only.                  Opens data.hdf5 only.
        │  returns                             │  returns
        ▼                                     ▼
  RoninRecordingMetadata                 RoninRawFrame
  (frozen dataclass)                     (dataset-specific, frozen)
  device, length, date,                  timestamp, gyro, acce,
  calibration, time_sync,                gravity, game_rv, linacce,
  orientation, errors                    start_frame applied
        │                                     │
        │         consumed together by        │
        └──────────────┬──────────────────────┘
                       ▼
  RoninCanonicalMapper             ← M1.5: domain boundary crossing
  (canonical_mapper.py)
  Converts RoninRawFrame → SensorRecord.
  The ONLY component allowed to import both
  adapter types (RoninRawFrame) and domain types (SensorRecord).
                       │  returns
                       ▼
  SensorRecord                     ← domain abstraction (sensor_record.py)
  (timestamp, sensor_type,           Completely ignorant of HDF5, RoNIN,
   values, frame_id, metadata)       or any dataset format.
                       │
                       ▼
  Application layer → Domain
```

### Why the HDF5 Reader Must Not Produce SensorRecord

If `RoninHDF5Reader` directly yielded `SensorRecord` objects, it would be responsible for two distinct concerns:

1. **File format decoding** — navigating HDF5 group structure, reading datasets, handling data types.
2. **Domain mapping** — deciding what a `timestamp`, `sensor_type`, and `values` mean in the domain model.

This violates the **Single Responsibility Principle**. It also forces the HDF5 reader to import from the Domain layer (`sensor_record.py`), which is an outward-to-inward dependency — a direct violation of the **Dependency Rule**.

With the four-stage design, every future dataset follows the identical pattern:

| Dataset | File Reader | Raw Frame | Mapper | Domain Output |
|:---|:---|:---|:---|:---|
| RoNIN | `RoninHDF5Reader` | `RoninRawFrame` | `RoninCanonicalMapper` | `SensorRecord` |
| TLIO | `TlioHDF5Reader` | `TlioRawFrame` | `TlioCanonicalMapper` | `SensorRecord` |
| EuRoC | `EurocCsvReader` | `EurocRawFrame` | `EurocCanonicalMapper` | `SensorRecord` |
| KITTI | `KittiOxtsReader` | `KittiRawFrame` | `KittiCanonicalMapper` | `SensorRecord` |

The Application layer and Domain **always receive `SensorRecord`** — regardless of which dataset is in use.

### Component Responsibility Boundaries

**RoninAdapter** (`adapter.py`) — M1.2:
- Discovers split directories by scanning the filesystem.
- Returns `Recording` path objects. Opens **no file**.

**RoninMetadataLoader** (`metadata_loader.py`) — M1.3:
- Accepts a `Recording`. Opens **only** `recording.info_path`.
- Returns an immutable `RoninRecordingMetadata`.
- Does **not** open `data.hdf5`.

**RoninHDF5Reader** (`hdf5_reader.py`) — M1.4:
- Accepts a `Recording` + `RoninRecordingMetadata`.
- Opens **only** `recording.hdf5_path` using `h5py`.
- Reads `synced/` group arrays; applies `start_frame` slice.
- Returns a lazy iterator of `RoninRawFrame` (dataset-specific frozen dataclass).
- Does **not** import from `domain/`. Does **not** re-parse `info.json`.

**RoninCanonicalMapper** (`canonical_mapper.py`) — M1.5:
- Accepts `RoninRawFrame` objects.
- Returns `SensorRecord` domain objects.
- Is the **only** component that imports both adapter types and domain types.
- Contains all RoNIN-to-domain field name translation logic.
- Is pure (stateless, no I/O).

### Why the Domain Layer Never Imports Dataset-Specific Code

The Domain layer (where Kalman filters, kinematics, and state models live) must remain completely ignorant of data sources. The `SensorRecord` type it receives carries only canonical field names. Any dataset can feed the Domain by implementing its own Reader + Mapper pair — zero changes to core algorithms.

See [ADR-0002](adr/0002-hdf5-reader-canonical-mapper-separation.md) for the full decision record.

---

## Application Layer

```
src/trinetra/application/
└── dataset/
    ├── __init__.py
    ├── filters.py              ← M1.6.2
    ├── recording_iterator.py   ← M1.6.1
    ├── samplers.py             ← M1.6.3
    └── split_loader.py         ← M1.6.4
```

### Purpose

The Application layer **orchestrates** adapter and domain components.
It holds no business logic and performs no parsing, decoding, or mapping itself.
Its sole responsibility is to sequence collaborator calls and expose a clean,
dataset-agnostic API to higher layers (e.g., training loops, EDA notebooks).

### SplitLoader — M1.6.4

**Location**: `src/trinetra/application/dataset/split_loader.py`

**Responsibility**: Orchestrate dataset discovery and single-recording iteration to expose a continuous, lazy stream of `SensorRecord` objects across an entire dataset split.

**Dependency Injection**:
- `adapter`: Used to list recordings belonging to a requested split.
- `recording_iterator`: Used to lazily load and map each recording in turn.

While `RecordingIterator` orchestrates **one recording**, `SplitLoader` orchestrates **multiple recordings** within a dataset partition (e.g., "train", "seen", "unseen").

**Example usage**:
```python
loader = SplitLoader(adapter, recording_iterator)

for record in loader.iter_split("train"):
    train(record)
```

### RecordingIterator — M1.6.1

**Location**: `src/trinetra/application/dataset/recording_iterator.py`

**Responsibility**: Orchestrate one recording's ingestion pipeline and produce
a lazy `Iterator[SensorRecord]` for the caller.

**What it does**:
1. Calls `MetadataLoader.load(recording)` — once, before iteration begins.
2. Calls `HDF5Reader.read(recording, metadata)` — returns a lazy raw-frame iterator.
3. Calls `CanonicalMapper.map_frames(raw_frames)` — wraps raw frames in a mapping generator.
4. `yield from` the mapped iterator — one `SensorRecord` per time step.

**What it does NOT do**:
- Open files (no `h5py`, no `open()`)
- Parse JSON or YAML
- Traverse the filesystem
- Perform unit conversion, calibration, or normalisation
- Filter, batch, or window frames

**Dependency Injection**: All three collaborators are injected through the
constructor. `RecordingIterator` never instantiates collaborators internally.
This design makes the service testable with lightweight stubs and decoupled
from any specific dataset implementation.

**Collaborator protocols**:

```python
class MetadataLoaderProtocol(Protocol):
    def load(self, recording: Any) -> Any: ...

class RawFrameReaderProtocol(Protocol):
    def read(self, recording: Any, metadata: Any) -> Iterator[Any]: ...

class CanonicalMapperProtocol(Protocol):
    def map_frames(self, frames: Iterable[Any]) -> Iterator[SensorRecord]: ...
```

Any class that satisfies the structural interface (duck-typing) can be injected —
not only the RoNIN-specific implementations.

**Example usage**:

```python
from trinetra.adapters.datasets.ronin import (
    RoninAdapter, RoninMetadataLoader,
    RoninHDF5Reader, RoninCanonicalMapper,
)
from trinetra.application.dataset.recording_iterator import RecordingIterator

adapter   = RoninAdapter()
recording = adapter.get_recording("a000_1")

iterator  = RecordingIterator(
    metadata_loader=RoninMetadataLoader(),
    hdf5_reader=RoninHDF5Reader(),
    canonical_mapper=RoninCanonicalMapper(),
)

for record in iterator.iter_recording(recording):
    print(record.timestamp, record.accelerometer)
```

**Extensibility**: When a second dataset (e.g., TLIO) is added, the caller
simply injects `TlioMetadataLoader`, `TlioHDF5Reader`, and `TlioCanonicalMapper`.
The `RecordingIterator` class itself requires **zero changes**.

### Filters — M1.6.2

**Location**: `src/trinetra/application/dataset/filters.py`

**Responsibility**: Provide pure, lazy transformation utilities over streams of `SensorRecord` objects.

These utilities are generic generator functions that consume an `Iterable[SensorRecord]` and yield an `Iterator[SensorRecord]`. They are entirely dataset-agnostic and hold no adapter dependencies, performing pure functional stream transformations.

**Functions Provided**:
- `filter_by_frame(records, start, end)`
- `filter_by_time(records, start, end)`
- `take(records, count)`
- `skip(records, count)`
- `predicate_filter(records, predicate)`

**Example Chaining**:
```python
records = iterator.iter_recording(recording)
records = filter_by_time(records, start=5.0, end=15.0)
records = skip(records, 100)
records = take(records, 500)

for record in records:
    ...
```

### Sampling & Batching — M1.6.3

**Location**: `src/trinetra/application/dataset/samplers.py`

**Responsibility**: Provide pure utilities for sampling and grouping streams of `SensorRecord` objects.

These utilities prepare `SensorRecord` streams for downstream preprocessing and machine learning. They operate exclusively on canonical domain records and remain completely dataset-agnostic. All functions return lazy iterators to avoid materializing large lists.

**Functions Provided**:
- `batch(records, batch_size)`: Group records into lists of `batch_size`.
- `window(records, size)`: Create a sliding window of exactly `size` records (yields tuples).
- `stride(records, step)`: Yield every `step`-th record.
- `chunk_by_time(records, duration)`: Group consecutive records whose timestamps fall within the same fixed-duration interval.

**Example Usage**:
```python
records = iterator.iter_recording(recording)
records = filter_by_time(records, start=5.0, end=25.0)

for batch_records in batch(records, batch_size=256):
    process(batch_records)

records = iterator.iter_recording(recording)
for window_records in window(records, size=200):
    extract_features(window_records)
```

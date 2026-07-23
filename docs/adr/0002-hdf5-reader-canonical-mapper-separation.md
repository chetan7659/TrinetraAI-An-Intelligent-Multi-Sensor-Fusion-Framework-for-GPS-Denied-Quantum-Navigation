# ADR-0002: Separate HDF5 Reader from Canonical Domain Mapper

**Status:** Accepted
**Date:** 2026-07-23
**Milestone:** M1.4 / M1.5 (pre-implementation)
**Author:** Trinetra-AI Architecture Team
**Reviewers:** Senior Software Architect (design review, 2026-07-23)

---

## Context

After completing M1.3 (RoNIN Metadata Loader), the original roadmap proposed:

```
M1.4: RoNIN HDF5 Reader
        ↓
    SensorRecord  (domain type)
```

The HDF5 Reader was expected to open `data.hdf5`, decode raw sensor arrays, and
directly produce `SensorRecord` objects for the Application layer.

This was identified as an architectural violation before any implementation began.

---

## Problem

Producing `SensorRecord` from inside the HDF5 Reader assigns it **two distinct
responsibilities**:

1. **File format decoding** — navigating HDF5 group structure, slicing arrays
   by `start_frame`, handling data types and shapes.
2. **Domain mapping** — deciding that RoNIN's `synced/gyro` maps to
   `SensorRecord.sensor_type = "gyroscope"`, that timestamps are in seconds, etc.

This violates the **Single Responsibility Principle**.

Additionally, producing `SensorRecord` requires the HDF5 Reader to import from
`trinetra.domain.interfaces.sensor_record`. This is an **Adapters → Domain**
import, which is correct. However, the mapping logic — which fields from the
HDF5 file map to which canonical field names — does not belong in a component
whose only job is to decode a binary file format.

When a second dataset (e.g. TLIO, EuRoC, KITTI) is added:

- TLIO stores IMU in `data/gyro` and `data/acce` (HDF5).
- EuRoC stores IMU in `data/imu0/data` (CSV).
- KITTI stores IMU in `oxts/data/*.txt` files.

If each reader decides independently how to map its own fields to `SensorRecord`,
the mapping logic is fragmented across unrelated modules with no single place
to audit or test the canonical field translation.

---

## Decision

**Split M1.4 into two separate milestones and two separate components:**

### M1.4 — RoNIN HDF5 Reader

**Module:** `src/trinetra/adapters/datasets/ronin/hdf5_reader.py`

- Input: `Recording`, `RoninRecordingMetadata`
- Output: sequence of `RoninRawFrame` (dataset-specific frozen dataclass)
- Opens `recording.hdf5_path` **only**
- Reads `synced/` group: `time`, `gyro`, `acce`, `gravity`, `game_rv`, `linacce`
- Applies `metadata.start_frame` slice to exclude pre-calibration frames
- Returns data in the **exact units and coordinate frames stored in the file**
- Does **not** import from `trinetra.domain`
- Does **not** re-parse `info.json`

### M1.5 — Canonical Mapper

**Module:** `src/trinetra/adapters/datasets/ronin/canonical_mapper.py`

- Input: `RoninRawFrame`
- Output: `SensorRecord` (domain type)
- Is **stateless and pure** (no I/O)
- Is the **only** component in the RoNIN package allowed to import both:
  - `RoninRawFrame` (adapter type)
  - `SensorRecord` (domain type)
- Contains all field-name translation logic for RoNIN → canonical schema

---

## New Pipeline

```
Recording
    │
    ├──────────────────────────────────────┐
    ▼                                      ▼
RoninMetadataLoader (M1.3)          RoninHDF5Reader (M1.4)
Opens info.json only.               Opens data.hdf5 only.
    │ returns                              │ returns
    ▼                                      ▼
RoninRecordingMetadata              RoninRawFrame
(calibration, time_sync, etc.)      (timestamp[], gyro[], acce[], ...)
    │                                      │
    └──────────────┬───────────────────────┘
                   ▼
    RoninCanonicalMapper (M1.5)
    Pure. No I/O.
                   │ returns
                   ▼
            SensorRecord (domain)
                   │
                   ▼
        Application layer → Domain
```

---

## Rationale

### Single Responsibility Principle

Each component does exactly one thing:

| Component | Responsibility |
|:---|:---|
| `RoninAdapter` | Filesystem discovery |
| `RoninMetadataLoader` | Parse `info.json` |
| `RoninHDF5Reader` | Decode `data.hdf5` → raw arrays |
| `RoninCanonicalMapper` | Translate raw arrays → domain `SensorRecord` |

### Dependency Rule

```
Domain layer (SensorRecord)
        ↑
    imports from
        │
CanonicalMapper  ←  imports RoninRawFrame (adapter)
                        + SensorRecord (domain)

HDF5Reader  →  produces RoninRawFrame
               does NOT import domain types
```

The HDF5 Reader never crosses the adapter/domain boundary.

### Scalability Across Datasets

Every future dataset implements the same two-step pattern:

```python
# TLIO
TlioHDF5Reader  → TlioRawFrame  → TlioCanonicalMapper → SensorRecord

# EuRoC
EurocCsvReader  → EurocRawFrame → EurocCanonicalMapper → SensorRecord

# KITTI
KittiOxtsReader → KittiRawFrame → KittiCanonicalMapper → SensorRecord
```

The Application layer always calls the Mapper, never the Reader directly.
The Domain always receives `SensorRecord`. Swapping datasets requires only
changing which Reader+Mapper pair the Application layer instantiates.

### Testability

- **HDF5 Reader** can be tested by providing a synthetic HDF5 file and
  asserting that `RoninRawFrame` contains the correct raw array shapes and
  dtypes — no domain knowledge required.
- **Canonical Mapper** can be tested with synthetic `RoninRawFrame` objects
  (pure dataclasses, no file I/O) and asserting that the resulting
  `SensorRecord` fields are correct — no HDF5 required.
- Both are independently unit-testable with no end-to-end fixture needed.

---

## Consequences

### Positive

- No SRP violation in any component.
- No Dependency Rule violation (Reader never imports domain types).
- Every future dataset reader is structurally identical.
- Mapper logic is isolated, auditable, and independently testable.
- The domain layer (`SensorRecord`) requires no changes as new datasets are added.

### Negative / Risks

- Two milestones instead of one: slightly more upfront structure.
- `RoninRawFrame` is a dataset-specific intermediate type that lives only
  inside the `ronin/` package. It has no value outside that context.
  Mitigation: mark it `__all__`-excluded and document it as an internal type.

---

## Updated Milestone Roadmap

```
M0   Infrastructure                        ✅ Complete
M1.1 Dataset Interfaces                    ✅ Complete
M1.2 RoNIN Adapter                         ✅ Complete
M1.3 RoNIN Metadata Loader                 ✅ Complete
M1.4 RoNIN HDF5 Reader   → RoninRawFrame   ⏳ Next
M1.5 RoNIN Canonical Mapper → SensorRecord ⏳
M1.6 Dataset Utilities                     ⏳
M1.7 EDA & Visualisation                   ⏳
M2   Preprocessing Pipeline                ⏳
M3   Model Development                     ⏳
M4   Navigation / Fusion Engine            ⏳
M5   Evaluation & Benchmarks               ⏳
```

---

## Related Files

| File | Role |
|:---|:---|
| [`architecture.md`](file:///c:/Users/cheta/OneDrive/Desktop/Trinetra-AI%20An%20Intelligent%20Multi-Sensor%20Fusion%20Framework%20for%20GPS-Denied%20Quantum%20Navigation/docs/architecture.md) | Full pipeline diagram (updated) |
| [`ADR-0001`](file:///c:/Users/cheta/OneDrive/Desktop/Trinetra-AI%20An%20Intelligent%20Multi-Sensor%20Fusion%20Framework%20for%20GPS-Denied%20Quantum%20Navigation/docs/adr/0001-dataset-registration.md) | Dataset adapter registration pattern |
| [`sensor_record.py`](file:///c:/Users/cheta/OneDrive/Desktop/Trinetra-AI%20An%20Intelligent%20Multi-Sensor%20Fusion%20Framework%20for%20GPS-Denied%20Quantum%20Navigation/src/trinetra/domain/interfaces/sensor_record.py) | Domain type produced by the Canonical Mapper |
| `ronin/hdf5_reader.py` | To be created in M1.4 |
| `ronin/canonical_mapper.py` | To be created in M1.5 |

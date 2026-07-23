# ADR-0001: Import-Time Side-Effect Registration for Dataset Adapters

**Status:** Accepted
**Date:** 2026-07-23
**Milestone:** M1.2 — RoNIN Dataset Adapter
**Author:** Trinetra-AI Architecture Team
**Reviewers:** Senior Software Architect (code review, 2026-07-23)

---

## Context

The `DatasetRegistry` is a global registry that maps dataset names (e.g. `"ronin_dataset"`) to their concrete adapter classes. The registry instance lives in `src/trinetra/adapters/datasets/registry.py` and is shared across the entire process at runtime.

When the first concrete adapter (`RoninAdapter`) was implemented, a decision was required: **how should adapter classes be registered into this global registry?**

Three mechanisms were considered:

1. **Import-time side effect** — a `registry.register_dataset(...)` call at the bottom of the adapter's module executes automatically when the module is imported.
2. **Explicit registration at application startup** — a dedicated `bootstrap.py` or `configure()` function called from `main.py` lists and registers all known adapters.
3. **Metaclass / decorator auto-registration** — a custom metaclass or `@register` decorator intercepts class creation and registers the class automatically.

---

## Decision

**Option 1 (Import-time side effect) was chosen as the initial implementation**, with a **mandatory guarantee** that the adapter packages are imported from the `adapters/datasets/__init__.py` initializer so the side effect is predictably triggered whenever the datasets package is loaded.

This means:

- `adapter.py` places `registry.register_dataset(_RONIN_NAME, RoninAdapter)` at module scope (the last line of the file).
- `adapters/datasets/__init__.py` imports every adapter subpackage (e.g. `from . import ronin`), so the side effect fires whenever any consumer imports from the datasets package.

---

## Rationale

### Why not Metaclass / Decorator registration (Option 3)?
Metaclass-based approaches introduce invisible coupling between the domain interface and the infrastructure registry. A `DatasetInterface` that auto-registers its subclasses would require it to import from the Adapters layer — a direct violation of the **Dependency Rule**. Decorators carry the same problem if applied at the base class level.

### Why not an explicit bootstrap file (Option 2)?
Explicit bootstrap is cleaner in large systems but introduces a new concern: **every developer must remember to register new adapters** in a central file. This is a form of shotgun surgery risk — adding an adapter in two physically distant places. For a research codebase where adapters will be added frequently, this creates friction.

### Why Import-time side effect (Option 1) is acceptable here
- **Open-Closed Principle**: The registry itself never changes when a new adapter is added.
- **Locality**: Registration logic lives next to the adapter, in the same module. No separate file to update.
- **Predictability**: Because `adapters/datasets/__init__.py` explicitly imports all adapter subpackages, the registration is guaranteed to fire during normal package loading — there is no hidden or accidental trigger.
- **Testability**: Unit tests override the registry or instantiate adapters directly via their class, bypassing the global registry entirely. There is no test pollution.

---

## Consequences

### Positive
- Adding a new dataset adapter requires editing exactly **one** file: `adapters/datasets/__init__.py` (to add the subpackage import). The adapter module itself handles its own registration.
- No central bootstrap file to maintain.
- The registry is always in a consistent state by the time any application code runs.

### Negative / Risks
- **Import ordering**: If an adapter module is imported before its dependencies are available, initialization can fail silently. Mitigation: all adapter dependencies are pure Python standard library.
- **Circular imports**: If an adapter imports from the registry, and the registry's `__init__` imports adapters, circular imports can arise. Mitigation: `registry.py` itself never imports from adapter subpackages.
- **Test isolation**: Global state shared between tests. Mitigation: the registry stores classes, not instances. Registering the same key twice is idempotent. Tests that need a clean registry should use a local `DatasetRegistry()` instance.

---

## Implementation Pattern

All current and future dataset adapters must follow this pattern:

### Step 1 — Adapter module (`adapters/datasets/<name>/adapter.py`)
```python
from trinetra.adapters.datasets.registry import registry

class MyDatasetAdapter(DatasetInterface):
    ...

# Last line of the module — triggers on import
registry.register_dataset("my_dataset", MyDatasetAdapter)
```

### Step 2 — Datasets package init (`adapters/datasets/__init__.py`)
```python
# Import each adapter subpackage so its registration side-effect fires.
from . import ronin       # registers "ronin_dataset"
from . import my_dataset  # registers "my_dataset"
```

### Step 3 — Consumer code
```python
from trinetra.adapters.datasets import registry

AdapterClass = registry.get_dataset("ronin_dataset")
adapter = AdapterClass()
```
The consumer never imports the adapter module directly; it only queries the registry.

---

## Future Consideration

If the number of adapters grows beyond ~10, or if conditional/lazy loading becomes necessary (e.g., adapters with heavy optional dependencies like `h5py`), this decision should be revisited in favor of:

- A lazy-loading plugin registry using `importlib.import_module` with a configuration-driven adapter manifest.
- Or an explicit `entry_points`-based plugin system via `pyproject.toml` if the project becomes a distributed library.

A new ADR should be created at that point.

---

## Related Files

| File | Role |
|:---|:---|
| [`registry.py`](file:///c:/Users/cheta/OneDrive/Desktop/Trinetra-AI%20An%20Intelligent%20Multi-Sensor%20Fusion%20Framework%20for%20GPS-Denied%20Quantum%20Navigation/src/trinetra/adapters/datasets/registry.py) | Global registry instance; never imports adapters |
| [`adapters/datasets/__init__.py`](file:///c:/Users/cheta/OneDrive/Desktop/Trinetra-AI%20An%20Intelligent%20Multi-Sensor%20Fusion%20Framework%20for%20GPS-Denied%20Quantum%20Navigation/src/trinetra/adapters/datasets/__init__.py) | Guarantees adapter imports fire on package load |
| [`ronin/adapter.py`](file:///c:/Users/cheta/OneDrive/Desktop/Trinetra-AI%20An%20Intelligent%20Multi-Sensor%20Fusion%20Framework%20for%20GPS-Denied%20Quantum%20Navigation/src/trinetra/adapters/datasets/ronin/adapter.py) | Reference implementation of the registration pattern |

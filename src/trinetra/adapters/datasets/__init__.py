# ── Adapter subpackage imports ─────────────────────────────────────────────────
# Each import triggers the module-level registry.register_dataset() call
# inside that adapter.  See docs/adr/0001-dataset-registration.md.
from . import ronin  # registers "ronin_dataset"
from .metadata import DatasetMetadata
from .registry import DatasetRegistry, registry

__all__ = ["DatasetMetadata", "DatasetRegistry", "registry", "ronin"]

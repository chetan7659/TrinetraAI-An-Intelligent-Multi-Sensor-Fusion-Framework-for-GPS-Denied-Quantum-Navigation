from trinetra.domain.interfaces.dataset_interface import DatasetInterface


class DatasetRegistry:
    """
    Registry for dataset adapters. Stores metadata and adapter classes.
    Does not instantiate datasets automatically.
    """

    def __init__(self):
        self._registry: dict[str, type[DatasetInterface]] = {}

    def register_dataset(self, name: str, adapter_class: type[DatasetInterface]) -> None:
        """Register a dataset adapter class under a given name."""
        self._registry[name] = adapter_class

    def get_dataset(self, name: str) -> type[DatasetInterface] | None:
        """Retrieve a registered dataset adapter class by its name."""
        return self._registry.get(name)

    def list_datasets(self) -> list[str]:
        """List the names of all registered dataset adapters."""
        return list(self._registry.keys())


# Global registry instance
registry = DatasetRegistry()

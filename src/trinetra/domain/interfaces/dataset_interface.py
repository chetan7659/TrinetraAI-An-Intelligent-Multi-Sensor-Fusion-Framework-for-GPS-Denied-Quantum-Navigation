from abc import ABC, abstractmethod
from typing import Any


class DatasetInterface(ABC):
    """
    Abstract interface for all sensor dataset adapters.
    Defines the contract that every dataset adapter must implement,
    ensuring a unified way to interact with different datasets.
    """

    @abstractmethod
    def name(self) -> str:
        """Return the name of the dataset."""
        pass

    @abstractmethod
    def version(self) -> str:
        """Return the version of the dataset."""
        pass

    @abstractmethod
    def sensor_types(self) -> list[str]:
        """Return a list of sensor types available in this dataset."""
        pass

    @abstractmethod
    def sampling_rate(self) -> float:
        """Return the base sampling rate of the dataset in Hz."""
        pass

    @abstractmethod
    def license(self) -> str:
        """Return the license of the dataset."""
        pass

    @abstractmethod
    def download_url(self) -> str:
        """Return the URL where the dataset can be downloaded."""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate the integrity and completeness of the dataset."""
        pass

    @abstractmethod
    def load_metadata(self) -> Any:
        """Load and return the metadata for the dataset."""
        pass

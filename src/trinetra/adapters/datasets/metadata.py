from dataclasses import dataclass


@dataclass
class DatasetMetadata:
    """
    Dataclass describing metadata for a sensor dataset.
    """

    name: str
    version: str
    description: str
    license: str
    homepage: str
    sensor_types: list[str]
    sampling_rate: float
    ground_truth_available: bool
    supported_tasks: list[str]

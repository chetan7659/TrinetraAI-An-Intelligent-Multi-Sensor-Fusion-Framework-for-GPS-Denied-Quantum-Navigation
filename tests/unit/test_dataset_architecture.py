from typing import Any

import pytest

from trinetra.adapters.datasets.metadata import DatasetMetadata
from trinetra.adapters.datasets.registry import DatasetRegistry
from trinetra.domain.interfaces.dataset_interface import DatasetInterface
from trinetra.domain.interfaces.sensor_record import SensorRecord


def test_dataset_interface_cannot_be_instantiated():
    with pytest.raises(TypeError):
        DatasetInterface()


def test_sensor_record_dataclass():
    record = SensorRecord(
        timestamp=12345.67,
        sensor_type="imu",
        values=[0.1, 0.2, 0.3],
        frame_id=1,
        metadata={"unit": "m/s^2"},
    )
    assert record.timestamp == 12345.67
    assert record.sensor_type == "imu"
    assert record.values == [0.1, 0.2, 0.3]
    assert record.frame_id == 1
    assert record.metadata["unit"] == "m/s^2"


def test_dataset_metadata_dataclass():
    metadata = DatasetMetadata(
        name="TestDataset",
        version="1.0",
        description="A test dataset",
        license="MIT",
        homepage="http://example.com",
        sensor_types=["imu", "gnss"],
        sampling_rate=100.0,
        ground_truth_available=True,
        supported_tasks=["navigation"],
    )
    assert metadata.name == "TestDataset"
    assert "imu" in metadata.sensor_types
    assert metadata.sampling_rate == 100.0


class DummyDatasetAdapter(DatasetInterface):
    def name(self) -> str:
        return "Dummy"

    def version(self) -> str:
        return "1.0"

    def sensor_types(self) -> list[str]:
        return ["dummy"]

    def sampling_rate(self) -> float:
        return 1.0

    def license(self) -> str:
        return "MIT"

    def download_url(self) -> str:
        return ""

    def validate(self) -> bool:
        return True

    def load_metadata(self) -> Any:
        return None


def test_registry_can_register_and_return_adapters():
    registry = DatasetRegistry()

    # Check that initially it's empty or does not contain 'dummy'
    assert registry.get_dataset("dummy") is None

    registry.register_dataset("dummy", DummyDatasetAdapter)

    assert "dummy" in registry.list_datasets()
    assert registry.get_dataset("dummy") == DummyDatasetAdapter
    assert registry.get_dataset("nonexistent") is None

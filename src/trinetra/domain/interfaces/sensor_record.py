from dataclasses import dataclass
from typing import Any


@dataclass
class SensorRecord:
    """
    Lightweight dataclass representing a single sensor reading or frame.
    Placeholder for processing logic, only containing data fields.
    """

    timestamp: float
    sensor_type: str
    values: Any
    frame_id: int
    metadata: dict[str, Any]

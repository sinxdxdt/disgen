"""Core module initialization."""

from disgen.core.base_model import (
    BaseModel,
    DiscriminativeModel,
    GenerativeModel,
    ModelRegistry,
)
from disgen.core.base_dataset import (
    BaseDataset,
    Track,
    DatasetRegistry,
)
from disgen.core.base_metric import (
    BaseMetric,
    MetricRegistry,
)

__all__ = [
    "BaseModel",
    "DiscriminativeModel",
    "GenerativeModel",
    "ModelRegistry",
    "BaseDataset",
    "Track",
    "DatasetRegistry",
    "BaseMetric",
    "MetricRegistry",
]

"""Datasets module."""

from disgen.datasets.mock import MockDataset

# Real dataset implementations
try:
    from disgen.datasets.musdb18hq import MUSDB18HQ
except ImportError:
    MUSDB18HQ = None

try:
    from disgen.datasets.slakh2100 import Slakh2100
except ImportError:
    Slakh2100 = None

try:
    from disgen.datasets.moisesdb import MoisesDB
except ImportError:
    MoisesDB = None

__all__ = [
    "MockDataset",
    "MUSDB18HQ",
    "Slakh2100",
    "MoisesDB",
]

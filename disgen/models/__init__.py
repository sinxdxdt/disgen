"""Models module initialization."""

from disgen.models.discriminative import MockDiscriminativeModel, HTDemucs
from disgen.models.generative import MockGenerativeModel, HiFiGAN

__all__ = [
    "MockDiscriminativeModel",
    "HTDemucs",
    "MockGenerativeModel",
    "HiFiGAN",
]

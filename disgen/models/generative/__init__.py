"""Generative models module."""

from disgen.models.generative.mock import MockGenerativeModel

# Real generative model implementations
try:
    from disgen.models.generative.hifigan import HiFiGAN
except ImportError:
    HiFiGAN = None

__all__ = [
    "MockGenerativeModel",
    "HiFiGAN",
]

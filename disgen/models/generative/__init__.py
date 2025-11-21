"""Generative models module."""

from disgen.models.generative.mock import MockGenerativeModel

# Real generative model implementations
try:
    from disgen.models.generative.hifigan import HiFiGAN
except ImportError:
    HiFiGAN = None

try:
    from disgen.models.generative.discoder import DisCoder
except ImportError:
    DisCoder = None

__all__ = [
    "MockGenerativeModel",
    "HiFiGAN",
    "DisCoder",
]

"""Discriminative models module."""

from disgen.models.discriminative.mock import MockDiscriminativeModel

# Real discriminative model implementations
try:
    from disgen.models.discriminative.htdemucs import HTDemucs
except ImportError:
    HTDemucs = None

__all__ = [
    "MockDiscriminativeModel",
    "HTDemucs",
]

"""
disgen - Music Source Separation Evaluation Library

A comprehensive framework for evaluating generative refinement of 
discriminative music source separation models.
"""

__version__ = "0.1.0"
__author__ = "disgen contributors"

from disgen.core.base_model import BaseModel, DiscriminativeModel, GenerativeModel
from disgen.core.base_dataset import BaseDataset
from disgen.core.base_metric import BaseMetric
from disgen.pipeline.evaluator import Evaluator

__all__ = [
    "BaseModel",
    "DiscriminativeModel", 
    "GenerativeModel",
    "BaseDataset",
    "BaseMetric",
    "Evaluator",
    "__version__",
]

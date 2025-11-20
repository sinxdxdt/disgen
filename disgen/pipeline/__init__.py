"""Pipeline module initialization."""

from disgen.pipeline.evaluator import Evaluator
from disgen.pipeline.statistics import StatisticalTests, compute_model_significance
from disgen.pipeline.reporter import Reporter

__all__ = [
    "Evaluator",
    "StatisticalTests",
    "compute_model_significance",
    "Reporter",
]

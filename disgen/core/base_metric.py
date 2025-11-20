"""Base metric classes and registry."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import numpy as np


class MetricRegistry:
    """Registry for managing metric implementations."""
    
    _metrics: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, tags: Optional[List[str]] = None):
        """Decorator to register a metric class.
        
        Args:
            name: Primary name for the metric
            tags: Optional additional tags/aliases
        """
        def decorator(metric_cls):
            cls._metrics[name.lower()] = metric_cls
            if tags:
                for tag in tags:
                    cls._metrics[tag.lower()] = metric_cls
            return metric_cls
        return decorator
    
    @classmethod
    def get(cls, name: str) -> type:
        """Get a metric class by name or tag.
        
        Args:
            name: Metric name or tag
            
        Returns:
            Metric class
            
        Raises:
            KeyError: If metric not found
        """
        key = name.lower()
        if key not in cls._metrics:
            raise KeyError(
                f"Metric '{name}' not found in registry. "
                f"Available metrics: {list(cls._metrics.keys())}"
            )
        return cls._metrics[key]
    
    @classmethod
    def list_metrics(cls) -> List[str]:
        """List all registered metric names."""
        return list(cls._metrics.keys())


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics."""
    
    def __init__(self, **kwargs):
        """Initialize the metric.
        
        Args:
            **kwargs: Metric-specific configuration
        """
        self.config = kwargs
    
    @abstractmethod
    def compute(
        self,
        estimated: np.ndarray,
        reference: np.ndarray,
        sample_rate: int,
        **kwargs
    ) -> Dict[str, float]:
        """Compute the metric.
        
        Args:
            estimated: Estimated/separated audio
            reference: Ground truth reference audio
            sample_rate: Sample rate in Hz
            **kwargs: Additional metric-specific parameters
            
        Returns:
            Dictionary of metric values. Keys are metric names, values are floats.
            For metrics that compute multiple values (e.g., SDR, SIR, SAR),
            return all of them.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the metric name."""
        pass
    
    @property
    @abstractmethod
    def higher_is_better(self) -> bool:
        """Return True if higher metric values indicate better quality."""
        pass
    
    def aggregate(
        self,
        results: List[Dict[str, float]],
        method: str = "mean"
    ) -> Dict[str, float]:
        """Aggregate metric results across multiple tracks.
        
        Args:
            results: List of metric dictionaries from individual tracks
            method: Aggregation method ('mean', 'median')
            
        Returns:
            Aggregated metric dictionary
        """
        if not results:
            return {}
        
        # Get all metric keys
        keys = set()
        for result in results:
            keys.update(result.keys())
        
        aggregated = {}
        for key in keys:
            values = [r[key] for r in results if key in r and not np.isnan(r[key])]
            if not values:
                aggregated[key] = np.nan
            elif method == "mean":
                aggregated[key] = np.mean(values)
            elif method == "median":
                aggregated[key] = np.median(values)
            else:
                raise ValueError(f"Unknown aggregation method: {method}")
        
        return aggregated

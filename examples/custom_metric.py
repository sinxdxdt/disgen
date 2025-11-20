"""Example: Implementing a custom metric."""

import numpy as np
from typing import Dict

from disgen.core.base_metric import BaseMetric, MetricRegistry


@MetricRegistry.register("custom_metric", tags=["example", "rmse"])
class CustomMetric(BaseMetric):
    """Example custom metric: Root Mean Square Error.
    
    This demonstrates how to implement a custom evaluation metric
    that can be used with disgen.
    """
    
    def __init__(self, normalize: bool = True, **kwargs):
        """Initialize the metric.
        
        Args:
            normalize: Whether to normalize by reference RMS
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.normalize = normalize
    
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
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with metric values
        """
        # Ensure same length
        min_len = min(len(estimated.flat), len(reference.flat))
        est_flat = estimated.flat[:min_len]
        ref_flat = reference.flat[:min_len]
        
        # Compute RMSE
        mse = np.mean((est_flat - ref_flat) ** 2)
        rmse = np.sqrt(mse)
        
        # Optional normalization
        if self.normalize:
            ref_rms = np.sqrt(np.mean(ref_flat ** 2))
            if ref_rms > 0:
                rmse = rmse / ref_rms
        
        # Can return multiple related metrics
        return {
            "rmse": float(rmse),
            "mse": float(mse),
        }
    
    @property
    def name(self) -> str:
        """Return the metric name."""
        return "custom_metric"
    
    @property
    def higher_is_better(self) -> bool:
        """Indicate if higher values are better.
        
        For RMSE, lower is better.
        """
        return False


@MetricRegistry.register("snr_metric", tags=["snr_custom"])
class SNRMetric(BaseMetric):
    """Signal-to-Noise Ratio metric example."""
    
    def compute(
        self,
        estimated: np.ndarray,
        reference: np.ndarray,
        sample_rate: int,
        **kwargs
    ) -> Dict[str, float]:
        """Compute SNR."""
        # Ensure same length
        min_len = min(len(estimated.flat), len(reference.flat))
        est_flat = estimated.flat[:min_len]
        ref_flat = reference.flat[:min_len]
        
        # Compute noise
        noise = est_flat - ref_flat
        
        # Compute powers
        signal_power = np.mean(ref_flat ** 2)
        noise_power = np.mean(noise ** 2)
        
        # Compute SNR in dB
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = float('inf')
        
        return {"snr": float(snr)}
    
    @property
    def name(self) -> str:
        return "snr_metric"
    
    @property
    def higher_is_better(self) -> bool:
        return True  # Higher SNR is better


def main():
    """Example usage of custom metrics."""
    
    # Create metric instances
    rmse_metric = CustomMetric(normalize=True)
    snr_metric = SNRMetric()
    
    # Test metrics
    reference = np.random.randn(16000)
    estimated = reference + np.random.randn(16000) * 0.1  # Add noise
    
    rmse_result = rmse_metric.compute(estimated, reference, 16000)
    snr_result = snr_metric.compute(estimated, reference, 16000)
    
    print(f"RMSE: {rmse_result}")
    print(f"SNR: {snr_result}")
    
    # Use with evaluator
    from disgen.models.discriminative import MockDiscriminativeModel
    from disgen.datasets import MockDataset
    from disgen.pipeline import Evaluator
    
    model = MockDiscriminativeModel()
    dataset = MockDataset(num_tracks=3)
    metrics = [rmse_metric, snr_metric]
    
    evaluator = Evaluator(
        discriminative_model=model,
        generative_model=None,
        dataset=dataset,
        metrics=metrics
    )
    
    results = evaluator.evaluate(max_tracks=3)
    
    print("\nAggregated results:")
    print(results['summary']['aggregated_metrics'])


if __name__ == "__main__":
    main()

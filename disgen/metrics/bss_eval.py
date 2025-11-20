"""BSS Eval metrics (SDR, SIR, SAR, SNR) using museval."""

import numpy as np
from typing import Dict
import warnings

from disgen.core.base_metric import BaseMetric, MetricRegistry

try:
    import museval
    MUSEVAL_AVAILABLE = True
except ImportError:
    MUSEVAL_AVAILABLE = False
    warnings.warn(
        "museval not installed. Install with: pip install museval"
    )


@MetricRegistry.register("bss_eval", tags=["sdr", "sir", "sar", "snr"])
class BSSEvalMetrics(BaseMetric):
    """BSS Eval metrics using museval library.
    
    Computes SDR (Source-to-Distortion Ratio), SIR (Source-to-Interference Ratio),
    SAR (Source-to-Artifact Ratio), and SNR (Signal-to-Noise Ratio).
    """
    
    def __init__(self, window: int = 44100, hop: int = 44100, **kwargs):
        """Initialize BSS Eval metrics.
        
        Args:
            window: Window size for framewise evaluation (samples)
            hop: Hop size for framewise evaluation (samples)
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        if not MUSEVAL_AVAILABLE:
            raise ImportError(
                "museval is required for BSS Eval metrics. "
                "Install with: pip install museval"
            )
        self.window = window
        self.hop = hop
    
    def compute(
        self,
        estimated: np.ndarray,
        reference: np.ndarray,
        sample_rate: int,
        **kwargs
    ) -> Dict[str, float]:
        """Compute BSS Eval metrics.
        
        Args:
            estimated: Estimated audio, shape (channels, samples) or (samples,)
            reference: Reference audio, shape (channels, samples) or (samples,)
            sample_rate: Sample rate in Hz
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with 'sdr', 'sir', 'sar', 'snr' keys
        """
        # Ensure arrays are 2D (channels, samples)
        if estimated.ndim == 1:
            estimated = estimated[np.newaxis, :]
        if reference.ndim == 1:
            reference = reference[np.newaxis, :]
        
        # Ensure both have same shape
        if estimated.shape != reference.shape:
            # Truncate to shorter length
            min_len = min(estimated.shape[1], reference.shape[1])
            estimated = estimated[:, :min_len]
            reference = reference[:, :min_len]
        
        # Convert to (nsrc=1, nsampl, nchan) format for museval
        # museval expects (nsrc, nsampl, nchan)
        estimated_museval = estimated.T[np.newaxis, :, :]  # (1, samples, channels)
        reference_museval = reference.T[np.newaxis, :, :]  # (1, samples, channels)
        
        try:
            # Compute BSS eval metrics
            scores = museval.evaluate(
                reference_museval,
                estimated_museval,
                win=self.window,
                hop=self.hop
            )
            
            # Extract median values across frames
            return {
                "sdr": float(np.nanmedian(scores[0])),  # SDR
                "sir": float(np.nanmedian(scores[1])),  # SIR
                "sar": float(np.nanmedian(scores[2])),  # SAR
                "snr": float(np.nanmedian(scores[3])),  # ISR (called SNR in some contexts)
            }
        except Exception as e:
            warnings.warn(f"Error computing BSS Eval metrics: {e}")
            return {
                "sdr": np.nan,
                "sir": np.nan,
                "sar": np.nan,
                "snr": np.nan,
            }
    
    @property
    def name(self) -> str:
        return "bss_eval"
    
    @property
    def higher_is_better(self) -> bool:
        return True

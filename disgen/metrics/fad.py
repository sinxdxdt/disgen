"""Frechet Audio Distance (FAD) metric."""

import numpy as np
from typing import Dict, List
import warnings
import tempfile
import os

from disgen.core.base_metric import BaseMetric, MetricRegistry

try:
    import fadtk
    FADTK_AVAILABLE = True
except ImportError:
    FADTK_AVAILABLE = False
    warnings.warn(
        "fadtk not installed. Install with: pip install fadtk"
    )

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    warnings.warn("soundfile not installed. Install with: pip install soundfile")


@MetricRegistry.register("fad", tags=["frechet_audio_distance"])
class FrechetAudioDistance(BaseMetric):
    """Frechet Audio Distance using fadtk library.
    
    FAD measures perceptual similarity by computing distance between
    embedding distributions of reference and estimated audio.
    """
    
    def __init__(
        self,
        model_name: str = "vggish",
        **kwargs
    ):
        """Initialize FAD metric.
        
        Args:
            model_name: Embedding model to use ('vggish', 'clap', 'mert', etc.)
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        if not FADTK_AVAILABLE:
            raise ImportError(
                "fadtk is required for FAD metric. "
                "Install with: pip install fadtk"
            )
        if not SOUNDFILE_AVAILABLE:
            raise ImportError("soundfile is required. Install with: pip install soundfile")
        
        self.model_name = model_name
        # Store samples for batch computation
        self._estimated_samples: List[np.ndarray] = []
        self._reference_samples: List[np.ndarray] = []
        self._sample_rate: int = 16000  # Default, will be updated
    
    def compute(
        self,
        estimated: np.ndarray,
        reference: np.ndarray,
        sample_rate: int,
        **kwargs
    ) -> Dict[str, float]:
        """Compute FAD between estimated and reference.
        
        Note: FAD is typically computed over a batch of samples.
        This method stores samples and returns NaN until compute_batch is called.
        
        Args:
            estimated: Estimated audio
            reference: Reference audio
            sample_rate: Sample rate in Hz
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with 'fad' key (NaN for single samples)
        """
        # Store samples for batch computation
        self._estimated_samples.append(estimated)
        self._reference_samples.append(reference)
        self._sample_rate = sample_rate
        
        # Return NaN as FAD needs batch computation
        return {"fad": np.nan}
    
    def compute_batch(self) -> Dict[str, float]:
        """Compute FAD over all stored samples.
        
        Returns:
            Dictionary with 'fad' key
        """
        if not self._estimated_samples or not self._reference_samples:
            return {"fad": np.nan}
        
        # Create temporary directories for audio files
        with tempfile.TemporaryDirectory() as est_dir, \
             tempfile.TemporaryDirectory() as ref_dir:
            
            # Save estimated samples
            for i, audio in enumerate(self._estimated_samples):
                audio_mono = self._to_mono(audio)
                sf.write(
                    os.path.join(est_dir, f"sample_{i:04d}.wav"),
                    audio_mono,
                    self._sample_rate
                )
            
            # Save reference samples
            for i, audio in enumerate(self._reference_samples):
                audio_mono = self._to_mono(audio)
                sf.write(
                    os.path.join(ref_dir, f"sample_{i:04d}.wav"),
                    audio_mono,
                    self._sample_rate
                )
            
            try:
                # Compute FAD
                fad_score = fadtk.compute_fad(
                    ref_dir,
                    est_dir,
                    model_name=self.model_name,
                    dtype="float32"
                )
                
                # Clear stored samples
                self._estimated_samples = []
                self._reference_samples = []
                
                return {"fad": float(fad_score)}
            except Exception as e:
                warnings.warn(f"Error computing FAD: {e}")
                return {"fad": np.nan}
    
    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        """Convert audio to mono.
        
        Args:
            audio: Audio array, shape (channels, samples) or (samples,)
            
        Returns:
            Mono audio, shape (samples,)
        """
        if audio.ndim == 1:
            return audio
        # Average across channels
        return np.mean(audio, axis=0)
    
    @property
    def name(self) -> str:
        return "fad"
    
    @property
    def higher_is_better(self) -> bool:
        return False  # Lower FAD is better

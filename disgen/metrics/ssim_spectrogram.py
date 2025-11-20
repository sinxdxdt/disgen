"""SSIM on spectrograms metric."""

import numpy as np
from typing import Dict
import warnings

from disgen.core.base_metric import BaseMetric, MetricRegistry

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    warnings.warn("librosa not installed. Install with: pip install librosa")

try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    warnings.warn("scikit-image not installed. Install with: pip install scikit-image")


@MetricRegistry.register("ssim", tags=["ssim_spectrogram"])
class SSIMSpectrogram(BaseMetric):
    """Structural Similarity Index Measure on spectrograms.
    
    Converts audio to mel-spectrograms and computes SSIM between them.
    """
    
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        **kwargs
    ):
        """Initialize SSIM metric.
        
        Args:
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel bands
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        if not LIBROSA_AVAILABLE:
            raise ImportError(
                "librosa is required for SSIM metric. "
                "Install with: pip install librosa"
            )
        if not SKIMAGE_AVAILABLE:
            raise ImportError(
                "scikit-image is required for SSIM metric. "
                "Install with: pip install scikit-image"
            )
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
    
    def compute(
        self,
        estimated: np.ndarray,
        reference: np.ndarray,
        sample_rate: int,
        **kwargs
    ) -> Dict[str, float]:
        """Compute SSIM on mel-spectrograms.
        
        Args:
            estimated: Estimated audio, shape (channels, samples) or (samples,)
            reference: Reference audio, shape (channels, samples) or (samples,)
            sample_rate: Sample rate in Hz
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with 'ssim' key
        """
        # Convert to mono
        est_mono = self._to_mono(estimated)
        ref_mono = self._to_mono(reference)
        
        # Ensure same length
        min_len = min(len(est_mono), len(ref_mono))
        est_mono = est_mono[:min_len]
        ref_mono = ref_mono[:min_len]
        
        try:
            # Compute mel-spectrograms
            est_mel = librosa.feature.melspectrogram(
                y=est_mono,
                sr=sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            ref_mel = librosa.feature.melspectrogram(
                y=ref_mono,
                sr=sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            
            # Convert to dB scale
            est_mel_db = librosa.power_to_db(est_mel, ref=np.max)
            ref_mel_db = librosa.power_to_db(ref_mel, ref=np.max)
            
            # Normalize to [0, 1] range for SSIM
            est_norm = self._normalize(est_mel_db)
            ref_norm = self._normalize(ref_mel_db)
            
            # Compute SSIM
            ssim_value = ssim(
                ref_norm,
                est_norm,
                data_range=1.0
            )
            
            return {"ssim": float(ssim_value)}
        except Exception as e:
            warnings.warn(f"Error computing SSIM: {e}")
            return {"ssim": np.nan}
    
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
    
    @staticmethod
    def _normalize(data: np.ndarray) -> np.ndarray:
        """Normalize data to [0, 1] range.
        
        Args:
            data: Input array
            
        Returns:
            Normalized array
        """
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max - data_min == 0:
            return np.zeros_like(data)
        return (data - data_min) / (data_max - data_min)
    
    @property
    def name(self) -> str:
        return "ssim"
    
    @property
    def higher_is_better(self) -> bool:
        return True  # Higher SSIM is better

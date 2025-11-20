"""Mock generative model for testing purposes."""

import numpy as np
from typing import List, Optional

from disgen.core.base_model import GenerativeModel, ModelRegistry


@ModelRegistry.register("mock_generative", tags=["mock_gen"])
class MockGenerativeModel(GenerativeModel):
    """Mock generative model for testing without real model dependencies."""
    
    def __init__(
        self,
        modes: Optional[List[str]] = None,
        improvement_factor: float = 1.1,
        **kwargs
    ):
        """Initialize mock model.
        
        Args:
            modes: Supported modes (default: ['refinement', 'vocoder'])
            improvement_factor: Factor to multiply audio by to simulate improvement
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self._modes = modes or [self.MODE_REFINEMENT, self.MODE_VOCODER]
        self.improvement_factor = improvement_factor
        self._is_loaded = False
    
    def load(self):
        """Load model (mock - does nothing)."""
        self._is_loaded = True
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    def refine(
        self,
        audio: np.ndarray,
        sample_rate: int,
        stem_name: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """Mock refinement - slightly modifies audio.
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            stem_name: Optional stem  name
            **kwargs: Additional parameters
            
        Returns:
            "Refined" audio (with small modifications)
        """
        # Simulate refinement by applying small gain and reducing noise
        refined = audio * self.improvement_factor
        # Add tiny smoothing to simulate denoising
        if refined.ndim == 1:
            refined[1:-1] = (refined[:-2] + refined[1:-1] + refined[2:]) / 3.0
        else:
            refined[:, 1:-1] = (refined[:, :-2] + refined[:, 1:-1] + refined[:, 2:]) / 3.0
        return refined
    
    def vocode(
        self,
        spectrogram: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Mock vocoding - returns random audio based on spectrogram shape.
        
        Args:
            spectrogram: Input spectrogram
            **kwargs: Additional parameters
            
        Returns:
            Mock waveform
        """
        # Generate mock audio based on spectrogram dimensions
        # Assume spectrogram is (freq_bins, time_frames)
        if spectrogram.ndim == 2:
            n_samples = spectrogram.shape[1] * 512  # Assuming hop_length=512
            waveform = np.random.randn(n_samples) * 0.1
        else:
            waveform = np.random.randn(16000) * 0.1  # 1 second at 16kHz
        
        return waveform
    
    @property
    def supported_modes(self) -> List[str]:
        return self._modes

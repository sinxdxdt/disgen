"""HTDemucs v4 implementation.

Installation required:
    pip install demucs torch torchaudio

Usage:
    from disgen.models.discriminative import HTDemucs
    model = HTDemucs(model_name='htdemucs_ft', device='cuda')
    model.load()
    separated = model.separate(mixture, sample_rate=44100)
"""

import warnings
import numpy as np
from typing import Dict, List, Optional

from disgen.core.base_model import DiscriminativeModel, ModelRegistry
from disgen.utils.device import get_device

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from demucs import pretrained
    from demucs.apply import apply_model
    from demucs.audio import convert_audio
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False


@ModelRegistry.register("htdemucs", tags=["demucs", "htdemucs_v4", "htdemucs_ft"])
class HTDemucs(DiscriminativeModel):
    """HTDemucs v4 (Hybrid Transformer Demucs) model wrapper.
    
    HTDemucs is a hybrid spectrogram/waveform separation model with:
    - Transformer layers for long-range dependencies
    - Convolutional encoder-decoder for time-domain processing
    - State-of-the-art performance on MUSDB18
    
    Args:
        model_name: Model checkpoint name (default: 'htdemucs_ft')
            Options: 'htdemucs', 'htdemucs_ft', 'htdemucs_6s'
        device: Device to run on ('cpu', 'cuda', 'mps', or None for auto)
        segment: Segment length in seconds for processing (default: None for full track)
        overlap: Overlap between segments (default: 0.25)
        **kwargs: Additional arguments
    """
    
    def __init__(
        self,
        model_name: str = "htdemucs_ft",
        device: Optional[str] = None,
        segment: Optional[float] = None,
        overlap: float = 0.25,
        **kwargs
    ):
        """Initialize HTDemucs model."""
        device = get_device(device)
        super().__init__(device=device, **kwargs)
        
        if not TORCH_AVAILABLE or not DEMUCS_AVAILABLE:
            raise ImportError(
                "HTDemucs requires torch and demucs. Install with:\n"
                "  pip install torch torchaudio demucs"
            )
        
        self.model_name = model_name
        self.segment = segment
        self.overlap = overlap
        self.model = None
        self._stems = ['drums', 'bass', 'other', 'vocals']  # Default order
    
    def load(self):
        """Load HTDemucs model from pretrained checkpoints."""
        print(f"Loading HTDemucs model: {self.model_name}")
        
        try:
            self.model = pretrained.get_model(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Get stem names from model
            if hasattr(self.model, 'sources'):
                self._stems = list(self.model.sources)
            
            print(f"HTDemucs loaded on {self.device}")
            print(f"Available stems: {self._stems}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load HTDemucs: {e}")
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def separate(
        self,
        mixture: np.ndarray,
        sample_rate: int,
        stem: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """Separate mixture using HTDemucs.
        
        Args:
            mixture: Audio mixture, shape (channels, samples) or (samples,)
            sample_rate: Sample rate in Hz
            stem: Optional specific stem to extract (returns all if None)
            
        Returns:
            Dictionary mapping stem names to separated audio
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Ensure correct shape (channels, samples)
        if mixture.ndim == 1:
            mixture = mixture[np.newaxis, :]  # (samples,) -> (1, samples)
        
        # Convert to torch tensor
        mixture_tensor = torch.from_numpy(mixture).float().to(self.device)
        
        # Convert sample rate if needed
        model_sr = self.model.samplerate
        if sample_rate != model_sr:
            mixture_tensor = convert_audio(
                mixture_tensor,
                sample_rate,
                model_sr,
                self.model.audio_channels
            )
            original_sr = sample_rate
            sample_rate = model_sr
        else:
            original_sr = None
        
        # Add batch dimension
        mixture_tensor = mixture_tensor.unsqueeze(0)  # (1, channels, samples)
        
        # Apply model
        with torch.no_grad():
            if self.segment is not None:
                # Segment processing for long tracks
                segment_samples = int(self.segment * sample_rate)
                sources = apply_model(
                    self.model,
                    mixture_tensor,
                    segment=segment_samples,
                    overlap=self.overlap,
                    device=self.device
                )
            else:
                # Full track processing
                sources = self.model(mixture_tensor)
        
        # Convert back to numpy
        # sources shape: (batch, stems, channels, samples)
        sources = sources.squeeze(0).cpu().numpy()  # (stems, channels, samples)
        
        # Convert back to original sample rate if needed
        if original_sr is not None:
            from disgen.utils.audio import resample_audio
            sources_resampled = []
            for i in range(sources.shape[0]):
                resampled = resample_audio(sources[i], model_sr, original_sr)
                sources_resampled.append(resampled)
            sources = np.array(sources_resampled)
        
        # Build result dictionary
        results = {}
        for i, stem_name in enumerate(self._stems):
            if stem is None or stem == stem_name:
                results[stem_name] = sources[i]
        
        return results
    
    @property
    def domain(self) -> str:
        """HTDemucs is hybrid but outputs in time domain."""
        return self.DOMAIN_TIME
    
    @property
    def available_stems(self) -> List[str]:
        """Get available stem names."""
        return self._stems

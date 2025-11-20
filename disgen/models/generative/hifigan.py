"""HiFiGAN vocoder implementation.

Installation required:
    pip install torch torchaudio
    # Download pretrained weights from https://github.com/jik876/hifi-gan

Usage:
    from disgen.models.generative import HiFiGAN
    model = HiFiGAN(checkpoint_path='/path/to/generator', device='cuda')
    model.load()
    waveform = model.vocode(mel_spectrogram)
"""

import json
import warnings
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict

from disgen.core.base_model import GenerativeModel, ModelRegistry
from disgen.utils.device import get_device

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not installed. GPU acceleration unavailable.")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


@ModelRegistry.register("hifigan", tags=["vocoder", "hifi_gan"])
class HiFiGAN(GenerativeModel):
    """HiFiGAN vocoder for mel-spectrogram to waveform synthesis.
    
    HiFiGAN is a GAN-based neural vocoder that generates high-fidelity audio
    from mel-spectrograms.
    
    Args:
        checkpoint_path: Path to generator checkpoint (.pth file)
        config_path: Path to config JSON file (optional if in same dir as checkpoint)
        device: Device to run on ('cpu', 'cuda', 'mps', or None for auto)
        **kwargs: Additional arguments
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ):
        """Initialize HiFiGAN vocoder."""
        device = get_device(device)
        super().__init__(device=device, **kwargs)
        
        if not TORCH_AVAILABLE:
            raise ImportError("HiFiGAN requires PyTorch. Install with: pip install torch")
        
        self.checkpoint_path = Path(checkpoint_path)
        
        # Try to find config in same directory if not specified
        if config_path is None:
            config_path = self.checkpoint_path.parent / "config.json"
        self.config_path = Path(config_path)
        
        self.generator = None
        self.config = None
        self._modes = [self.MODE_VOCODER, self.MODE_REFINEMENT]
    
    def load(self):
        """Load HiFiGAN model and config."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}. "
                f"Download from https://github.com/jik876/hifi-gan"
            )
        
        # Load config
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            warnings.warn(f"Config not found: {self.config_path}. Using defaults.")
            self.config = self._get_default_config()
        
        print(f"Loading HiFiGAN from {self.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Create generator (this is a simplified version - in practice you'd import from HiFiGAN repo)
        # For now, we'll create a placeholder that can load the state dict
        if 'generator' in checkpoint:
            state_dict = checkpoint['generator']
        else:
            state_dict = checkpoint
        
        # In a real implementation, you would instantiate the actual HiFiGAN generator
        # from the HiFiGAN repository and load the state dict
        # For now, we'll use a placeholder
        self.generator = torch.nn.Module()  # Placeholder
        
        # Load state dict (in practice)
        # self.generator.load_state_dict(state_dict)
        # self.generator.to(self.device)
        # self.generator.eval()
        
        warnings.warn(
            "HiFiGAN implementation is a template. "
            "Please integrate with actual HiFiGAN repository code."
        )
        
        print(f"HiFiGAN loaded on {self.device}")
    
    def _get_default_config(self) -> Dict:
        """Get default HiFi config."""
        return {
            "sampling_rate": 22050,
            "n_fft": 1024,
            "hop_size": 256,
            "win_size": 1024,
            "n_mel_channels": 80,
            "mel_fmin": 0.0,
            "mel_fmax": 8000.0,
        }
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.generator is not None
    
    def refine(
        self,
        audio: np.ndarray,
        sample_rate: int,
        stem_name: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """Refine audio using analysis-synthesis with HiFiGAN.
        
        Args:
            audio: Input audio waveform
            sample_rate: Sample rate in Hz
            stem_name: Optional stem name
            **kwargs: Additional parameters
            
        Returns:
            Refined audio waveform
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if not LIBROSA_AVAILABLE:
            raise RuntimeError("librosa required for mel-spectrogram extraction")
        
        # Extract mel-spectrogram
        mel_spec = self._extract_mel(audio, sample_rate)
        
        # Vocode
        refined = self.vocode(mel_spec, **kwargs)
        
        return refined
    
    def vocode(
        self,
        spectrogram: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Convert mel-spectrogram to waveform.
        
        Args:
            spectrogram: Mel-spectrogram, shape (n_mels, time)
            **kwargs: Additional parameters
            
        Returns:
            Audio waveform
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Convert to tensor
        mel_tensor = torch.from_numpy(spectrogram).float().to(self.device)
        
        # Add batch dimension if needed
        if mel_tensor.dim() == 2:
            mel_tensor = mel_tensor.unsqueeze(0)
        
        # Generate waveform
        with torch.no_grad():
            # In practice: audio = self.generator(mel_tensor)
            # For now, return placeholder
            warnings.warn("HiFiGAN vocoding not fully implemented - returning silence")
            hop_size = self.config.get('hop_size', 256)
            n_samples = spectrogram.shape[1] * hop_size
            audio = np.zeros(n_samples, dtype=np.float32)
        
        return audio
    
    def _extract_mel(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Extract mel spectrogram from audio.
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate
            
        Returns:
            Mel spectrogram
        """
        if not LIBROSA_AVAILABLE:
            raise RuntimeError("librosa required")
        
        # Convert to mono if stereo
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)
        
        # Resample if needed
        target_sr = self.config.get('sampling_rate', 22050)
        if sample_rate != target_sr:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=target_sr,
            n_fft=self.config.get('n_fft', 1024),
            hop_length=self.config.get('hop_size', 256),
            win_length=self.config.get('win_size', 1024),
            n_mels=self.config.get('n_mel_channels', 80),
            fmin=self.config.get('mel_fmin', 0.0),
            fmax=self.config.get('mel_fmax', 8000.0),
        )
        
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec)
        
        return mel_spec
    
    @property
    def supported_modes(self) -> List[str]:
        """Get supported modes."""
        return self._modes

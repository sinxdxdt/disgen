"""DisCoder neural vocoder implementation.

DisCoder is a high-fidelity music vocoder using neural audio codecs.
Paper: https://arxiv.org/abs/2502.12759
Model: https://huggingface.co/disco-eth/discoder

DisCoder is vendored in disgen.vendor.discoder - no separate installation needed!

Usage:
    from disgen.models.generative import DisCoder
    
    # Use pretrained model from Hugging Face
    model = DisCoder(use_pretrained=True, device='cuda')
    model.load()
    refined = model.refine(audio, sample_rate=44100)
    
    # Or use custom checkpoint
    model = DisCoder(
        checkpoint_path='/path/to/model.pt',
        config_path='/path/to/config_z.json',
        device='cuda'
    )
"""

import json
import warnings
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Union

from disgen.core.base_model import GenerativeModel, ModelRegistry
from disgen.utils.device import get_device

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    # Use vendored DisCoder (no separate installation needed)
    from disgen.vendor.discoder.models import DisCoder as DisCoderModel
    from disgen.vendor.discoder import meldataset
    from disgen.vendor.discoder import utils as discoder_utils
    DISCODER_AVAILABLE = True
except ImportError as e:
    DISCODER_AVAILABLE = False
    _import_error = str(e)

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


@ModelRegistry.register("discoder", tags=["vocoder", "neural_codec", "dac"])
class DisCoder(GenerativeModel):
    """DisCoder neural vocoder for mel-spectrogram to waveform synthesis.
    
    DisCoder uses a GAN-based encoder-decoder architecture informed by the
    Descript Audio Codec (DAC) to reconstruct high-fidelity 44.1 kHz audio
    from mel spectrograms. The model transforms mel spectrograms into a
    lower-dimensional representation aligned with DAC's latent space before
    reconstructing the audio signal.
    
    Key features:
    - Operates at 44.1 kHz sample rate
    - Requires exactly 128 mel bins
    - Superior quality for music signals
    - Pretrained model available on Hugging Face
    
    Args:
        checkpoint_path: Path to custom checkpoint (.pt file). If None and
            use_pretrained=True, loads from Hugging Face.
        config_path: Path to config JSON file. If None, uses model's config.
        device: Device to run on ('cpu', 'cuda', 'mps', or None for auto)
        use_pretrained: If True, loads pretrained model from Hugging Face
        **kwargs: Additional arguments
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        use_pretrained: bool = True,
        **kwargs
    ):
        """Initialize DisCoder vocoder."""
        device = get_device(device)
        super().__init__(device=device, **kwargs)
        
        if not TORCH_AVAILABLE:
            raise ImportError(
                "DisCoder requires PyTorch. Install with: pip install torch torchaudio"
            )
        
        if not DISCODER_AVAILABLE:
            raise ImportError(
                "DisCoder vendored code not found or missing dependencies. Install:\n"
                "  pip install descript-audio-codec einops huggingface_hub\n"
                f"Error: {_import_error if '_import_error' in locals() else 'unknown'}"
            )
        
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.config_path = Path(config_path) if config_path else None
        self.use_pretrained = use_pretrained
        
        self.model = None
        self.config = None
        self._modes = [self.MODE_VOCODER, self.MODE_REFINEMENT]
    
    def load(self):
        """Load DisCoder model from Hugging Face or custom checkpoint."""
        if self.use_pretrained and self.checkpoint_path is None:
            print("Loading pretrained DisCoder model from Hugging Face...")
            try:
                self.model = DisCoderModel.from_pretrained(
                    "disco-eth/discoder"
                ).eval().to(self.device)
                self.config = self.model.config
                print(f"DisCoder loaded on {self.device}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load pretrained DisCoder: {e}\n"
                    f"Make sure you have internet connection and huggingface_hub installed."
                )
        else:
            # Load custom checkpoint
            if self.checkpoint_path is None:
                raise ValueError("checkpoint_path required when use_pretrained=False")
            
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found: {self.checkpoint_path}"
                )
            
            print(f"Loading DisCoder from {self.checkpoint_path}")
            
            # Load config
            if self.config_path and self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                # Use default config
                self.config = self._get_default_config()
                warnings.warn(
                    f"Config not found at {self.config_path}. Using default."
                )
            
            # Create and load model
            # Note: This assumes the DisCoder class can be instantiated
            # In practice, you may need to import specific architecture
            self.model = DisCoderModel(self.config).to(self.device)
            
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            if 'generator' in checkpoint:
                self.model.load_state_dict(checkpoint['generator'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            print(f"DisCoder loaded on {self.device}")
        
        # Validate config
        if self.config.get('num_mels') != 128:
            warnings.warn(
                f"DisCoder expects 128 mel bins, but config has {self.config.get('num_mels')}. "
                f"This may cause issues."
            )
        
        print(f"Sample rate: {self.config.get('sample_rate', 44100)} Hz")
        print(f"Mel bins: {self.config.get('num_mels', 128)}")
    
    def _get_default_config(self) -> Dict:
        """Get default DisCoder config."""
        return {
            "sample_rate": 44100,
            "num_mels": 128,
            "n_fft": 2048,
            "hop_size": 512,
            "win_size": 2048,
            "fmin": 0,
            "fmax": None,  # Will be SR/2
            "segment_size": 65536,  # Default segment size
        }
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def refine(
        self,
        audio: np.ndarray,
        sample_rate: int,
        stem_name: Optional[str] = None,
        return_at_original_sr: bool = False,
        **kwargs
    ) -> np.ndarray:
        """Refine audio using analysis-synthesis with DisCoder.
        
        Args:
            audio: Input audio waveform, shape (channels, samples) or (samples,)
            sample_rate: Sample rate in Hz
            stem_name: Optional stem name (not used, for compatibility)
            return_at_original_sr: If True, resample output to original SR
            **kwargs: Additional parameters
            
        Returns:
            Refined audio waveform at 44.1 kHz (or original SR if requested)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Extract mel spectrogram
        mel_spec, original_length = self._extract_mel(audio, sample_rate)
        
        # Vocode
        refined = self.vocode(mel_spec, original_length=original_length, **kwargs)
        
        # Resample if requested
        if return_at_original_sr and sample_rate != self.config["sample_rate"]:
            if not LIBROSA_AVAILABLE:
                warnings.warn("librosa not available for resampling, returning at 44.1kHz")
            else:
                refined = librosa.resample(
                    refined,
                    orig_sr=self.config["sample_rate"],
                    target_sr=sample_rate
                )
        
        return refined
    
    def vocode(
        self,
        spectrogram: Union[np.ndarray, torch.Tensor],
        original_length: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """Convert mel-spectrogram to waveform using DisCoder.
        
        Args:
            spectrogram: Mel-spectrogram, shape (n_mels, time) or (batch, n_mels, time)
            original_length: Original audio length in samples (for unpadding)
            **kwargs: Additional parameters
            
        Returns:
            Audio waveform as numpy array
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Convert to tensor if needed
        if isinstance(spectrogram, np.ndarray):
            mel_tensor = torch.from_numpy(spectrogram).float().to(self.device)
        else:
            mel_tensor = spectrogram.to(self.device)
        
        # Add batch dimension if needed
        if mel_tensor.dim() == 2:
            mel_tensor = mel_tensor.unsqueeze(0)  # (n_mels, time) -> (1, n_mels, time)
        
        # Generate waveform
        with torch.no_grad():
            # DisCoder expects (batch, n_mels, time)
            wav_recon = self.model(mel_tensor)  # Returns (batch, 1, time)
        
        # Remove batch and channel dimensions
        wav_recon = wav_recon.squeeze().cpu().numpy()
        
        # Unpad if original length provided
        if original_length is not None and len(wav_recon) > original_length:
            wav_recon = wav_recon[:original_length]
        
        return wav_recon
    
    def _extract_mel(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> tuple[torch.Tensor, int]:
        """Extract mel spectrogram from audio using DisCoder's configuration.
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate
            
        Returns:
            Tuple of (mel_tensor, original_length)
        """
        if not LIBROSA_AVAILABLE:
            raise RuntimeError("librosa required for mel extraction")
        
        # Convert to mono if stereo
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)
        
        # Store original length
        original_length = len(audio)
        
        # Resample to DisCoder's sample rate if needed
        target_sr = self.config["sample_rate"]
        if sample_rate != target_sr:
            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=target_sr
            )
        
        # Normalize
        audio = audio / np.max(np.abs(audio) + 1e-8)
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        
        # Apply padding for segment alignment
        segment_size = self.config.get("segment_size", 65536)
        to_pad = segment_size - (audio_tensor.shape[-1] % segment_size)
        if to_pad < segment_size:  # Don't pad if already aligned
            audio_tensor = F.pad(audio_tensor, (0, to_pad), mode="constant", value=0)
        
        # Extract mel using DisCoder's utility function
        mel_spec = discoder_utils.get_mel_spectrogram_from_config(
            audio_tensor,
            self.config
        )
        
        return mel_spec, original_length
    
    @property
    def supported_modes(self) -> List[str]:
        """Get supported modes."""
        return self._modes
    
    def __repr__(self) -> str:
        """String representation."""
        if self.use_pretrained:
            source = "Hugging Face (disco-eth/discoder)"
        else:
            source = str(self.checkpoint_path) if self.checkpoint_path else "custom"
        
        return (
            f"DisCoder(device={self.device}, source={source}, "
            f"loaded={self.is_loaded})"
        )

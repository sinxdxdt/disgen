"""Audio I/O utilities."""

import numpy as np
import soundfile as sf
import warnings
from pathlib import Path
from typing import Tuple, Union, Optional

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    warnings.warn("librosa not installed. Some audio processing features unavailable.")


def load_audio(
    path: Union[str, Path],
    sample_rate: Optional[int] = None,
    mono: bool = False,
    duration: Optional[float] = None,
    offset: float = 0.0
) -> Tuple[np.ndarray, int]:
    """Load audio file.
    
    Args:
        path: Path to audio file
        sample_rate: Target sample rate (None to keep original)
        mono: Convert to mono
        duration: Duration to load in seconds (None for full file)
        offset: Start offset in seconds
        
    Returns:
        Tuple of (audio, sample_rate)
        - audio: shape (samples,) for mono or (channels, samples) for stereo
        - sample_rate: sampling rate in Hz
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    
    # Load with soundfile
    if duration is not None:
        frames = int(duration * sample_rate) if sample_rate else -1
        audio, sr = sf.read(str(path), frames=frames, start=int(offset * sample_rate) if sample_rate else 0, always_2d=True)
    else:
        audio, sr = sf.read(str(path), always_2d=True)
    
    # Convert to (channels, samples) if stereo
    if audio.shape[1] > 1:
        audio = audio.T  # (samples, channels) -> (channels, samples)
    else:
        audio = audio.T[0]  # (samples, 1) -> (samples,)
    
    # Resample if needed
    if sample_rate is not None and sr != sample_rate:
        if not LIBROSA_AVAILABLE:
            raise RuntimeError("librosa required for resampling")
        
        if audio.ndim == 2:
            # Resample each channel
            audio = np.array([
                librosa.resample(audio[i], orig_sr=sr, target_sr=sample_rate)
                for i in range(audio.shape[0])
            ])
        else:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate
    
    # Convert to mono if requested
    if mono and audio.ndim == 2:
        audio = np.mean(audio, axis=0)
    
    return audio, sr


def save_audio(
    path: Union[str, Path],
    audio: np.ndarray,
    sample_rate: int,
    subtype: str = 'PCM_16'
):
    """Save audio to file.
    
    Args:
        path: Output path
        audio: Audio data, shape (samples,) or (channels, samples)
        sample_rate: Sample rate in Hz
        subtype: Audio subtype (e.g., 'PCM_16', 'FLOAT')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to (samples, channels) format for soundfile
    if audio.ndim == 2:
        audio = audio.T
    
    sf.write(str(path), audio, sample_rate, subtype=subtype)


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int
) -> np.ndarray:
    """Resample audio to target sample rate.
    
    Args:
        audio: Input audio
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio
    """
    if not LIBROSA_AVAILABLE:
        raise RuntimeError("librosa required for resampling")
    
    if orig_sr == target_sr:
        return audio
    
    if audio.ndim == 2:
        # Resample each channel
        return np.array([
            librosa.resample(audio[i], orig_sr=orig_sr, target_sr=target_sr)
            for i in range(audio.shape[0])
        ])
    else:
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def ensure_length(
    audio: np.ndarray,
    target_length: int,
    pad_mode: str = 'constant'
) -> np.ndarray:
    """Ensure audio has target length by padding or truncating.
    
    Args:
        audio: Input audio
        target_length: Target length in samples
        pad_mode: Padding mode for np.pad
        
    Returns:
        Audio with target length
    """
    current_length = audio.shape[-1] if audio.ndim == 2 else len(audio)
    
    if current_length > target_length:
        # Truncate
        if audio.ndim == 2:
            return audio[:, :target_length]
        else:
            return audio[:target_length]
    elif current_length < target_length:
        # Pad
        pad_length = target_length - current_length
        if audio.ndim == 2:
            return np.pad(audio, ((0, 0), (0, pad_length)), mode=pad_mode)
        else:
            return np.pad(audio, (0, pad_length), mode=pad_mode)
    else:
        return audio


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert stereo audio to mono.
    
    Args:
        audio: Input audio, shape (channels, samples) or (samples,)
        
    Returns:
        Mono audio, shape (samples,)
    """
    if audio.ndim == 1:
        return audio
    elif audio.ndim == 2:
        return np.mean(audio, axis=0)
    else:
        raise ValueError(f"Unexpected audio shape: {audio.shape}")


def get_duration(audio: np.ndarray, sample_rate: int) -> float:
    """Get audio duration in seconds.
    
    Args:
        audio: Input audio
        sample_rate: Sample rate in Hz
        
    Returns:
        Duration in seconds
    """
    n_samples = audio.shape[-1] if audio.ndim == 2 else len(audio)
    return n_samples / sample_rate


def normalize_audio(audio: np.ndarray, target_level: float = -1.0) -> np.ndarray:
    """Normalize audio to target peak level in dB.
    
    Args:
        audio: Input audio
        target_level: Target peak level in dB
        
    Returns:
        Normalized audio
    """
    peak = np.abs(audio).max()
    if peak == 0:
        return audio
    
    target_peak = 10 ** (target_level / 20.0)
    return audio * (target_peak / peak)

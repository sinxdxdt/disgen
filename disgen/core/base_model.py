"""Base model classes and registry for discriminative and generative models."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List
import numpy as np


class ModelRegistry:
    """Registry for managing model implementations."""
    
    _models: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, tags: Optional[List[str]] = None):
        """Decorator to register a model class.
        
        Args:
            name: Primary name for the model
            tags: Optional additional tags/aliases for the model
        """
        def decorator(model_cls):
            cls._models[name.lower()] = model_cls
            if tags:
                for tag in tags:
                    cls._models[tag.lower()] = model_cls
            return model_cls
        return decorator
    
    @classmethod
    def get(cls, name: str) -> type:
        """Get a model class by name or tag.
        
        Args:
            name: Model name or tag
            
        Returns:
            Model class
            
        Raises:
            KeyError: If model not found
        """
        key = name.lower()
        if key not in cls._models:
            raise KeyError(
                f"Model '{name}' not found in registry. "
                f"Available models: {list(cls._models.keys())}"
            )
        return cls._models[key]
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered model names."""
        return list(cls._models.keys())


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, device: str = "cpu", **kwargs):
        """Initialize the model.
        
        Args:
            device: Device to run the model on ('cpu', 'cuda', 'mps')
            **kwargs: Additional model-specific arguments
        """
        self.device = device
        self.config = kwargs
    
    @abstractmethod
    def load(self):
        """Load model weights and prepare for inference."""
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        pass


class DiscriminativeModel(BaseModel):
    """Abstract base class for discriminative source separation models."""
    
    DOMAIN_TIME = "time"
    DOMAIN_FREQUENCY = "frequency"
    
    @abstractmethod
    def separate(
        self, 
        mixture: np.ndarray, 
        sample_rate: int,
        stem: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """Separate mixture into sources.
        
        Args:
            mixture: Audio mixture, shape (channels, samples) or (samples,)
            sample_rate: Sample rate in Hz
            stem: Optional specific stem to extract (e.g., 'vocals', 'drums')
                 If None, returns all available stems
        
        Returns:
            Dictionary mapping stem names to separated audio arrays
            Each audio array has shape (channels, samples) or (samples,)
        """
        pass
    
    @property
    @abstractmethod
    def domain(self) -> str:
        """Return the processing domain of the model.
        
        Returns:
            Either DOMAIN_TIME or DOMAIN_FREQUENCY
        """
        pass
    
    @property
    @abstractmethod
    def available_stems(self) -> List[str]:
        """Return list of stems this model can separate.
        
        Returns:
            List of stem names (e.g., ['vocals', 'drums', 'bass', 'other'])
        """
        pass
    
    def get_spectrogram(
        self,
        mixture: np.ndarray,
        sample_rate: int,
        stem: Optional[str] = None
    ) -> Optional[Dict[str, np.ndarray]]:
        """Get spectrogram representation if model operates in frequency domain.
        
        This method should be implemented by frequency-domain models to provide
        spectrograms for vocoder-based generative refinement.
        
        Args:
            mixture: Audio mixture
            sample_rate: Sample rate in Hz
            stem: Optional specific stem
            
        Returns:
            Dictionary mapping stem names to spectrograms (e.g., mel-spectrograms),
            or None if model is time-domain
        """
        if self.domain == self.DOMAIN_TIME:
            return None
        return None  # Subclasses should override if they support this


class GenerativeModel(BaseModel):
    """Abstract base class for generative refinement/vocoder models."""
    
    MODE_REFINEMENT = "refinement"  # Waveform-to-waveform refinement
    MODE_VOCODER = "vocoder"  # Spectrogram-to-waveform vocoding
    
    @abstractmethod
    def refine(
        self,
        audio: np.ndarray,
        sample_rate: int,
        stem_name: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """Refine/enhance audio waveform.
        
        Args:
            audio: Input audio, shape (channels, samples) or (samples,)
            sample_rate: Sample rate in Hz
            stem_name: Optional stem name for conditional models (e.g., DPM-TSE)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Refined audio with same shape as input
        """
        pass
    
    @property
    @abstractmethod
    def supported_modes(self) -> List[str]:
        """Return list of supported modes.
        
        Returns:
            List containing MODE_REFINEMENT and/or MODE_VOCODER
        """
        pass
    
    def vocode(
        self,
        spectrogram: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Convert spectrogram to waveform (vocoder mode).
        
        Args:
            spectrogram: Input spectrogram (e.g., mel-spectrogram)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Audio waveform
            
        Raises:
            NotImplementedError: If model doesn't support vocoding
        """
        if self.MODE_VOCODER not in self.supported_modes:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support vocoder mode"
            )
        raise NotImplementedError("Subclass must implement vocode()")

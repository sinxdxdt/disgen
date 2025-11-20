"""Base dataset classes and registry."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Iterator
import numpy as np
import random


class DatasetRegistry:
    """Registry for managing dataset implementations."""
    
    _datasets: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, tags: Optional[List[str]] = None):
        """Decorator to register a dataset class.
        
        Args:
            name: Primary name for the dataset
            tags: Optional additional tags/aliases
        """
        def decorator(dataset_cls):
            cls._datasets[name.lower()] = dataset_cls
            if tags:
                for tag in tags:
                    cls._datasets[tag.lower()] = dataset_cls
            return dataset_cls
        return decorator
    
    @classmethod
    def get(cls, name: str) -> type:
        """Get a dataset class by name or tag.
        
        Args:
            name: Dataset name or tag
            
        Returns:
            Dataset class
            
        Raises:
            KeyError: If dataset not found
        """
        key = name.lower()
        if key not in cls._datasets:
            raise KeyError(
                f"Dataset '{name}' not found in registry. "
                f"Available datasets: {list(cls._datasets.keys())}"
            )
        return cls._datasets[key]
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all registered dataset names."""
        return list(cls._datasets.keys())


class Track:
    """Represents a single track with mixture and reference stems."""
    
    def __init__(
        self,
        track_id: str,
        mixture: np.ndarray,
        references: Dict[str, np.ndarray],
        sample_rate: int,
        metadata: Optional[Dict] = None
    ):
        """Initialize a track.
        
        Args:
            track_id: Unique identifier for the track
            mixture: Audio mixture, shape (channels, samples) or (samples,)
            references: Dictionary mapping stem names to reference audio
            sample_rate: Sample rate in Hz
            metadata: Optional metadata dictionary
        """
        self.track_id = track_id
        self.mixture = mixture
        self.references = references
        self.sample_rate = sample_rate
        self.metadata = metadata or {}
    
    @property
    def stems(self) -> List[str]:
        """Get list of available stems."""
        return list(self.references.keys())
    
    @property
    def duration(self) -> float:
        """Get track duration in seconds."""
        if self.mixture.ndim == 1:
            return len(self.mixture) / self.sample_rate
        return self.mixture.shape[1] / self.sample_rate


class BaseDataset(ABC):
    """Abstract base class for music source separation datasets."""
    
    def __init__(
        self,
        root_path: str,
        subset: str = "test",
        test_subset_size: float = 0.15,
        seed: int = 42,
        **kwargs
    ):
        """Initialize the dataset.
        
        Args:
            root_path: Path to dataset root directory
            subset: Subset to use ('train', 'test', 'validation')
            test_subset_size: Fraction of data to use as test set if explicit
                             test split not available (default: 0.15)
            seed: Random seed for subset sampling
            **kwargs: Additional dataset-specific arguments
        """
        self.root_path = root_path
        self.subset = subset
        self.test_subset_size = test_subset_size
        self.seed = seed
        self.config = kwargs
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
    
    @abstractmethod
    def load(self):
        """Load and prepare the dataset."""
        pass
    
    @property
    @abstractmethod
    def has_test_split(self) -> bool:
        """Check if dataset has an explicit test split.
        
        Returns:
            True if dataset has predefined test split, False otherwise
        """
        pass
    
    @abstractmethod
    def get_test_tracks(self) -> List[str]:
        """Get list of track IDs in the test set.
        
        If dataset has explicit test split, returns those tracks.
        Otherwise, samples test_subset_size fraction of tracks.
        
        Returns:
            List of track IDs for testing
        """
        pass
    
    @abstractmethod
    def get_track(self, track_id: str) -> Track:
        """Load a single track.
        
        Args:
            track_id: Track identifier
            
        Returns:
            Track object with mixture and references
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return number of tracks in current subset."""
        pass
    
    def __iter__(self) -> Iterator[Track]:
        """Iterate over tracks in the dataset."""
        track_ids = self.get_test_tracks() if self.subset == "test" else self.get_all_tracks()
        for track_id in track_ids:
            yield self.get_track(track_id)
    
    @abstractmethod
    def get_all_tracks(self) -> List[str]:
        """Get list of all track IDs in the dataset.
        
        Returns:
            List of all track IDs
        """
        pass
    
    @property
    @abstractmethod
    def available_stems(self) -> List[str]:
        """Get list of stems available in this dataset.
        
        Returns:
            List of stem names (e.g., ['vocals', 'drums', 'bass', 'other'])
        """
        pass

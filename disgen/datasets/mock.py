"""Mock dataset for testing purposes."""

import numpy as np
from typing import List
import random

from disgen.core.base_dataset import BaseDataset, Track, DatasetRegistry


@DatasetRegistry.register("mock_dataset", tags=["mock"])
class MockDataset(BaseDataset):
    """Mock dataset for testing without real data files."""
    
    def __init__(
        self,
        num_tracks: int = 10,
        duration: float = 5.0,
        sample_rate: int = 16000,
        stems: List[str] = None,
        has_test_split: bool = True,
        **kwargs
    ):
        """Initialize mock dataset.
        
        Args:
            num_tracks: Number of tracks to generate
            duration: Duration of each track in seconds
            sample_rate: Sample rate in Hz
            stems: List of stem names (default: ['vocals', 'drums', 'bass', 'other'])
            has_test_split: Whether dataset has explicit test split
            **kwargs: Additional arguments
        """
        super().__init__(root_path="", **kwargs)
        self.num_tracks = num_tracks
        self.duration = duration
        self.sample_rate = sample_rate
        self._stems = stems or ['vocals', 'drums', 'bass', 'other']
        self._has_test_split = has_test_split
        self._loaded = False
    
    def load(self):
        """Load dataset (mock - does nothing)."""
        self._loaded = True
    
    @property
    def has_test_split(self) -> bool:
        return self._has_test_split
    
    def get_test_tracks(self) -> List[str]:
        """Get test track IDs."""
        all_tracks = self.get_all_tracks()
        
        if self._has_test_split:
            # Return last 30% as test
            test_size = max(1, int(self.num_tracks * 0.3))
            return all_tracks[-test_size:]
        else:
            # Sample test subset
            test_size = max(1, int(self.num_tracks * self.test_subset_size))
            return random.sample(all_tracks, test_size)
    
    def get_track(self, track_id: str) -> Track:
        """Generate a mock track.
        
        Args:
            track_id: Track identifier
            
        Returns:
            Mock Track object
        """
        n_samples = int(self.duration * self.sample_rate)
        
        # Generate mixture as sum of stems with noise
        mixture = np.random.randn(n_samples) * 0.1
        
        # Generate references for each stem
        references = {}
        for stem_name in self._stems:
            # Each stem is a random signal
            stem_signal = np.random.randn(n_samples) * 0.3
            references[stem_name] = stem_signal
            mixture += stem_signal * 0.25
        
        return Track(
            track_id=track_id,
            mixture=mixture,
            references=references,
            sample_rate=self.sample_rate,
            metadata={"mock": True}
        )
    
    def __len__(self) -> int:
        if self.subset == "test":
            return len(self.get_test_tracks())
        return self.num_tracks
    
    def get_all_tracks(self) -> List[str]:
        """Get all track IDs."""
        return [f"track_{i:03d}" for i in range(self.num_tracks)]
    
    @property
    def available_stems(self) -> List[str]:
        return self._stems

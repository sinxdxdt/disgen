"""MoisesDB dataset implementation using official moisesdb package."""

import random
from pathlib import Path
from typing import List, Optional, Dict

from disgen.core.base_dataset import BaseDataset, Track, DatasetRegistry

try:
    from moisesdb.dataset import MoisesDB as MoisesDBNative
    MOISESDB_AVAILABLE = True
except ImportError:
    MOISESDB_AVAILABLE = False


@DatasetRegistry.register("moisesdb", tags=["moises"])
class MoisesDB(BaseDataset):
    """MoisesDB dataset adapter using official moisesdb package.
    
    Dataset structure:
    - 240 tracks total
    - No explicit test split (uses custom sampling)
    - Hierarchical stem structure
    - Multiple genres
    - Sample rate: 44100 Hz
    
    Uses the official moisesdb package for robust track loading and metadata access.
    
    Args:
        root_path: Path to MoisesDB root directory
        subset: 'train', 'test', or 'all' (default: 'test')
        test_subset_size: Fraction for test subset if no explicit split (default: 0.15)
        sample_rate: Target sample rate (default: 44100)
        seed: Random seed for test split sampling
        **kwargs: Additional arguments
    """
    
    def __init__(
        self,
        root_path: str,
        subset: str = "test",
        test_subset_size: float = 0.15,
        sample_rate: int = 44100,
        seed: int = 42,
        **kwargs
    ):
        """Initialize MoisesDB dataset."""
        if not MOISESDB_AVAILABLE:
            raise ImportError(
                "moisesdb package not installed. Install with:\n"
                "  pip install git+https://github.com/moises-ai/moises-db.git"
            )
        
        super().__init__(
            root_path=root_path,
            subset=subset,
            test_subset_size=test_subset_size,
            **kwargs
        )
        
        self.sample_rate = sample_rate
        self.seed = seed
        self.native_db = None
        self._test_indices = None
    
    def load(self):
        """Load MoisesDB dataset using official package."""
        if not Path(self.root_path).exists():
            raise FileNotFoundError(
                f"MoisesDB not found at {self.root_path}. "
                f"Download the dataset first."
            )
        
        # Load using official moisesdb package
        self.native_db = MoisesDBNative(
            data_path=self.root_path,
            sample_rate=self.sample_rate,
            quiet=False  # Show progress bars during loading
        )
        
        if len(self.native_db) == 0:
            raise ValueError(f"No tracks found in {self.root_path}")
        
        # Create test split indices
        total = len(self.native_db)
        
        if self.subset == "test":
            # Sample test subset
            test_size = max(1, int(total * self.test_subset_size))
            random.seed(self.seed)
            self._test_indices = random.sample(range(total), test_size)
        elif self.subset == "train":
            # Training = complement of test
            test_size = max(1, int(total * self.test_subset_size))
            random.seed(self.seed)
            test_indices = set(random.sample(range(total), test_size))
            self._test_indices = [i for i in range(total) if i not in test_indices]
        else:  # 'all'
            self._test_indices = list(range(total))
    
    @property
    def has_test_split(self) -> bool:
        """MoisesDB does not have explicit test split."""
        return False
    
    def get_test_tracks(self) -> List[str]:
        """Get test track IDs (sampled subset)."""
        if self.native_db is None:
            self.load()
        
        return [self.native_db[i].id for i in self._test_indices 
                if self.subset in ["test", "all"]]
    
    def get_track(self, track_id: str) -> Track:
        """Get a single track.
        
        Args:
            track_id: Track ID
            
        Returns:
            Track object with mixture and references
        """
        if self.native_db is None:
            self.load()
        
        # Find track index by ID
        idx = None
        for i in range(len(self.native_db)):
            if self.native_db[i].id == track_id:
                idx = i
                break
        
        if idx is None:
            raise ValueError(f"Track {track_id} not found")
        
        # Get native track
        native_track = self.native_db[idx]
        
        # Convert to disgen Track format
        return Track(
            track_id=native_track.id,
            mixture=native_track.audio,  # np.ndarray from official package
            references=native_track.stems,  # dict of np.ndarray
            sample_rate=self.sample_rate,
            metadata={
                'provider': native_track.provider,
                'artist': native_track.artist,
                'name': native_track.name,
                'genre': native_track.genre,
                'sources': native_track.sources,
                'bleedings': native_track.bleedings,
                'activity': getattr(native_track, 'activity', None),
            }
        )
    
    def __len__(self) -> int:
        """Get dataset length."""
        if self.native_db is None:
            self.load()
        
        return len(self._test_indices) if self._test_indices else 0
    
    def get_all_tracks(self) -> List[str]:
        """Get all track IDs for current subset."""
        if self.native_db is None:
            self.load()
        
        return [self.native_db[i].id for i in self._test_indices]
    
    @property
    def available_stems(self) -> List[str]:
        """Get available stem names."""
        if self.native_db is None:
            self.load()
        
        if len(self.native_db) == 0:
            return []
        
        # Use first track to determine available stems
        return list(self.native_db[0].stems.keys())

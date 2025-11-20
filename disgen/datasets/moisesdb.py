"""MoisesDB dataset implementation."""

import json
import random
import warnings
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np

from disgen.core.base_dataset import BaseDataset, Track, DatasetRegistry
from disgen.utils.audio import load_audio


@DatasetRegistry.register("moisesdb", tags=["moises"])
class MoisesDB(BaseDataset):
    """MoisesDB dataset adapter.
    
    Dataset structure:
    - 240 tracks total
    - No explicit test split (uses custom sampling)
    - Hierarchical stem structure
    - Multiple genres
    - Sample rate: 44100 Hz
    
    Each track has:
    - mixture.wav
    - stems/ directory with various stem files
    - metadata.json with stem information
    
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
        super().__init__(
            root_path=root_path,
            subset=subset,
            test_subset_size=test_subset_size,
            **kwargs
        )
        
        self.sample_rate = sample_rate
        self.seed = seed
        self.track_list = []
        self._test_tracks = None
        
        # Standard stem categories
        self._stem_categories = ['vocals', 'drums', 'bass', 'other']
    
    def load(self):
        """Load MoisesDB dataset."""
        root = Path(self.root_path)
        if not root.exists():
            raise FileNotFoundError(
                f"MoisesDB not found at {self.root_path}. "
                f"Install moisesdb and download data."
            )
        
        # Get all track directories
        self.track_list = sorted([
            d.name for d in root.iterdir()
            if d.is_dir() and (d / "mixture.wav").exists()
        ])
        
        if not self.track_list:
            raise ValueError(f"No tracks found in {self.root_path}")
        
        # Sample test subset if needed
        if self.subset == "test":
            test_size = max(1, int(len(self.track_list) * self.test_subset_size))
            random.seed(self.seed)
            self._test_tracks = random.sample(self.track_list, test_size)
        elif self.subset == "train":
            # Get complement of test set
            test_size = max(1, int(len(self.track_list) * self.test_subset_size))
            random.seed(self.seed)
            test_tracks = set(random.sample(self.track_list, test_size))
            self._test_tracks = [t for t in self.track_list if t not in test_tracks]
        else:  # 'all'
            self._test_tracks = self.track_list
    
    @property
    def has_test_split(self) -> bool:
        """MoisesDB does not have explicit test split."""
        return False
    
    def get_test_tracks(self) -> List[str]:
        """Get test track IDs (sampled subset)."""
        if self._test_tracks is None:
            self.load()
        
        if self.subset == "test":
            return self._test_tracks
        else:
            # Sample test subset from all tracks
            test_size = max(1, int(len(self.track_list) * self.test_subset_size))
            random.seed(self.seed)
            return random.sample(self.track_list, test_size)
    
    def get_track(self, track_id: str) -> Track:
        """Get a single track.
        
        Args:
            track_id: Track directory name
            
        Returns:
            Track object with mixture and references
        """
        if not self.track_list:
            self.load()
        
        track_dir = Path(self.root_path) / track_id
        if not track_dir.exists():
            raise ValueError(f"Track {track_id} not found")
        
        # Load mixture
        mix_path = track_dir / "mixture.wav"
        if not mix_path.exists():
            raise FileNotFoundError(f"Mixture not found: {mix_path}")
        
        mixture, sr = load_audio(mix_path, sample_rate=self.sample_rate)
        
        # Load metadata
        metadata = self._load_metadata(track_dir / "metadata.json")
        
        # Load stems from stems directory
        stems_dir = track_dir / "stems"
        references = {}
        
        if stems_dir.exists():
            # Try to load standard stems
            stem_files = {
                'vocals': ['vocals.wav', 'vocal.wav'],
                'drums': ['drums.wav', 'drum.wav', 'percussion.wav'],
                'bass': ['bass.wav'],
                'other': ['other.wav', 'accompaniment.wav', 'instrumental.wav']
            }
            
            for stem_name, possible_files in stem_files.items():
                for filename in possible_files:
                    stem_path = stems_dir / filename
                    if stem_path.exists():
                        stem_audio, _ = load_audio(stem_path, sample_rate=self.sample_rate)
                        references[stem_name] = stem_audio
                        break
                
                # If stem not found, create silence
                if stem_name not in references:
                    references[stem_name] = np.zeros_like(mixture)
        else:
            # No stems directory - create silent references
            for stem_name in self._stem_categories:
                references[stem_name] = np.zeros_like(mixture)
        
        return Track(
            track_id=track_id,
            mixture=mixture,
            references=references,
            sample_rate=self.sample_rate,
            metadata=metadata
        )
    
    def _load_metadata(self, metadata_path: Path) -> Dict:
        """Load track metadata from JSON file."""
        if not metadata_path.exists():
            return {}
        
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            warnings.warn(f"Failed to load metadata: {e}")
            return {}
    
    def __len__(self) -> int:
        """Get dataset length."""
        if not self.track_list:
            self.load()
        
        if self.subset == "test" or self.subset == "train":
            return len(self._test_tracks) if self._test_tracks else 0
        return len(self.track_list)
    
    def get_all_tracks(self) -> List[str]:
        """Get all track IDs for current subset."""
        if not self.track_list:
            self.load()
        
        if self.subset == "test" or self.subset == "train":
            return self._test_tracks if self._test_tracks else []
        return self.track_list
    
    @property
    def available_stems(self) -> List[str]:
        """Get available stem names."""
        return self._stem_categories

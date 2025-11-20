"""MUSDB18-HQ dataset implementation."""

import warnings
from pathlib import Path
from typing import List, Optional
import numpy as np

from disgen.core.base_dataset import BaseDataset, Track, DatasetRegistry
from disgen.utils.audio import load_audio

try:
    import musdb
    MUSDB_AVAILABLE = True
except ImportError:
    MUSDB_AVAILABLE = False
    warnings.warn(
        "musdb not installed. To use MUSDB18-HQ, install with: pip install musdb"
    )


@DatasetRegistry.register("musdb18hq", tags=["musdb", "musdb18"])
class MUSDB18HQ(BaseDataset):
    """MUSDB18-HQ dataset adapter.
    
    Dataset structure:
    - 100 train tracks
    - 50 test tracks
    - 4 stems: vocals, drums, bass, other
    - Sample rate: 44100 Hz
    - Stereo audio
    
    Args:
        root_path: Path to MUSDB18-HQ root directory
        subset: 'train' or 'test' (default: 'test')
        sample_rate: Target sample rate (default: 44100)
        **kwargs: Additional arguments
    """
    
    def __init__(
        self,
        root_path: str,
        subset: str = "test",
        sample_rate: int = 44100,
        **kwargs
    ):
        """Initialize MUSDB18-HQ dataset."""
        super().__init__(root_path=root_path, subset=subset, **kwargs)
        
        if not MUSDB_AVAILABLE:
            raise ImportError(
                "musdb package required for MUSDB18-HQ. Install with: pip install musdb"
            )
        
        self.sample_rate = sample_rate
        self.mus = None
        self._stem_names = ['vocals', 'drums', 'bass', 'other']
    
    def load(self):
        """Load MUSDB18-HQ dataset."""
        if not Path(self.root_path).exists():
            raise FileNotFoundError(
                f"MUSDB18-HQ not found at {self.root_path}. "
                f"Download from https://zenodo.org/record/3338373"
            )
        
        # Initialize musdb with specified subset
        self.mus = musdb.DB(
            root=self.root_path,
            subsets=[self.subset] if self.subset in ["train", "test"] else "train",
            is_wav=True  # MUSDB18-HQ uses WAV format
        )
    
    @property
    def has_test_split(self) -> bool:
        """MUSDB18-HQ has explicit test split."""
        return True
    
    def get_test_tracks(self) -> List[str]:
        """Get test track IDs."""
        if self.mus is None:
            self.load()
        
        # Load test subset
        test_mus = musdb.DB(root=self.root_path, subsets="test", is_wav=True)
        return [track.name for track in test_mus.tracks]
    
    def get_track(self, track_id: str) -> Track:
        """Get a single track.
        
        Args:
            track_id: Track name/ID
            
        Returns:
            Track object with mixture and references
        """
        if self.mus is None:
            self.load()
        
        # Find track by name
        track = None
        for t in self.mus.tracks:
            if t.name == track_id:
                track = t
                break
        
        if track is None:
            raise ValueError(f"Track {track_id} not found in dataset")
        
        # Load audio (musdb returns (samples, channels))
        mixture_audio = track.audio.T  # Convert to (channels, samples)
        
        # Load stems
        references = {}
        for stem_name in self._stem_names:
            if stem_name == 'other':
                # 'other' is 'accompaniment' in musdb
                stem_audio = track.targets['accompaniment'].audio.T
            else:
                stem_audio = track.targets[stem_name].audio.T
            references[stem_name] = stem_audio
        
        # Resample if needed
        if track.rate != self.sample_rate:
            from disgen.utils.audio import resample_audio
            mixture_audio = resample_audio(mixture_audio, track.rate, self.sample_rate)
            for stem_name in references:
                references[stem_name] = resample_audio(
                    references[stem_name], track.rate, self.sample_rate
                )
        
        return Track(
            track_id=track_id,
            mixture=mixture_audio,
            references=references,
            sample_rate=self.sample_rate,
            metadata={
                "duration": track.duration,
                "original_sr": track.rate,
            }
        )
    
    def __len__(self) -> int:
        """Get dataset length."""
        if self.mus is None:
            self.load()
        return len(self.mus.tracks)
    
    def get_all_tracks(self) -> List[str]:
        """Get all track IDs."""
        if self.mus is None:
            self.load()
        return [track.name for track in self.mus.tracks]
    
    @property
    def available_stems(self) -> List[str]:
        """Get available stem names."""
        return self._stem_names

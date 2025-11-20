"""Slakh2100 dataset implementation."""

import json
import warnings
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np

from disgen.core.base_dataset import BaseDataset, Track, DatasetRegistry
from disgen.utils.audio import load_audio


@DatasetRegistry.register("slakh2100", tags=["slakh"])
class Slakh2100(BaseDataset):
    """Slakh2100 dataset adapter.
    
    Dataset structure:
    - 1500 train tracks
    - 375 validation tracks
    - 225 test tracks
    - Variable number of stems per track
    - Sample rate: 44100 Hz
    - Mono audio in FLAC format
    
    Each track has a directory structure:
    - Track_XXXXX/
      - mix.flac (mixture)
      - stems/
        - S00.flac, S01.flac, ... (individual stems)
      - metadata.yaml (stem information)
    
    Args:
        root_path: Path to Slakh2100 root directory
        subset: 'train', 'validation', or 'test' (default: 'test')
        sample_rate: Target sample rate (default: 44100)
        aggregate_stems: Whether to aggregate stems into standard categories
        **kwargs: Additional arguments
    """
    
    def __init__(
        self,
        root_path: str,
        subset: str = "test",
        sample_rate: int = 44100,
        aggregate_stems: bool = True,
        **kwargs
    ):
        """Initialize Slakh2100 dataset."""
        super().__init__(root_path=root_path, subset=subset, **kwargs)
        
        self.sample_rate = sample_rate
        self.aggregate_stems = aggregate_stems
        self.track_list = []
        self._stem_map = {
            # Map instrument classes to standard stems
            'Drums': 'drums',
            'Bass': 'bass',
            'Guitar': 'other',
            'Piano': 'other',
            'Strings': 'other',
            'Brass': 'other',
            'Winds': 'other',
            'Vocals': 'vocals',
        }
    
    def load(self):
        """Load Slakh2100 dataset."""
        root = Path(self.root_path)
        if not root.exists():
            raise FileNotFoundError(
                f"Slakh2100 not found at {self.root_path}. "
                f"Download from http://www.slakh.com/"
            )
        
        # Check for split directory
        split_dir = root / self.subset
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split '{self.subset}' not found. "
                f"Expected directory: {split_dir}"
            )
        
        # Get all track directories
        self.track_list = sorted([
            d.name for d in split_dir.iterdir()
            if d.is_dir() and d.name.startswith('Track')
        ])
    
    @property
    def has_test_split(self) -> bool:
        """Slakh2100 has explicit test split."""
        return True
    
    def get_test_tracks(self) -> List[str]:
        """Get test track IDs."""
        # Load test subset
        test_dataset = Slakh2100(
            root_path=self.root_path,
            subset="test",
            sample_rate=self.sample_rate
        )
        test_dataset.load()
        return test_dataset.get_all_tracks()
    
    def get_track(self, track_id: str) -> Track:
        """Get a single track.
        
        Args:
            track_id: Track directory name (e.g., 'Track00001')
            
        Returns:
            Track object with mixture and references
        """
        if not self.track_list:
            self.load()
        
        track_dir = Path(self.root_path) / self.subset / track_id
        if not track_dir.exists():
            raise ValueError(f"Track {track_id} not found in {self.subset} subset")
        
        # Load mixture
        mix_path = track_dir / "mix.flac"
        if not mix_path.exists():
            raise FileNotFoundError(f"Mixture not found: {mix_path}")
        
        mixture, sr = load_audio(mix_path, sample_rate=self.sample_rate)
        
        # Load metadata
        metadata_path = track_dir / "metadata.yaml"
        metadata = self._load_metadata(metadata_path)
        
        # Load stems
        stems_dir = track_dir / "stems"
        references = {}
        
        if self.aggregate_stems:
            # Aggregate into standard categories
            aggregated = {'vocals': [], 'drums': [], 'bass': [], 'other': []}
            
            for stem_file in sorted(stems_dir.glob("S*.flac")):
                stem_id = stem_file.stem
                stem_audio, _ = load_audio(stem_file, sample_rate=self.sample_rate)
                
                # Get instrument class from metadata
                inst_class = metadata.get('stems', {}).get(stem_id, {}).get('inst_class', 'Other')
                target_stem = self._stem_map.get(inst_class, 'other')
                aggregated[target_stem].append(stem_audio)
            
            # Sum aggregated stems
            for stem_name, stem_list in aggregated.items():
                if stem_list:
                    references[stem_name] = np.sum(stem_list, axis=0)
                else:
                    # Create silent stem if no sources
                    references[stem_name] = np.zeros_like(mixture)
        else:
            # Load individual stems
            for stem_file in sorted(stems_dir.glob("S*.flac")):
                stem_id = stem_file.stem
                stem_audio, _ = load_audio(stem_file, sample_rate=self.sample_rate)
                references[stem_id] = stem_audio
        
        return Track(
            track_id=track_id,
            mixture=mixture,
            references=references,
            sample_rate=self.sample_rate,
            metadata=metadata
        )
    
    def _load_metadata(self, metadata_path: Path) -> Dict:
        """Load track metadata from YAML file."""
        if not metadata_path.exists():
            return {}
        
        try:
            import yaml
            with open(metadata_path, 'r') as f:
                return yaml.safe_load(f)
        except ImportError:
            warnings.warn("PyYAML not installed. Metadata will not be loaded.")
            return {}
        except Exception as e:
            warnings.warn(f"Failed to load metadata: {e}")
            return {}
    
    def __len__(self) -> int:
        """Get dataset length."""
        if not self.track_list:
            self.load()
        return len(self.track_list)
    
    def get_all_tracks(self) -> List[str]:
        """Get all track IDs."""
        if not self.track_list:
            self.load()
        return self.track_list
    
    @property
    def available_stems(self) -> List[str]:
        """Get available stem names."""
        if self.aggregate_stems:
            return ['vocals', 'drums', 'bass', 'other']
        else:
            # Variable stems - return standard + numbered stems
            return ['vocals', 'drums', 'bass', 'other'] + [f'S{i:02d}' for i in range(10)]

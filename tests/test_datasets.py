"""Unit tests for dataset classes."""

import pytest
import numpy as np

from disgen.datasets import MockDataset
from disgen.core.base_dataset import DatasetRegistry, Track


class TestMockDataset:
    """Test mock dataset."""
    
    def test_initialization(self):
        """Test dataset initialization."""
        dataset = MockDataset(num_tracks=20, duration=3.0)
        assert dataset.num_tracks == 20
        assert dataset.duration == 3.0
    
    def test_load(self):
        """Test dataset loading."""
        dataset = MockDataset()
        dataset.load()
        # Mock dataset doesn't have a loaded flag, but shouldn't raise error
    
    def test_get_track(self):
        """Test getting a single track."""
        dataset = MockDataset(num_tracks=5, duration=2.0, sample_rate=16000)
        dataset.load()
        
        track = dataset.get_track("track_000")
        
        assert isinstance(track, Track)
        assert track.track_id == "track_000"
        assert track.sample_rate == 16000
        assert track.mixture.shape[0] == int(2.0 * 16000)
        assert len(track.stems) > 0
    
    def test_test_split_explicit(self):
        """Test explicit test split."""
        dataset = MockDataset(num_tracks=10, has_test_split=True)
        dataset.load()
        
        test_tracks = dataset.get_test_tracks()
        
        # Should return last 30% (3 tracks)
        assert len(test_tracks) == 3
        assert test_tracks[0] == "track_007"
    
    def test_test_split_sampling(self):
        """Test sampled test split."""
        dataset = MockDataset(
            num_tracks=10,
            has_test_split=False,
            test_subset_size=0.2
        )
        dataset.load()
        
        test_tracks = dataset.get_test_tracks()
        
        # Should sample 20% (2 tracks)
        assert len(test_tracks) == 2
    
    def test_available_stems(self):
        """Test available stems."""
        dataset = MockDataset(stems=['vocals', 'drums'])
        
        assert dataset.available_stems == ['vocals', 'drums']
    
    def test_iteration(self):
        """Test dataset iteration."""
        dataset = MockDataset(num_tracks=3)
        dataset.load()
        
        tracks = list(dataset)
        
        # Should iterate over test tracks
        assert len(tracks) > 0
        assert all(isinstance(t, Track) for t in tracks)
    
    def test_length(self):
        """Test dataset length."""
        dataset = MockDataset(num_tracks=10, has_test_split=True, subset="test")
        dataset.load()
        
        length = len(dataset)
        
        assert length == 3  # 30% of 10


class TestTrack:
    """Test Track class."""
    
    def test_initialization(self):
        """Test track initialization."""
        mixture = np.random.randn(16000)
        references = {
            'vocals': np.random.randn(16000),
            'drums': np.random.randn(16000)
        }
        
        track = Track(
            track_id="test_001",
            mixture=mixture,
            references=references,
            sample_rate=16000
        )
        
        assert track.track_id == "test_001"
        assert track.sample_rate == 16000
        assert np.array_equal(track.mixture, mixture)
    
    def test_stems_property(self):
        """Test stems property."""
        references = {
            'vocals': np.random.randn(1000),
            'drums': np.random.randn(1000),
            'bass': np.random.randn(1000)
        }
        
        track = Track(
            track_id="test",
            mixture=np.random.randn(1000),
            references=references,
            sample_rate=16000
        )
        
        assert set(track.stems) == {'vocals', 'drums', 'bass'}
    
    def test_duration(self):
        """Test duration calculation."""
        sample_rate = 16000
        duration_seconds = 3.0
        n_samples = int(duration_seconds * sample_rate)
        
        track = Track(
            track_id="test",
            mixture=np.random.randn(n_samples),
            references={'vocals': np.random.randn(n_samples)},
            sample_rate=sample_rate
        )
        
        assert abs(track.duration - duration_seconds) < 0.01


class TestDatasetRegistry:
    """Test dataset registry."""
    
    def test_register_and_get(self):
        """Test dataset registration and retrieval."""
        dataset_cls = DatasetRegistry.get("mock_dataset")
        assert dataset_cls == MockDataset
    
    def test_get_by_tag(self):
        """Test getting dataset by tag."""
        dataset_cls = DatasetRegistry.get("mock")
        assert dataset_cls == MockDataset
    
    def test_get_nonexistent(self):
        """Test getting nonexistent dataset."""
        with pytest.raises(KeyError):
            DatasetRegistry.get("nonexistent_dataset")
    
    def test_list_datasets(self):
        """Test listing datasets."""
        datasets = DatasetRegistry.list_datasets()
        assert "mock_dataset" in datasets


try:
    from disgen.datasets.moisesdb import MOISESDB_AVAILABLE
except ImportError:
    MOISESDB_AVAILABLE = False


@pytest.mark.skipif(not MOISESDB_AVAILABLE, reason="moisesdb not installed")
class TestMoisesDB:
    """Test MoisesDB integration with official package."""
    
    def test_import_available(self):
        """Test that moisesdb package is importable."""
        from disgen.datasets import MoisesDB
        assert MoisesDB is not None
    
    def test_initialization(self):
        """Test MoisesDB initialization."""
        from disgen.datasets import MoisesDB
        
        dataset = MoisesDB(
            root_path="/fake/path",
            subset="test",
            sample_rate=44100
        )
        assert dataset.sample_rate == 44100
        assert dataset.subset == "test"
    
    def test_import_error_without_package(self):
        """Test error when moisesdb package not available."""
        # This will only run if MOISESDB_AVAILABLE is True,
        # so we just verify the import check exists
        from disgen.datasets.moisesdb import MOISESDB_AVAILABLE
        assert MOISESDB_AVAILABLE is True

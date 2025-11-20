"""Unit tests for model classes."""

import pytest
import numpy as np

from disgen.models.discriminative import MockDiscriminativeModel
from disgen.models.generative import MockGenerativeModel
from disgen.core.base_model import ModelRegistry, DiscriminativeModel, GenerativeModel


class TestMockDiscriminativeModel:
    """Test mock discriminative model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = MockDiscriminativeModel(domain="time")
        assert model.domain == "time"
        assert not model.is_loaded
    
    def test_load(self):
        """Test model loading."""
        model = MockDiscriminativeModel()
        model.load()
        assert model.is_loaded
    
    def test_separate(self):
        """Test separation."""
        model = MockDiscriminativeModel(stems=['vocals', 'drums'])
        model.load()
        
        # Create test mixture
        mixture = np.random.randn(16000)  # 1 second at 16kHz
        
        # Separate
        separated = model.separate(mixture, 16000)
        
        assert 'vocals' in separated
        assert 'drums' in separated
        assert separated['vocals'].shape == mixture.shape
    
    def test_domain_property(self):
        """Test domain property."""
        time_model = MockDiscriminativeModel(domain="time")
        freq_model = MockDiscriminativeModel(domain="frequency")
        
        assert time_model.domain == DiscriminativeModel.DOMAIN_TIME
        assert freq_model.domain == DiscriminativeModel.DOMAIN_FREQUENCY


class TestMockGenerativeModel:
    """Test mock generative model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = MockGenerativeModel()
        assert not model.is_loaded
    
    def test_load(self):
        """Test model loading."""
        model = MockGenerativeModel()
        model.load()
        assert model.is_loaded
    
    def test_refine(self):
        """Test refinement."""
        model = MockGenerativeModel(improvement_factor=1.2)
        model.load()
        
        # Create test audio
        audio = np.random.randn(16000)
        
        # Refine
        refined = model.refine(audio, 16000)
        
        assert refined.shape == audio.shape
        assert not np.array_equal(refined, audio)
    
    def test_vocode(self):
        """Test vocoding."""
        model = MockGenerativeModel()
        model.load()
        
        # Create test spectrogram
        spectrogram = np.random.randn(128, 100)  # (freq_bins, time_frames)
        
        # Vocode
        waveform = model.vocode(spectrogram)
        
        assert waveform.ndim == 1
        assert len(waveform) > 0
    
    def test_supported_modes(self):
        """Test supported modes."""
        model = MockGenerativeModel()
        
        assert GenerativeModel.MODE_REFINEMENT in model.supported_modes
        assert GenerativeModel.MODE_VOCODER in model.supported_modes


class TestModelRegistry:
    """Test model registry."""
    
    def test_register_and_get(self):
        """Test model registration and retrieval."""
        # Models should already be registered
        disc_cls = ModelRegistry.get("mock_discriminative")
        assert disc_cls == MockDiscriminativeModel
        
        gen_cls = ModelRegistry.get("mock_generative")
        assert gen_cls == MockGenerativeModel
    
    def test_get_by_tag(self):
        """Test getting model by tag."""
        disc_cls = ModelRegistry.get("mock_disc")
        assert disc_cls == MockDiscriminativeModel
    
    def test_get_nonexistent(self):
        """Test getting nonexistent model."""
        with pytest.raises(KeyError):
            ModelRegistry.get("nonexistent_model")
    
    def test_list_models(self):
        """Test listing models."""
        models = ModelRegistry.list_models()
        assert "mock_discriminative" in models
        assert "mock_generative" in models

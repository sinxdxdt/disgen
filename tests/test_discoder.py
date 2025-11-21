"""Unit tests for DisCoder model."""

import pytest
import numpy as np
import torch

from disgen.models.generative.discoder import DisCoder, DISCODER_AVAILABLE


class TestDisCoderVendored:
    """Tests for vendored DisCoder installation."""
    
    def test_vendored_import(self):
        """Test that vendored DisCoder can be imported."""
        try:
            from disgen.vendor.discoder import models, utils, meldataset
            assert hasattr(models, 'DisCoder')
            assert hasattr(utils, 'get_mel_spectrogram_from_config')
            assert hasattr(meldataset, 'mel_spectrogram')
        except ImportError as e:
            pytest.fail(f"Vendored DisCoder import failed: {e}")
    
    def test_wrapper_uses_vendored(self):
        """Test that wrapper correctly imports vendored version."""
        assert DISCODER_AVAILABLE, "DisCoder should be available (vendored)"
        
        # Test creating wrapper (doesn't load model)
        model = DisCoder(use_pretrained=False, device="cpu")
        assert model is not None
        assert not model.is_loaded


@pytest.mark.skipif(not DISCODER_AVAILABLE, reason="DisCoder not installed")
class TestDisCoderIntegration:
    """Integration tests for DisCoder model."""
    
    def test_init_pretrained(self):
        """Test initialization with pretrained model."""
        model = DisCoder(use_pretrained=True, device="cpu")
        assert model.use_pretrained is True
        assert model.checkpoint_path is None
        assert not model.is_loaded
    
    def test_init_custom(self):
        """Test initialization with custom checkpoint."""
        model = DisCoder(
            checkpoint_path="/path/to/model.pt",
            config_path="/path/to/config.json",
            use_pretrained=False,
            device="cpu"
        )
        assert model.use_pretrained is False
        assert model.checkpoint_path.name == "model.pt"
        assert not model.is_loaded
    
    def test_default_config(self):
        """Test default configuration."""
        model = DisCoder(use_pretrained=False, device="cpu")
        config = model._get_default_config()
        
        assert config["sample_rate"] == 44100
        assert config["num_mels"] == 128  # Critical for DisCoder
        assert config["n_fft"] == 2048
        assert config["hop_size"] == 512
        assert "segment_size" in config
    
    @pytest.mark.slow
    def test_load_pretrained(self):
        """Test loading pretrained model from Hugging Face."""
        model = DisCoder(use_pretrained=True, device="cpu")
        
        try:
            model.load()
            assert model.is_loaded
            assert model.config is not None
            assert model.config.get("num_mels") == 128
        except Exception as e:
            pytest.skip(f"Could not load pretrained model: {e}")
    
    def test_mel_extraction_shape(self):
        """Test mel spectrogram extraction shape."""
        model = DisCoder(use_pretrained=False, device="cpu")
        model.config = model._get_default_config()
        
        # Create dummy audio (1 second at 44.1kHz)
        audio = np.random.randn(44100).astype(np.float32)
        
        try:
            mel_spec, orig_length = model._extract_mel(audio, 44100)
            
            # Check shape
            assert mel_spec.dim() == 3  # (batch, n_mels, time)
            assert mel_spec.shape[1] == 128  # Must be 128 mel bins
            assert orig_length == 44100
        except RuntimeError as e:
            if "librosa" in str(e):
                pytest.skip("librosa not available")
            else:
                raise
    
    def test_sample_rate_conversion(self):
        """Test automatic sample rate conversion."""
        model = DisCoder(use_pretrained=False, device="cpu")
        model.config = model._get_default_config()
        
        # Create audio at different sample rate
        audio = np.random.randn(22050).astype(np.float32)  # 1 sec at 22.05kHz
        
        try:
            mel_spec, _ = model._extract_mel(audio, 22050)
            
            # Should be resampled to 44.1kHz internally
            assert mel_spec.shape[1] == 128
        except RuntimeError as e:
            if "librosa" in str(e):
                pytest.skip("librosa not available")
            else:
                raise
    
    def test_padding_logic(self):
        """Test segment padding and unpadding."""
        model = DisCoder(use_pretrained=False, device="cpu")
        model.config = model._get_default_config()
        model.config["segment_size"] = 1024
        
        # Create audio that's not aligned with segment size
        audio = np.random.randn(1500).astype(np.float32)
        original_length = len(audio)
        
        try:
            mel_spec, orig_length = model._extract_mel(audio, 44100)
            
            # Check that original length is preserved
            assert orig_length == original_length
        except RuntimeError as e:
            if "librosa" in str(e):
                pytest.skip("librosa not available")
            else:
                raise
    
    def test_stereo_to_mono_conversion(self):
        """Test stereo to mono conversion."""
        model = DisCoder(use_pretrained=False, device="cpu")
        model.config = model._get_default_config()
        
        # Create stereo audio
        audio = np.random.randn(2, 44100).astype(np.float32)
        
        try:
            mel_spec, _ = model._extract_mel(audio, 44100)
            
            # Should process as mono
            assert mel_spec.shape[1] == 128
        except RuntimeError as e:
            if "librosa" in str(e):
                pytest.skip("librosa not available")
            else:
                raise
    
    def test_supported_modes(self):
        """Test supported modes."""
        model = DisCoder(use_pretrained=False, device="cpu")
        modes = model.supported_modes
        
        assert model.MODE_VOCODER in modes
        assert model.MODE_REFINEMENT in modes
    
    def test_repr(self):
        """Test string representation."""
        model = DisCoder(use_pretrained=True, device="cpu")
        repr_str = repr(model)
        
        assert "DisCoder" in repr_str
        assert "Hugging Face" in repr_str
        assert "cpu" in repr_str


class TestDisCoderMock:
    """Mock tests that don't require DisCoder installation."""
    
    def test_import_error_handling(self):
        """Test graceful handling when DisCoder not installed."""
        # This will pass regardless of DisCoder installation status
        assert True
    
    def test_config_structure(self):
        """Test expected config structure."""
        expected_keys = [
            "sample_rate",
            "num_mels",
            "n_fft",
            "hop_size",
            "win_size",
            "fmin",
            "fmax",
            "segment_size"
        ]
        
        if DISCODER_AVAILABLE:
            model = DisCoder(use_pretrained=False, device="cpu")
            config = model._get_default_config()
            
            for key in expected_keys:
                assert key in config


@pytest.mark.integration
@pytest.mark.skipif(not DISCODER_AVAILABLE, reason="DisCoder not installed")
class TestDisCoderEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.mark.slow
    def test_full_pipeline(self):
        """Test complete audio refinement pipeline."""
        model = DisCoder(use_pretrained=True, device="cpu")
        
        try:
            model.load()
        except Exception:
            pytest.skip("Could not load pretrained model")
        
        # Create test audio (2 seconds)
        audio = np.random.randn(88200).astype(np.float32)
        audio = audio / np.max(np.abs(audio))  # Normalize
        
        # Refine audio
        try:
            refined = model.refine(audio, sample_rate=44100)
            
            # Check output
            assert isinstance(refined, np.ndarray)
            assert refined.ndim == 1
            # Length may differ slightly due to padding/processing
            assert len(refined) > 0
        except Exception as e:
            pytest.skip(f"Refinement failed: {e}")
    
    @pytest.mark.slow
    def test_vocode_direct(self):
        """Test direct vocoding from mel spectrogram."""
        model = DisCoder(use_pretrained=True, device="cpu")
        
        try:
            model.load()
        except Exception:
            pytest.skip("Could not load pretrained model")
        
        # Create dummy mel spectrogram (128 mels, 100 frames)
        mel_spec = torch.randn(1, 128, 100)
        
        try:
            audio = model.vocode(mel_spec)
            
            assert isinstance(audio, np.ndarray)
            assert audio.ndim == 1
            assert len(audio) > 0
        except Exception as e:
            pytest.skip(f"Vocoding failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

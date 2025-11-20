"""Unit tests for metrics."""

import pytest
import numpy as np

from disgen.metrics import BSSEvalMetrics, FrechetAudioDistance, SSIMSpectrogram
from disgen.core.base_metric import MetricRegistry


class TestBSSEvalMetrics:
    """Test BSS Eval metrics."""
    
    def test_initialization(self):
        """Test metric initialization."""
        metric = BSSEvalMetrics()
        assert metric.name == "bss_eval"
        assert metric.higher_is_better == True
    
    def test_compute(self):
        """Test metric computation."""
        metric = BSSEvalMetrics()
        
        # Create test audio
        reference = np.random.randn(2, 16000)  # Stereo, 1 sec at 16kHz
        estimated = reference + np.random.randn(2, 16000) * 0.1
        
        results = metric.compute(estimated, reference, 16000)
        
        assert 'sdr' in results
        assert 'sir' in results
        assert 'sar' in results
        assert 'snr' in results
    
    def test_compute_mono(self):
        """Test computation with mono audio."""
        metric = BSSEvalMetrics()
        
        reference = np.random.randn(16000)
        estimated = reference + np.random.randn(16000) * 0.1
        
        results = metric.compute(estimated, reference, 16000)
        
        assert all(k in results for k in ['sdr', 'sir', 'sar', 'snr'])


class TestFrechetAudioDistance:
    """Test FAD metric."""
    
    def test_initialization(self):
        """Test metric initialization."""
        metric = FrechetAudioDistance()
        assert metric.name == "fad"
        assert metric.higher_is_better == False
    
    def test_compute_single(self):
        """Test single sample computation returns NaN."""
        metric = FrechetAudioDistance()
        
        reference = np.random.randn(16000)
        estimated = np.random.randn(16000)
        
        result = metric.compute(estimated, reference, 16000)
        
        # Single FAD should return NaN (needs batch)
        assert 'fad' in result
        assert np.isnan(result['fad'])
    
    def test_batch_storage(self):
        """Test that samples are stored for batch computation."""
        metric = FrechetAudioDistance()
        
        # Add multiple samples
        for i in range(5):
            reference = np.random.randn(8000)
            estimated = np.random.randn(8000)
            metric.compute(estimated, reference, 16000)
        
        assert len(metric._estimated_samples) == 5
        assert len(metric._reference_samples) == 5


class TestSSIMSpectrogram:
    """Test SSIM metric."""
    
    def test_initialization(self):
        """Test metric initialization."""
        metric = SSIMSpectrogram()
        assert metric.name == "ssim"
        assert metric.higher_is_better == True
    
    def test_compute(self):
        """Test metric computation."""
        metric = SSIMSpectrogram()
        
        # Create test audio
        reference = np.random.randn(16000)
        estimated = reference + np.random.randn(16000) * 0.1
        
        result = metric.compute(estimated, reference, 16000)
        
        assert 'ssim' in result
        assert 0 <= result['ssim'] <= 1  # SSIM is typically in [0, 1]
    
    def test_compute_stereo(self):
        """Test with stereo audio."""
        metric = SSIMSpectrogram()
        
        reference = np.random.randn(2, 16000)
        estimated = reference + np.random.randn(2, 16000) * 0.1
        
        result = metric.compute(estimated, reference, 16000)
        
        assert 'ssim' in result
    
    def test_perfect_match(self):
        """Test SSIM with perfect match."""
        metric = SSIMSpectrogram()
        
        audio = np.random.randn(16000)
        
        result = metric.compute(audio, audio, 16000)
        
        # Perfect match should give high SSIM (close to 1)
        assert result['ssim'] > 0.9


class TestMetricRegistry:
    """Test metric registry."""
    
    def test_register_and_get(self):
        """Test metric registration and retrieval."""
        metric_cls = MetricRegistry.get("bss_eval")
        assert metric_cls == BSSEvalMetrics
        
        metric_cls = MetricRegistry.get("fad")
        assert metric_cls == FrechetAudioDistance
        
        metric_cls = MetricRegistry.get("ssim")
        assert metric_cls == SSIMSpectrogram
    
    def test_get_by_tag(self):
        """Test getting metric by tag."""
        # BSS Eval has tags for individual metrics
        metric_cls = MetricRegistry.get("sdr")
        assert metric_cls == BSSEvalMetrics
    
    def test_get_nonexistent(self):
        """Test getting nonexistent metric."""
        with pytest.raises(KeyError):
            MetricRegistry.get("nonexistent_metric")
    
    def test_list_metrics(self):
        """Test listing metrics."""
        metrics = MetricRegistry.list_metrics()
        assert "bss_eval" in metrics
        assert "fad" in metrics
        assert "ssim" in metrics


class TestMetricAggregation:
    """Test metric aggregation."""
    
    def test_aggregate_mean(self):
        """Test mean aggregation."""
        metric = BSSEvalMetrics()
        
        results = [
            {'sdr': 10.0, 'sir': 15.0},
            {'sdr': 12.0, 'sir': 17.0},
            {'sdr': 11.0, 'sir': 16.0},
        ]
        
        aggregated = metric.aggregate(results, method="mean")
        
        assert abs(aggregated['sdr'] - 11.0) < 0.01
        assert abs(aggregated['sir'] - 16.0) < 0.01
    
    def test_aggregate_median(self):
        """Test median aggregation."""
        metric = BSSEvalMetrics()
        
        results = [
            {'sdr': 10.0},
            {'sdr': 20.0},
            {'sdr': 15.0},
        ]
        
        aggregated = metric.aggregate(results, method="median")
        
        assert aggregated['sdr'] == 15.0
    
    def test_aggregate_with_nan(self):
        """Test aggregation with NaN values."""
        metric = BSSEvalMetrics()
        
        results = [
            {'sdr': 10.0},
            {'sdr': np.nan},
            {'sdr': 12.0},
        ]
        
        aggregated = metric.aggregate(results, method="mean")
        
        # Should ignore NaN
        assert abs(aggregated['sdr'] - 11.0) < 0.01

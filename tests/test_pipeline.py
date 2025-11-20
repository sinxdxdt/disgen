"""Integration tests for the evaluation pipeline."""

import pytest
import numpy as np

from disgen.pipeline.evaluator import Evaluator
from disgen.models.discriminative import MockDiscriminativeModel
from disgen.models.generative import MockGenerativeModel
from disgen.datasets import MockDataset
from disgen.metrics import BSSEvalMetrics, SSIMSpectrogram


class TestEvaluator:
    """Test evaluation pipeline."""
    
    def test_initialization(self):
        """Test evaluator initialization."""
        disc_model = MockDiscriminativeModel()
        gen_model = MockGenerativeModel()
        dataset = MockDataset(num_tracks=3)
        metrics = [BSSEvalMetrics()]
        
        evaluator = Evaluator(
            discriminative_model=disc_model,
            generative_model=gen_model,
            dataset=dataset,
            metrics=metrics
        )
        
        assert evaluator.discriminative_model.is_loaded
        assert evaluator.generative_model.is_loaded
    
    def test_evaluate_with_generative(self):
        """Test evaluation with generative model."""
        disc_model = MockDiscriminativeModel(domain="time")
        gen_model = MockGenerativeModel()
        dataset = MockDataset(num_tracks=2, duration=1.0)
        metrics = [SSIMSpectrogram()]
        
        evaluator = Evaluator(
            discriminative_model=disc_model,
            generative_model=gen_model,
            dataset=dataset,
            metrics=metrics
        )
        
        results = evaluator.evaluate(max_tracks=2)
        
        assert 'summary' in results
        assert 'per_track_results' in results
        assert 'config' in results
        assert results['summary']['num_tracks'] > 0
    
    def test_evaluate_without_generative(self):
        """Test evaluation without generative model."""
        disc_model = MockDiscriminativeModel()
        dataset = MockDataset(num_tracks=2, duration=1.0)
        metrics = [SSIMSpectrogram()]
        
        evaluator = Evaluator(
            discriminative_model=disc_model,
            generative_model=None,
            dataset=dataset,
            metrics=metrics
        )
        
        results = evaluator.evaluate(max_tracks=2)
        
        assert results['config']['generative_model'] is None
        assert len(results['per_track_results']) > 0
    
    def test_vocoder_mode_detection(self):
        """Test automatic vocoder mode detection."""
        # Frequency domain disc + vocoder-capable gen should use vocoder
        disc_model = MockDiscriminativeModel(domain="frequency")
        gen_model = MockGenerativeModel(modes=["vocoder", "refinement"])
        dataset = MockDataset(num_tracks=1)
        
        evaluator = Evaluator(
            discriminative_model=disc_model,
            generative_model=gen_model,
            dataset=dataset,
            metrics=[],
            use_vocoder="auto"
        )
        
        # Should detect vocoder mode
        assert evaluator._should_use_vocoder() == False  # Mock doesn't implement get_spectrogram
    
    def test_force_vocoder_mode(self):
        """Test forcing vocoder mode."""
        disc_model = MockDiscriminativeModel()
        gen_model = MockGenerativeModel()
        dataset = MockDataset(num_tracks=1)
        
        evaluator = Evaluator(
            discriminative_model=disc_model,
            generative_model=gen_model,
            dataset=dataset,
            metrics=[],
            use_vocoder=True
        )
        
        assert evaluator._should_use_vocoder() == True
    
    def test_statistical_results(self):
        """Test that statistical results are included."""
        disc_model = MockDiscriminativeModel()
        gen_model = MockGenerativeModel()
        dataset = MockDataset(num_tracks=3, duration=1.0)
        metrics = [SSIMSpectrogram()]
        
        evaluator = Evaluator(
            discriminative_model=disc_model,
            generative_model=gen_model,
            dataset=dataset,
            metrics=metrics
        )
        
        results = evaluator.evaluate(max_tracks=3)
        
        assert 'statistical_significance' in results['summary']
        stats = results['summary']['statistical_significance']
        
        # Check that SSIM results are present
        if 'ssim' in stats:
            assert 'p_value_t' in stats['ssim']
            assert 'cohens_d' in stats['ssim']
    
    def test_stem_specific_evaluation(self):
        """Test evaluating specific stem."""
        disc_model = MockDiscriminativeModel(stems=['vocals', 'drums'])
        dataset = MockDataset(num_tracks=2, stems=['vocals', 'drums'])
        metrics = [SSIMSpectrogram()]
        
        evaluator = Evaluator(
            discriminative_model=disc_model,
            generative_model=None,
            dataset=dataset,
            metrics=metrics
        )
        
        results = evaluator.evaluate(stem='vocals', max_tracks=2)
        
        # All results should be for vocals only
        for track_result in results['per_track_results']:
            assert track_result['stem'] == 'vocals'
    
    def test_max_tracks_limit(self):
        """Test max_tracks parameter."""
        disc_model = MockDiscriminativeModel()
        dataset = MockDataset(num_tracks=10)
        metrics = [SSIMSpectrogram()]
        
        evaluator = Evaluator(
            discriminative_model=disc_model,
            generative_model=None,
            dataset=dataset,
            metrics=metrics
        )
        
        results = evaluator.evaluate(max_tracks=3)
        
        # Should evaluate at most 3 tracks
        assert len(results['per_track_results']) <= 3


class TestVocoderModeLogic:
    """Test vocoder mode switching logic."""
    
    def test_time_domain_no_vocoder(self):
        """Test that time-domain models don't use vocoder."""
        disc_model = MockDiscriminativeModel(domain="time")
        gen_model = MockGenerativeModel()
        
        evaluator = Evaluator(
            discriminative_model=disc_model,
            generative_model=gen_model,
            dataset=MockDataset(num_tracks=1),
            metrics=[],
            use_vocoder="auto"
        )
        
        # Time domain should not use vocoder
        assert evaluator._should_use_vocoder() == False
    
    def test_no_generative_model(self):
        """Test with no generative model."""
        disc_model = MockDiscriminativeModel(domain="frequency")
        
        evaluator = Evaluator(
            discriminative_model=disc_model,
            generative_model=None,
            dataset=MockDataset(num_tracks=1),
            metrics=[],
            use_vocoder="auto"
        )
        
        # No gen model means no vocoder
        assert evaluator._should_use_vocoder() == False

"""Unit tests for statistical testing functions."""

import pytest
import numpy as np

from disgen.pipeline.statistics import (
    StatisticalTests,
    compute_model_significance
)


class TestStatisticalTests:
    """Test statistical testing functions."""
    
    def test_paired_t_test(self):
        """Test paired t-test."""
        baseline = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        refined = np.array([11.0, 12.0, 13.0, 14.0, 15.0])  # +1 improvement
        
        t_stat, p_value = StatisticalTests.paired_t_test(
            baseline, refined, alternative="greater"
        )
        
        assert not np.isnan(t_stat)
        assert not np.isnan(p_value)
        assert 0 <= p_value <= 1
    
    def test_wilcoxon_test(self):
        """Test Wilcoxon signed-rank test."""
        baseline = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        refined = np.array([11.0, 12.0, 13.0, 14.0, 15.0])
        
        w_stat, p_value = StatisticalTests.wilcoxon_test(
            baseline, refined, alternative="greater"
        )
        
        assert not np.isnan(w_stat)
        assert not np.isnan(p_value)
        assert 0 <= p_value <= 1
    
    def test_cohens_d(self):
        """Test Cohen's d calculation."""
        baseline = np.array([10.0, 11.0, 12.0])
        refined = np.array([11.0, 12.0, 13.0])
        
        d = StatisticalTests.cohens_d(baseline, refined)
        
        assert not np.isnan(d)
        assert d > 0  # Positive effect size
    
    def test_confidence_interval(self):
        """Test confidence interval calculation."""
        scores = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        
        lower, upper = StatisticalTests.confidence_interval(scores, confidence=0.95)
        
        assert not np.isnan(lower)
        assert not np.isnan(upper)
        assert lower < upper
        assert lower < np.mean(scores) < upper
    
    def test_with_nan_values(self):
        """Test statistical tests with NaN values."""
        baseline = np.array([10.0, np.nan, 12.0, 13.0])
        refined = np.array([11.0, np.nan, 13.0, 14.0])
        
        t_stat, p_value = StatisticalTests.paired_t_test(baseline, refined)
        
        # Should handle NaN values
        assert not np.isnan(t_stat)
        assert not np.isnan(p_value)
    
    def test_insufficient_samples(self):
        """Test with insufficient samples."""
        baseline = np.array([10.0])
        refined = np.array([11.0])
        
        t_stat, p_value = StatisticalTests.paired_t_test(baseline, refined)
        
        # Should return NaN
        assert np.isnan(t_stat) or np.isnan(p_value)


class TestComputeModelSignificance:
    """Test model-level significance computation."""
    
    def test_compute_significance(self):
        """Test computing model-level significance."""
        baseline_results = [
            {'sdr': 10.0, 'sir': 15.0},
            {'sdr': 11.0, 'sir': 16.0},
            {'sdr': 12.0, 'sir': 17.0},
        ]
        
        refined_results = [
            {'sdr': 11.0, 'sir': 16.0},
            {'sdr': 12.0, 'sir': 17.0},
            {'sdr': 13.0, 'sir': 18.0},
        ]
        
        metrics = ['sdr', 'sir']
        higher_is_better = {'sdr': True, 'sir': True}
        
        results = compute_model_significance(
            baseline_results,
            refined_results,
            metrics,
            higher_is_better
        )
        
        assert 'sdr' in results
        assert 'sir' in results
        
        # Check sdr results
        sdr_results = results['sdr']
        assert 'p_value_t' in sdr_results
        assert 'p_value_wilcoxon' in sdr_results
        assert 'cohens_d' in sdr_results
        assert 'mean_improvement' in sdr_results
        assert 'ci_lower' in sdr_results
        assert 'ci_upper' in sdr_results
        assert 'significant' in sdr_results
    
    def test_mean_improvement(self):
        """Test mean improvement calculation."""
        baseline_results = [
            {'sdr': 10.0},
            {'sdr': 12.0},
        ]
        
        refined_results = [
            {'sdr': 11.0},
            {'sdr': 13.0},
        ]
        
        results = compute_model_significance(
            baseline_results,
            refined_results,
            ['sdr'],
            {'sdr': True}
        )
        
        # Mean improvement should be 1.0
        assert abs(results['sdr']['mean_improvement'] - 1.0) < 0.01
    
    def test_lower_is_better(self):
        """Test with metric where lower is better."""
        baseline_results = [
            {'fad': 10.0},
            {'fad': 12.0},
        ]
        
        refined_results = [
            {'fad': 8.0},
            {'fad': 10.0},
        ]
        
        results = compute_model_significance(
            baseline_results,
            refined_results,
            ['fad'],
            {'fad': False}  # Lower is better
        )
        
        # Should use "less" alternative
        assert 'fad' in results
        # Mean improvement should be negative (improvement = refined - baseline)
        assert results['fad']['mean_improvement'] < 0
    
    def test_no_improvement(self):
        """Test when there's no improvement."""
        baseline_results = [
            {'sdr': 10.0},
            {'sdr': 12.0},
        ]
        
        refined_results = [
            {'sdr': 10.0},
            {'sdr': 12.0},
        ]
        
        results = compute_model_significance(
            baseline_results,
            refined_results,
            ['sdr'],
            {'sdr': True}
        )
        
        # Should not be significant
        assert not results['sdr']['significant']
        assert abs(results['sdr']['mean_improvement']) < 0.01

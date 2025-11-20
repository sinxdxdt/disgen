"""Statistical significance testing for evaluation results."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings


class StatisticalTests:
    """Statistical tests for source separation evaluation."""
    
    @staticmethod
    def paired_t_test(
        baseline_scores: np.ndarray,
        refined_scores: np.ndarray,
        alternative: str = "greater"
    ) -> Tuple[float, float]:
        """Perform paired t-test.
        
        Args:
            baseline_scores: Baseline (discriminative only) scores
            refined_scores: Refined (with generative) scores
            alternative: Alternative hypothesis ('greater', 'less', 'two-sided')
            
        Returns:
            Tuple of (t_statistic, p_value)
        """
        # Remove NaN values
        valid_mask = ~(np.isnan(baseline_scores) | np.isnan(refined_scores))
        baseline_valid = baseline_scores[valid_mask]
        refined_valid = refined_scores[valid_mask]
        
        if len(baseline_valid) < 2:
            warnings.warn("Not enough valid samples for t-test")
            return np.nan, np.nan
        
        t_stat, p_value = stats.ttest_rel(
            refined_valid,
            baseline_valid,
            alternative=alternative
        )
        
        return float(t_stat), float(p_value)
    
    @staticmethod
    def wilcoxon_test(
        baseline_scores: np.ndarray,
        refined_scores: np.ndarray,
        alternative: str = "greater"
    ) -> Tuple[float, float]:
        """Perform Wilcoxon signed-rank test (non-parametric).
        
        Args:
            baseline_scores: Baseline (discriminative only) scores
            refined_scores: Refined (with generative) scores
            alternative: Alternative hypothesis ('greater', 'less', 'two-sided')
            
        Returns:
            Tuple of (statistic, p_value)
        """
        # Remove NaN values
        valid_mask = ~(np.isnan(baseline_scores) | np.isnan(refined_scores))
        baseline_valid = baseline_scores[valid_mask]
        refined_valid = refined_scores[valid_mask]
        
        if len(baseline_valid) < 2:
            warnings.warn("Not enough valid samples for Wilcoxon test")
            return np.nan, np.nan
        
        # Compute differences
        differences = refined_valid - baseline_valid
        
        # Check if all differences are zero
        if np.all(differences == 0):
            return 0.0, 1.0
        
        try:
            result = stats.wilcoxon(
                differences,
                alternative=alternative
            )
            return float(result.statistic), float(result.pvalue)
        except Exception as e:
            warnings.warn(f"Error in Wilcoxon test: {e}")
            return np.nan, np.nan
    
    @staticmethod
    def cohens_d(
        baseline_scores: np.ndarray,
        refined_scores: np.ndarray
    ) -> float:
        """Compute Cohen's d effect size.
        
        Args:
            baseline_scores: Baseline scores
            refined_scores: Refined scores
            
        Returns:
            Cohen's d value
        """
        # Remove NaN values
        valid_mask = ~(np.isnan(baseline_scores) | np.isnan(refined_scores))
        baseline_valid = baseline_scores[valid_mask]
        refined_valid = refined_scores[valid_mask]
        
        if len(baseline_valid) < 2:
            return np.nan
        
        # Compute means
        mean_baseline = np.mean(baseline_valid)
        mean_refined = np.mean(refined_valid)
        
        # Compute pooled standard deviation
        std_baseline = np.std(baseline_valid, ddof=1)
        std_refined = np.std(refined_valid, ddof=1)
        n_baseline = len(baseline_valid)
        n_refined = len(refined_valid)
        
        pooled_std = np.sqrt(
            ((n_baseline - 1) * std_baseline**2 + (n_refined - 1) * std_refined**2) /
            (n_baseline + n_refined - 2)
        )
        
        if pooled_std == 0:
            return 0.0
        
        d = (mean_refined - mean_baseline) / pooled_std
        return float(d)
    
    @staticmethod
    def confidence_interval(
        scores: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Compute confidence interval for mean.
        
        Args:
            scores: Array of scores
            confidence: Confidence level (default: 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Remove NaN values
        valid_scores = scores[~np.isnan(scores)]
        
        if len(valid_scores) < 2:
            return np.nan, np.nan
        
        mean = np.mean(valid_scores)
        se = stats.sem(valid_scores)
        
        # Use t-distribution for CI
        h = se * stats.t.ppf((1 + confidence) / 2, len(valid_scores) - 1)
        
        return float(mean - h), float(mean + h)


def compute_model_significance(
    baseline_results: List[Dict[str, float]],
    refined_results: List[Dict[str, float]],
    metrics: List[str],
    higher_is_better: Dict[str, bool]
) -> Dict[str, Dict[str, float]]:
    """Compute model-level statistical significance across all tracks.
    
    Args:
        baseline_results: List of metric dictionaries for baseline (disc only)
        refined_results: List of metric dictionaries for refined (disc + gen)
        metrics: List of metric names to test
        higher_is_better: Dictionary mapping metric names to bool
        
    Returns:
        Dictionary with statistical test results for each metric:
        {
            'metric_name': {
                't_statistic': float,
                'p_value_t': float,
                'wilcoxon_statistic': float,
                'p_value_wilcoxon': float,
                'cohens_d': float,
                'mean_improvement': float,
                'ci_lower': float,
                'ci_upper': float,
                'significant': bool  # p < 0.05
            }
        }
    """
    results = {}
    
    for metric in metrics:
        # Extract scores for this metric
        baseline_scores = np.array([r.get(metric, np.nan) for r in baseline_results])
        refined_scores = np.array([r.get(metric, np.nan) for r in refined_results])
        
        # Determine alternative hypothesis
        alternative = "greater" if higher_is_better.get(metric, True) else "less"
        
        # Perform tests
        t_stat, p_t = StatisticalTests.paired_t_test(
            baseline_scores, refined_scores, alternative
        )
        w_stat, p_w = StatisticalTests.wilcoxon_test(
            baseline_scores, refined_scores, alternative
        )
        effect_size = StatisticalTests.cohens_d(baseline_scores, refined_scores)
        
        # Compute improvement
        improvements = refined_scores - baseline_scores
        mean_improvement = float(np.nanmean(improvements))
        ci_lower, ci_upper = StatisticalTests.confidence_interval(improvements)
        
        # Determine if significant (using both tests)
        alpha = 0.05
        significant = (p_t < alpha or p_w < alpha) and not np.isnan(p_t) and not np.isnan(p_w)
        
        results[metric] = {
            't_statistic': t_stat,
            'p_value_t': p_t,
            'wilcoxon_statistic': w_stat,
            'p_value_wilcoxon': p_w,
            'cohens_d': effect_size,
            'mean_improvement': mean_improvement,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': significant,
        }
    
    return results

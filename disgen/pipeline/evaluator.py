"""Main evaluation orchestrator."""

import numpy as np
from typing import Dict, List, Optional, Any
import warnings
from tqdm import tqdm

from disgen.core.base_model import DiscriminativeModel, GenerativeModel
from disgen.core.base_dataset import BaseDataset
from disgen.core.base_metric import BaseMetric
from disgen.pipeline.statistics import compute_model_significance
from disgen.pipeline.reporter import Reporter


class Evaluator:
    """Main evaluation orchestrator for source separation + generative refinement."""
    
    def __init__(
        self,
        discriminative_model: DiscriminativeModel,
        generative_model: Optional[GenerativeModel],
        dataset: BaseDataset,
        metrics: List[BaseMetric],
        use_vocoder: bool = "auto",
        allow_full_dataset: bool = False,
    ):
        """Initialize the evaluator.
        
        Args:
            discriminative_model: Discriminative separation model
            generative_model: Optional generative refinement/vocoder model
            dataset: Dataset to evaluate on
            metrics: List of metrics to compute
            use_vocoder: Whether to use vocoder mode ('auto', True, False)
                        'auto' uses vocoder if disc model is freq-domain and gen supports it
            allow_full_dataset: If True, evaluate on full dataset instead of test subset
        """
        self.discriminative_model = discriminative_model
        self.generative_model = generative_model
        self.dataset = dataset
        self.metrics = metrics
        self.use_vocoder = use_vocoder
        self.allow_full_dataset = allow_full_dataset
        
        # Load models
        if not self.discriminative_model.is_loaded:
            self.discriminative_model.load()
        
        if self.generative_model and not self.generative_model.is_loaded:
            self.generative_model.load()
        
        # Load dataset
        self.dataset.load()
        
        # Emit warning if using full dataset
        if allow_full_dataset and not dataset.has_test_split:
            warnings.warn(
                "⚠️  WARNING: Evaluating on full dataset instead of test subset. "
                "This may lead to unreliable, statistically insignificant results if "
                "models were trained on this dataset. Use at your own risk.",
                UserWarning
            )
    
    def _should_use_vocoder(self) -> bool:
        """Determine if vocoder mode should be used.
        
        Returns:
            True if vocoder mode should be used
        """
        if self.use_vocoder == True:
            return True
        elif self.use_vocoder == False:
            return False
        else:  # 'auto'
            # Use vocoder if disc model is freq-domain and gen model supports vocoding
            if not self.generative_model:
                return False
            
            is_freq_domain = (
                self.discriminative_model.domain == DiscriminativeModel.DOMAIN_FREQUENCY
            )
            supports_vocoding = (
                GenerativeModel.MODE_VOCODER in self.generative_model.supported_modes
            )
            
            return is_freq_domain and supports_vocoding
    
    def evaluate(
        self,
        stem: Optional[str] = None,
        max_tracks: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run evaluation.
        
        Args:
            stem: Optional specific stem to evaluate (e.g., 'vocals')
            max_tracks: Optional maximum number of tracks to evaluate
            
        Returns:
            Dictionary with results
        """
        # Get tracks to evaluate
        if self.allow_full_dataset:
            track_ids = self.dataset.get_all_tracks()
        else:
            track_ids = self.dataset.get_test_tracks()
        
        if max_tracks:
            track_ids = track_ids[:max_tracks]
        
        # Determine vocoder mode
        vocoder_mode = self._should_use_vocoder()
        
        # Collect results
        baseline_results = []  # Discriminative only
        refined_results = []  # Discriminative + Generative
        track_results = []
        
        # Evaluate each track
        for track_id in tqdm(track_ids, desc="Evaluating tracks"):
            track = self.dataset.get_track(track_id)
            
            # Determine which stems to evaluate
            stems_to_eval = [stem] if stem else track.stems
            
            for stem_name in stems_to_eval:
                if stem_name not in track.references:
                    continue
                
                # Get reference
                reference = track.references[stem_name]
                
                # Separate using discriminative model
                separated = self.discriminative_model.separate(
                    track.mixture,
                    track.sample_rate,
                    stem=stem_name
                )
                
                if stem_name not in separated:
                    warnings.warn(f"Stem '{stem_name}' not found in separation output")
                    continue
                
                estimated_baseline = separated[stem_name]
                
                # Apply generative refinement if model provided
                if self.generative_model:
                    if vocoder_mode:
                        # Use vocoder mode: get spectrogram from disc model
                        spectrograms = self.discriminative_model.get_spectrogram(
                            track.mixture,
                            track.sample_rate,
                            stem=stem_name
                        )
                        
                        if spectrograms and stem_name in spectrograms:
                            estimated_refined = self.generative_model.vocode(
                                spectrograms[stem_name]
                            )
                        else:
                            warnings.warn(
                                f"Vocoder mode requested but spectrogram not available "
                                f"for stem '{stem_name}'. Falling back to refinement mode."
                            )
                            estimated_refined = self.generative_model.refine(
                                estimated_baseline,
                                track.sample_rate,
                                stem_name=stem_name
                            )
                    else:
                        # Use normal refinement mode
                        estimated_refined = self.generative_model.refine(
                            estimated_baseline,
                            track.sample_rate,
                            stem_name=stem_name
                        )
                else:
                    # No generative model, refined same as baseline
                    estimated_refined = estimated_baseline
                
                # Compute metrics for baseline
                baseline_metrics = self._compute_metrics(
                    estimated_baseline,
                    reference,
                    track.sample_rate
                )
                baseline_results.append(baseline_metrics)
                
                # Compute metrics for refined
                refined_metrics = self._compute_metrics(
                    estimated_refined,
                    reference,
                    track.sample_rate
                )
                refined_results.append(refined_metrics)
                
                # Store track result
                track_results.append({
                    "track_id": track_id,
                    "stem": stem_name,
                    "baseline": baseline_metrics,
                    "refined": refined_metrics,
                })
        
        # Aggregate results
        aggregated_baseline = self._aggregate_results(baseline_results)
        aggregated_refined = self._aggregate_results(refined_results)
        
        # Compute statistical significance
        metric_names = list(aggregated_baseline.keys())
        higher_is_better = {
            metric.name: metric.higher_is_better
            for metric in self.metrics
        }
        
        statistical_results = compute_model_significance(
            baseline_results,
            refined_results,
            metric_names,
            higher_is_better
        )
        
        # Format results
        config = {
            "discriminative_model": self.discriminative_model.__class__.__name__,
            "generative_model": (
                self.generative_model.__class__.__name__
                if self.generative_model else None
            ),
            "dataset": self.dataset.__class__.__name__,
            "metrics": [m.name for m in self.metrics],
            "vocoder_mode": vocoder_mode,
            "stem": stem,
        }
        
        results = Reporter.format_results(
            track_results,
            aggregated_refined,
            statistical_results,
            config
        )
        
        return results
    
    def _compute_metrics(
        self,
        estimated: np.ndarray,
        reference: np.ndarray,
        sample_rate: int
    ) -> Dict[str, float]:
        """Compute all metrics for a single estimate.
        
        Args:
            estimated: Estimated audio
            reference: Reference audio
            sample_rate: Sample rate in Hz
            
        Returns:
            Dictionary of metric values
        """
        results = {}
        
        for metric in self.metrics:
            try:
                metric_values = metric.compute(estimated, reference, sample_rate)
                results.update(metric_values)
            except Exception as e:
                warnings.warn(f"Error computing {metric.name}: {e}")
                results[metric.name] = np.nan
        
        return results
    
    def _aggregate_results(
        self,
        results: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Aggregate results across tracks.
        
        Args:
            results: List of metric dictionaries
            
        Returns:
            Aggregated metric dictionary
        """
        if not results:
            return {}
        
        # Use first metric's aggregate method (they all use the same logic)
        if self.metrics:
            return self.metrics[0].aggregate(results)
        
        return {}

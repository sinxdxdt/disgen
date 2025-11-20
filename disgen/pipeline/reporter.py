"""Results reporting and formatting."""

import json
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path
import warnings


class Reporter:
    """Generate and export evaluation reports."""
    
    @staticmethod
    def format_results(
        track_results: List[Dict[str, Any]],
        aggregated_results: Dict[str, float],
        statistical_results: Dict[str, Dict[str, float]],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format results into a structured dictionary.
        
        Args:
            track_results: List of per-track results
            aggregated_results: Aggregated metrics across tracks
            statistical_results: Statistical significance test results
            config: Evaluation configuration
            
        Returns:
            Formatted results dictionary
        """
        return {
            "config": config,
            "summary": {
                "num_tracks": len(track_results),
                "aggregated_metrics": aggregated_results,
                "statistical_significance": statistical_results,
            },
            "per_track_results": track_results,
        }
    
    @staticmethod
    def export_json(results: Dict[str, Any], output_path: str):
        """Export results to JSON file.
        
        Args:
            results: Results dictionary
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    @staticmethod
    def export_csv(results: Dict[str, Any], output_path: str):
        """Export per-track results to CSV file.
        
        Args:
            results: Results dictionary
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(results["per_track_results"])
        df.to_csv(output_path, index=False)
    
    @staticmethod
    def export_markdown(results: Dict[str, Any], output_path: str):
        """Export results to Markdown file.
        
        Args:
            results: Results dictionary
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("# Music Source Separation Evaluation Results\n\n")
            
            # Configuration
            f.write("## Configuration\n\n")
            config = results["config"]
            f.write(f"- **Discriminative Model**: {config.get('discriminative_model', 'N/A')}\n")
            f.write(f"- **Generative Model**: {config.get('generative_model', 'N/A')}\n")
            f.write(f"- **Dataset**: {config.get('dataset', 'N/A')}\n")
            f.write(f"- **Metrics**: {', '.join(config.get('metrics', []))}\n")
            f.write(f"- **Number of Tracks**: {results['summary']['num_tracks']}\n\n")
            
            # Aggregated Results
            f.write("## Aggregated Results\n\n")
            agg = results["summary"]["aggregated_metrics"]
            
            if agg:
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                for metric, value in agg.items():
                    f.write(f"| {metric} | {value:.4f} |\n")
                f.write("\n")
            
            # Statistical Significance
            f.write("## Statistical Significance\n\n")
            stats = results["summary"]["statistical_significance"]
            
            if stats:
                f.write("| Metric | Mean Improvement | Cohen's d | p-value (t-test) | p-value (Wilcoxon) | Significant? |\n")
                f.write("|--------|------------------|-----------|------------------|--------------------|--------------|\n")
                for metric, stat_results in stats.items():
                    sig_marker = "✓" if stat_results.get("significant", False) else "✗"
                    f.write(
                        f"| {metric} | "
                        f"{stat_results.get('mean_improvement', 0):.4f} | "
                        f"{stat_results.get('cohens_d', 0):.4f} | "
                        f"{stat_results.get('p_value_t', 1):.4f} | "
                        f"{stat_results.get('p_value_wilcoxon', 1):.4f} | "
                        f"{sig_marker} |\n"
                    )
                f.write("\n")
                
                # Interpretation
                f.write("**Interpretation**:\n")
                f.write("- ✓ indicates statistically significant improvement (p < 0.05)\n")
                f.write("- Cohen's d: small (0.2), medium (0.5), large (0.8)\n\n")
            
            # Per-track results summary
            f.write("## Per-Track Results\n\n")
            f.write(f"See full results in the JSON or CSV export.\n")
            f.write(f"Total tracks evaluated: {results['summary']['num_tracks']}\n")
    
    @staticmethod
    def print_summary(results: Dict[str, Any]):
        """Print summary to console.
        
        Args:
            results: Results dictionary
        """
        print("\n" + "="*80)
        print("EVALUATION RESULTS SUMMARY")
        print("="*80 + "\n")
        
        config = results["config"]
        print(f"Discriminative Model: {config.get('discriminative_model', 'N/A')}")
        print(f"Generative Model: {config.get('generative_model', 'N/A')}")
        print(f"Dataset: {config.get('dataset', 'N/A')}")
        print(f"Number of Tracks: {results['summary']['num_tracks']}\n")
        
        # Aggregated metrics
        print("Aggregated Metrics:")
        print("-" * 40)
        agg = results["summary"]["aggregated_metrics"]
        for metric, value in agg.items():
            print(f"  {metric:20s}: {value:8.4f}")
        
        # Statistical significance
        print("\nStatistical Significance:")
        print("-" * 40)
        stats = results["summary"]["statistical_significance"]
        for metric, stat_results in stats.items():
            sig = "SIGNIFICANT" if stat_results.get("significant", False) else "not significant"
            print(f"  {metric:20s}: {sig} (p={stat_results.get('p_value_t', 1):.4f})")
        
        print("\n" + "="*80 + "\n")

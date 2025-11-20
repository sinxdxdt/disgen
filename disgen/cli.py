"""Command-line interface for disgen."""

import argparse
import sys
from pathlib import Path

from disgen.core.base_model import ModelRegistry
from disgen.core.base_dataset import DatasetRegistry
from disgen.core.base_metric import MetricRegistry
from disgen.pipeline.evaluator import Evaluator
from disgen.pipeline.reporter import Reporter
from disgen.config import Config

# Import to register models, datasets, metrics
import disgen.models.discriminative  # noqa
import disgen.models.generative  # noqa
import disgen.datasets  # noqa
import disgen.metrics  # noqa


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate generative refinement of discriminative music source separation"
    )
    
    # Model arguments
    parser.add_argument(
        "--discriminative-model",
        type=str,
        required=True,
        help="Name or tag of discriminative model"
    )
    parser.add_argument(
        "--generative-model",
        type=str,
        default=None,
        help="Name or tag of generative model (optional)"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name or tag of dataset"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="",
        help="Path to dataset root directory"
    )
    parser.add_argument(
        "--test-subset-size",
        type=float,
        default=Config.DEFAULT_TEST_SUBSET_SIZE,
        help="Fraction of dataset to use as test subset if no explicit split (default: 0.15)"
    )
    parser.add_argument(
        "--allow-full-dataset",
        action="store_true",
        help="Evaluate on full dataset instead of test subset (not recommended)"
    )
    
    # Metric arguments
    parser.add_argument(
        "--metrics",
        type=str,
        required=True,
        help="Comma-separated list of metrics (e.g., 'sdr,fad,ssim')"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Specific stem to evaluate (e.g., 'vocals'). If not specified, evaluates all stems"
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Maximum number of tracks to evaluate (for testing)"
    )
    parser.add_argument(
        "--use-vocoder",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="Whether to use vocoder mode (default: auto)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (JSON format). If not specified, uses default output dir"
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Also export results as CSV"
    )
    parser.add_argument(
        "--export-markdown",
        action="store_true",
        help="Also export results as Markdown"
    )
    
    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run models on ('cpu', 'cuda', 'mps')"
    )
    
    # Utility arguments
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit"
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List available metrics and exit"
    )
    
    args = parser.parse_args()
    
    # Handle list commands
    if args.list_models:
        print("Available discriminative models:")
        for model in ModelRegistry.list_models():
            print(f"  - {model}")
        return 0
    
    if args.list_datasets:
        print("Available datasets:")
        for dataset in DatasetRegistry.list_datasets():
            print(f"  - {dataset}")
        return 0
    
    if args.list_metrics:
        print("Available metrics:")
        for metric in MetricRegistry.list_metrics():
            print(f"  - {metric}")
        return 0
    
    # Initialize models
    print(f"Loading discriminative model: {args.discriminative_model}")
    disc_model_cls = ModelRegistry.get(args.discriminative_model)
    disc_model = disc_model_cls(device=args.device)
    
    gen_model = None
    if args.generative_model:
        print(f"Loading generative model: {args.generative_model}")
        gen_model_cls = ModelRegistry.get(args.generative_model)
        gen_model = gen_model_cls(device=args.device)
    
    # Initialize dataset
    print(f"Loading dataset: {args.dataset}")
    dataset_cls = DatasetRegistry.get(args.dataset)
    dataset = dataset_cls(
        root_path=args.dataset_path,
        test_subset_size=args.test_subset_size
    )
    
    # Initialize metrics
    metric_names = [m.strip() for m in args.metrics.split(",")]
    metrics = []
    for metric_name in metric_names:
        metric_cls = MetricRegistry.get(metric_name)
        metrics.append(metric_cls())
    
    print(f"Metrics: {', '.join(metric_names)}")
    
    # Convert use_vocoder to bool
    use_vocoder = args.use_vocoder
    if args.use_vocoder == "true":
        use_vocoder = True
    elif args.use_vocoder == "false":
        use_vocoder = False
    
    # Initialize evaluator
    evaluator = Evaluator(
        discriminative_model=disc_model,
        generative_model=gen_model,
        dataset=dataset,
        metrics=metrics,
        use_vocoder=use_vocoder,
        allow_full_dataset=args.allow_full_dataset
    )
    
    # Run evaluation
    print("\nStarting evaluation...")
    results = evaluator.evaluate(
        stem=args.stem,
        max_tracks=args.max_tracks
    )
    
    # Print summary
    Reporter.print_summary(results)
    
    # Export results
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Config.get_output_dir()
        output_path = output_dir / "results.json"
    
    print(f"\nExporting results to: {output_path}")
    Reporter.export_json(results, str(output_path))
    
    if args.export_csv:
        csv_path = output_path.with_suffix(".csv")
        Reporter.export_csv(results, str(csv_path))
        print(f"Exported CSV to: {csv_path}")
    
    if args.export_markdown:
        md_path = output_path.with_suffix(".md")
        Reporter.export_markdown(results, str(md_path))
        print(f"Exported Markdown to: {md_path}")
    
    print("\nEvaluation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

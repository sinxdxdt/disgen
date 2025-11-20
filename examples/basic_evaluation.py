"""Basic evaluation example using mock models."""

from disgen.models.discriminative import MockDiscriminativeModel
from disgen.models.generative import MockGenerativeModel
from disgen.datasets import MockDataset
from disgen.metrics import BSSEvalMetrics, SSIMSpectrogram, FrechetAudioDistance
from disgen.pipeline import Evaluator, Reporter


def main():
    """Run a basic evaluation with mock models."""
    
    print("="*80)
    print("BASIC EVALUATION EXAMPLE")
    print("="*80)
    print()
    
    # Initialize discriminative model
    print("1. Initializing discriminative model...")
    disc_model = MockDiscriminativeModel(
        domain="time",
        stems=['vocals', 'drums', 'bass', 'other']
    )
    
    # Initialize generative model
    print("2. Initializing generative model...")
    gen_model = MockGenerativeModel(
        improvement_factor=1.15  # Simulate 15% improvement
    )
    
    # Initialize dataset
    print("3. Initializing dataset...")
    dataset = MockDataset(
        num_tracks=10,
        duration=3.0,  # 3 seconds per track
        sample_rate=16000,
        has_test_split=True
    )
    
    # Initialize metrics
    print("4. Initializing metrics...")
    metrics = [
        BSSEvalMetrics(),
        SSIMSpectrogram(),
        # FrechetAudioDistance(),  # Commented out as it needs batch computation
    ]
    
    # Create evaluator
    print("5. Creating evaluator...")
    evaluator = Evaluator(
        discriminative_model=disc_model,
        generative_model=gen_model,
        dataset=dataset,
        metrics=metrics,
        use_vocoder="auto",
        allow_full_dataset=False  # Use test subset only
    )
    
    # Run evaluation
    print("6. Running evaluation...")
    print()
    results = evaluator.evaluate(
        stem='vocals',  # Evaluate only vocals
        max_tracks=5    # Limit to 5 tracks for quick demo
    )
    
    # Print summary
    print()
    Reporter.print_summary(results)
    
    # Export results
    print("7. Exporting results...")
    Reporter.export_json(results, "example_results.json")
    Reporter.export_markdown(results, "example_results.md")
    print("   - Saved to example_results.json")
    print("   - Saved to example_results.md")
    print()
    
    print("="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

"""Example using DisCoder vocoder for audio refinement."""

from disgen.models.discriminative import HTDemucs
from disgen.models.generative import DisCoder
from disgen.datasets import MUSDB18HQ
from disgen.metrics import BSSEvalMetrics, SSIMSpectrogram
from disgen.pipeline import Evaluator, Reporter
from disgen.utils import print_device_info


def main():
    """Run evaluation with HTDemucs + DisCoder refinement."""
    
    print("="*80)
    print("DISCODER REFINEMENT EXAMPLE")
    print("="*80)
    print()
    
    # Check GPU availability
    print("GPU Information:")
    print_device_info()
    print()
    
    # Initialize HTDemucs discriminative model
    print("1. Initializing HTDemucs model...")
    disc_model = HTDemucs(
        model_name="htdemucs_ft",
        device=None  # Auto-detect GPU
    )
    
    # Initialize DisCoder generative model
    print("2. Initializing DisCoder vocoder...")
    gen_model = DisCoder(
        use_pretrained=True,  # Load from Hugging Face
        device=None  # Auto-detect GPU
    )
    
    # Initialize dataset
    print("3. Initializing MUSDB18-HQ dataset...")
    dataset = MUSDB18HQ(
        root_path="/path/to/musdb18hq",  # Update this path
        subset="test",
        sample_rate=44100
    )
    
    # Initialize metrics
    print("4. Initializing metrics...")
    metrics = [
        BSSEvalMetrics(),
        SSIMSpectrogram(),
    ]
    
    # Create evaluator
    print("5. Creating evaluator...")
    evaluator = Evaluator(
        discriminative_model=disc_model,
        generative_model=gen_model,
        dataset=dataset,
        metrics=metrics,
        use_vocoder="auto"
    )
    
    # Run evaluation
    print("6. Running evaluation (this may take a while)...")
    print()
    results = evaluator.evaluate(
        stem='vocals',
        max_tracks=10  # Evaluate 10 tracks
    )
    
    # Print summary
    print()
    print("="*80)
    print("RESULTS")
    print("="*80)
    Reporter.print_summary(results)
    
    # Export results
    print()
    print("7. Exporting results...")
    Reporter.export_json(results, "discoder_results.json")
    Reporter.export_markdown(results, "discoder_results.md")
    Reporter.export_csv(results, "discoder_results.csv")
    print("   - Saved to discoder_results.json")
    print("   - Saved to discoder_results.md")
    print("   - Saved to discoder_results.csv")
    print()
    
    print("="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

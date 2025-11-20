"""Example: Implementing a custom discriminative model."""

import numpy as np
from typing import Dict, List, Optional

from disgen.core.base_model import DiscriminativeModel, ModelRegistry


@ModelRegistry.register("custom_separator", tags=["custom", "example"])
class CustomSeparator(DiscriminativeModel):
    """Example custom discriminative model.
    
    This is a simple example that demonstrates how to implement
    a custom discriminative model for use with disgen.
    """
    
    def __init__(self, device: str = "cpu", **kwargs):
        """Initialize the custom separator.
        
        Args:
            device: Device to run on
            **kwargs: Additional arguments
        """
        super().__init__(device=device, **kwargs)
        self._is_loaded = False
        
        # Your model initialization here
        # self.model = YourModel()
    
    def load(self):
        """Load the model weights."""
        # Load your model here
        # self.model.load_state_dict(torch.load('weights.pth'))
        # self.model.to(self.device)
        # self.model.eval()
        
        self._is_loaded = True
        print(f"Custom separator loaded on {self.device}")
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    def separate(
        self,
        mixture: np.ndarray,
        sample_rate: int,
        stem: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """Separate mixture into sources.
        
        Args:
            mixture: Audio mixture, shape (channels, samples) or (samples,)
            sample_rate: Sample rate in Hz
            stem: Optional specific stem to extract
            
        Returns:
            Dictionary mapping stem names to separated audio
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Your separation logic here
        # For this example, we'll just return copies of the mixture
        # In a real implementation, you would:
        # 1. Preprocess the mixture
        # 2. Run your model
        # 3. Postprocess the outputs
        
        stems_to_return = [stem] if stem else self.available_stems
        
        results = {}
        for stem_name in stems_to_return:
            # Placeholder: replace with actual separation
            results[stem_name] = mixture.copy()
        
        return results
    
    @property
    def domain(self) -> str:
        """Return the processing domain.
        
        Returns:
            Either DOMAIN_TIME or DOMAIN_FREQUENCY
        """
        # If your model outputs time-domain audio directly
        return self.DOMAIN_TIME
        
        # If your model outputs spectrograms
        # return self.DOMAIN_FREQUENCY
    
    @property
    def available_stems(self) -> List[str]:
        """Return list of stems this model can separate.
        
        Returns:
            List of stem names
        """
        return ['vocals', 'drums', 'bass', 'other']
    
    def get_spectrogram(
        self,
        mixture: np.ndarray,
        sample_rate: int,
        stem: Optional[str] = None
    ) -> Optional[Dict[str, np.ndarray]]:
        """Optional: Return spectrograms for vocoder mode.
        
        Implement this if your model is frequency-domain and you want
        to support using a generative model as a vocoder.
        
        Args:
            mixture: Audio mixture
            sample_rate: Sample rate in Hz
            stem: Optional specific stem
            
        Returns:
            Dictionary mapping stem names to spectrograms, or None
        """
        if self.domain == self.DOMAIN_TIME:
            return None
        
        # For frequency-domain models, implement spectrogram extraction
        # For example:
        # spectrograms = {}
        # for stem_name in self.available_stems:
        #     spec = self.model.get_spectrogram(mixture, stem_name)
        #     spectrograms[stem_name] = spec
        # return spectrograms
        
        return None


def main():
    """Example usage of custom model."""
    
    # Create and load model
    model = CustomSeparator(device="cpu")
    model.load()
    
    # Test separation
    mixture = np.random.randn(16000)  # 1 second at 16 kHz
    separated = model.separate(mixture, 16000, stem='vocals')
    
    print(f"Separated stems: {list(separated.keys())}")
    print(f"Vocals shape: {separated['vocals'].shape}")
    
    # Use with evaluator
    from disgen.datasets import MockDataset
    from disgen.metrics import SSIMSpectrogram
    from disgen.pipeline import Evaluator
    
    dataset = MockDataset(num_tracks=3)
    metrics = [SSIMSpectrogram()]
    
    evaluator = Evaluator(
        discriminative_model=model,
        generative_model=None,
        dataset=dataset,
        metrics=metrics
    )
    
    results = evaluator.evaluate(max_tracks=3)
    print(f"Evaluation complete: {results['summary']['num_tracks']} tracks")


if __name__ == "__main__":
    main()

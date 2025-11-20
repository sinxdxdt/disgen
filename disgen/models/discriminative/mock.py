"""Mock discriminative model for testing purposes."""

import numpy as np
from typing import Dict, List, Optional

from disgen.core.base_model import DiscriminativeModel, ModelRegistry


@ModelRegistry.register("mock_discriminative", tags=["mock_disc"])
class MockDiscriminativeModel(DiscriminativeModel):
    """Mock discriminative model for testing without real model dependencies."""
    
    def __init__(self, domain: str = "time", stems: Optional[List[str]] = None, **kwargs):
        """Initialize mock model.
        
        Args:
            domain: Processing domain ('time' or 'frequency')
            stems: List of stems to separate (default: ['vocals', 'drums', 'bass', 'other'])
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self._domain = domain
        self._stems = stems or ['vocals', 'drums', 'bass', 'other']
        self._is_loaded = False
    
    def load(self):
        """Load model (mock - does nothing)."""
        self._is_loaded = True
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    def separate(
        self,
        mixture: np.ndarray,
        sample_rate: int,
        stem: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """Mock separation - returns mixture with small random noise.
        
        Args:
            mixture: Audio mixture
            sample_rate: Sample rate
            stem: Optional specific stem
            
        Returns:
            Dictionary of separated stems
        """
        # Add small random noise to simulate separation
        stems_to_return = [stem] if stem else self._stems
        
        results = {}
        for stem_name in stems_to_return:
            # Add some random noise to mixture
            noise = np.random.randn(*mixture.shape) * 0.01
            results[stem_name] = mixture + noise
        
        return results
    
    @property
    def domain(self) -> str:
        return self._domain
    
    @property
    def available_stems(self) -> List[str]:
        return self._stems

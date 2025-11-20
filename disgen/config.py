"""Configuration management."""

import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Global configuration for disgen."""
    
    # Default paths
    DEFAULT_CACHE_DIR = Path.home() / ".cache" / "disgen"
    DEFAULT_OUTPUT_DIR = Path.cwd() / "disgen_results"
    
    # Default hyperparameters
    DEFAULT_SAMPLE_RATE = 441 00
    DEFAULT_TEST_SUBSET_SIZE = 0.15
    DEFAULT_SEED = 42
    
    # Metric defaults
    DEFAULT_BSS_EVAL_WINDOW = 44100
    DEFAULT_BSS_EVAL_HOP = 44100
    DEFAULT_FAD_MODEL = "vggish"
    DEFAULT_SSIM_N_FFT = 2048
    DEFAULT_SSIM_HOP_LENGTH = 512
    DEFAULT_SSIM_N_MELS = 128
    
    @classmethod
    def get_cache_dir(cls) -> Path:
        """Get cache directory, creating if it doesn't exist."""
        cache_dir = Path(os.getenv("DISGEN_CACHE_DIR", cls.DEFAULT_CACHE_DIR))
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    @classmethod
    def get_output_dir(cls) -> Path:
        """Get output directory, creating if it doesn't exist."""
        output_dir = Path(os.getenv("DISGEN_OUTPUT_DIR", cls.DEFAULT_OUTPUT_DIR))
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "cache_dir": str(cls.get_cache_dir()),
            "output_dir": str(cls.get_output_dir()),
            "sample_rate": cls.DEFAULT_SAMPLE_RATE,
            "test_subset_size": cls.DEFAULT_TEST_SUBSET_SIZE,
            "seed": cls.DEFAULT_SEED,
        }

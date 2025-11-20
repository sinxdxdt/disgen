"""Stub implementations for generative models.

Installation instructions for various generative models:

HiFiGAN:
    pip install torch
    # Manual implementation required

BigVGAN:
    pip install torch
    git clone https://github.com/NVIDIA/BigVGAN.git

DPM-TSE:
    # See https://github.com/bigpon/DPM-TSE for implementation details
"""

import warnings
from typing import List

from disgen.core.base_model import GenerativeModel, ModelRegistry


class HiFiGANStub:
    """Stub for HiFiGAN vocoder.
    
    To implement HiFiGAN:
    1. Install dependencies: pip install torch
    2. Download pretrained weights
    3. Implement GenerativeModel interface
    4. Register with @ModelRegistry.register("hifigan")
    """
    pass


class BigVGANStub:
    """Stub for BigVGAN vocoder.
    
    To implement BigVGAN:
    1. Clone repo: git clone https://github.com/NVIDIA/BigVGAN.git
    2. Follow installation instructions
    3. Implement GenerativeModel interface
    4. Register with @ModelRegistry.register("bigvgan")
    """
    pass


class DPMTSEStub:
    """Stub for DPM-TSE (Diffusion Probabilistic Model for Target Sound Extraction).
    
    To implement DPM-TSE:
    1. Follow https://github.com/bigpon/DPM-TSE for installation
    2. Implement GenerativeModel interface
    3. Register with @ModelRegistry.register("dpm_tse")
    """
    pass

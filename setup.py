"""Setup script for disgen package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# Core dependencies
install_requires = [
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "librosa>=0.9.0",
    "soundfile>=0.10.0",
    "scikit-image>=0.18.0",
    "pandas>=1.3.0",
    "tqdm>=4.60.0",
]

# Optional dependencies for metrics
metrics_requires = [
    "museval>=0.4.0",
    "fadtk>=1.0.0",
]

# Optional dependencies for models
models_requires = [
    "torch>=1.9.0",
    "torchaudio>=0.9.0",
]

# Test dependencies
test_requires = [
    "pytest>=7.0.0",
    "pytest-cov>=3.0.0",
]

# All optional dependencies
extras_require = {
    "metrics": metrics_requires,
    "models": models_requires,
    "test": test_requires,
    "all": metrics_requires + models_requires + test_requires,
}

setup(
    name="disgen",
    version="0.1.0",
    author="disgen contributors",
    description="Music Source Separation Evaluation Library for Generative Refinement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/disgen",  # Update with actual URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "disgen=disgen.cli:main",
        ],
    },
)

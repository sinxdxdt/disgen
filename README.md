# disgen - Music Source Separation Evaluation

## Features

- **Modular Architecture**: Easy-to-extend framework with registry pattern for models, datasets, and metrics
- **Multiple Models**: Support for discriminative models (HTDemucs, Spleeter, Open-Unmix, etc.) and generative refiners (HiFiGAN, BigVGAN, DPM-TSE, etc.)
- **Multiple Datasets**: MUSDB18-HQ, MoisesDB, Slakh2100 with automatic test subset handling
- **Comprehensive Metrics**: BSS Eval (SDR/SIR/SAR/SNR), Frechet Audio Distance, SSIM on spectrograms
- **Statistical Significance**: Paired t-tests, Wilcoxon tests, Cohen's d, confidence intervals, model-level p-values
- **Vocoder Mode**: Automatic detection and support for using generative models as vocoders
- **CLI Interface**: Complete command-line tool for running evaluations

## Installation

### Basic Installation

```bash
cd disgen
pip install -e .
```

This installs core dependencies including PyTorch for GPU support.

### Model-Specific Dependencies

```bash
# DisCoder vocoder (for neural codec-based refinement)
pip install git+https://github.com/ETH-DISCO/discoder.git

# HiFiGAN (requires manual setup)
# Clone https://github.com/jik876/hifi-gan and follow their instructions
```

### Development/Testing

```bash
pip install -e .[test]
```

## Quick Start

### Using Mock Models (No GPU Required)

```python
from disgen.models.discriminative import MockDiscriminativeModel
from disgen.models.generative import MockGenerativeModel
from disgen.datasets import MockDataset
from disgen.metrics import BSSEvalMetrics, SSIMSpectrogram
from disgen.pipeline import Evaluator, Reporter

# Initialize models
disc_model = MockDiscriminativeModel(domain="time")
gen_model = MockGenerativeModel()

# Initialize dataset
dataset = MockDataset(num_tracks=10, duration=5.0)

# Initialize metrics
metrics = [BSSEvalMetrics(), SSIMSpectrogram()]

# Create evaluator
evaluator = Evaluator(
    discriminative_model=disc_model,
    generative_model=gen_model,
    dataset=dataset,
    metrics=metrics
)

# Run evaluation
results = evaluator.evaluate(max_tracks=5)

# Print summary
Reporter.print_summary(results)

# Export results
Reporter.export_json(results, "results.json")
Reporter.export_markdown(results, "results.md")
```

### Using Real Models

```python
from disgen.models.discriminative import HTDemucs
from disgen.models.generative import HiFiGAN
from disgen.datasets import MUSDB18HQ
from disgen.metrics import BSSEvalMetrics, SSIMSpectrogram
from disgen.pipeline import Evaluator, Reporter
from disgen.utils import print_device_info

# Check GPU availability
print_device_info()

# Initialize HTDemucs (will use GPU if available)
disc_model = HTDemucs(
    model_name="htdemucs_ft",
    device=None  # Auto-detect GPU
)

# Initialize HiFiGAN vocoder
gen_model = HiFiGAN(
    checkpoint_path="/path/to/generator.pth",
    config_path="/path/to/config.json",
    device=None  # Auto-detect GPU
)

# Initialize MUSDB18-HQ dataset
dataset = MUSDB18HQ(
    root_path="/path/to/musdb18hq",
    subset="test"  # Use test split
)

# Initialize metrics
metrics = [BSSEvalMetrics(), SSIMSpectrogram()]

# Create evaluator
evaluator = Evaluator(
    discriminative_model=disc_model,
    generative_model=gen_model,
    dataset=dataset,
    metrics=metrics
)

# Run evaluation on GPU
results = evaluator.evaluate(stem="vocals", max_tracks=10)

# Print and export results
Reporter.print_summary(results)
Reporter.export_json(results, "results.json")
Reporter.export_markdown(results, "results.md")
```

### Using DisCoder Vocoder

```python
from disgen.models.discriminative import HTDemucs
from disgen.models.generative import DisCoder
from disgen.datasets import MUSDB18HQ
from disgen.metrics import BSSEvalMetrics, FrechetAudioDistance
from disgen.pipeline import Evaluator, Reporter

# Initialize HTDemucs
disc_model = HTDemucs(model_name="htdemucs_ft", device="cuda")

# Initialize DisCoder with pretrained model from Hugging Face
gen_model = DisCoder(
    use_pretrained=True,  # Loads disco-eth/discoder from HuggingFace
    device="cuda"
)

# Or use custom checkpoint
# gen_model = DisCoder(
#     checkpoint_path="/path/to/discoder.pt",
#     config_path="/path/to/config_z.json",
#     use_pretrained=False,
#     device="cuda"
# )

# Initialize dataset
dataset = MUSDB18HQ(root_path="/path/to/musdb18hq", subset="test")

# Create evaluator
evaluator = Evaluator(
    discriminative_model=disc_model,
    generative_model=gen_model,
    dataset=dataset,
    metrics=[BSSEvalMetrics(), FrechetAudioDistance()]
)

# Run evaluation - DisCoder will refine HTDemucs outputs
results = evaluator.evaluate(stem="vocals", max_tracks=10)

Reporter.print_summary(results)
Reporter.export_json(results, "results.json")
```

### Using CLI

```bash
# Check GPU availability
python -c "from disgen.utils import print_device_info; print_device_info()"

# List available models, datasets, and metrics
disgen --list-models
disgen --list-datasets
disgen --list-metrics

# Run with real models on GPU
disgen \
  --discriminative-model htdemucs \
  --generative-model hifigan \
  --dataset musdb18hq \
  --dataset-path /path/to/musdb18hq \
  --metrics sdr,sir,sar,ssim \
  --device cuda \
  --stem vocals \
  --max-tracks 10 \
  --output results.json \
  --export-markdown

# Run with mock models (no GPU)
disgen \
  --discriminative-model mock_discriminative \
  --generative-model mock_generative \
  --dataset mock_dataset \
  --metrics sdr,ssim \
  --max-tracks 5 \
  --output results.json
```

## Architecture

### Core Components

- **Base Classes** (`disgen/core/`):
  - `DiscriminativeModel`: Abstract base for separation models
  - `GenerativeModel`: Abstract base for refinement/vocoder models
  - `BaseDataset`: Abstract base for datasets
  - `BaseMetric`: Abstract base for metrics

- **Models** (`disgen/models/`):
  - Discriminative: HTDemucs, Spleeter, Open-Unmix, etc.
  - Generative: HiFiGAN, BigVGAN, DPM-TSE, etc.

- **Datasets** (`disgen/datasets/`):
  - MUSDB18-HQ (50 test tracks)
  - MoisesDB (custom sampling)
  - Slakh2100 (225 test tracks)

- **Metrics** (`disgen/metrics/`):
  - BSS Eval (SDR, SIR, SAR, SNR)
  - Frechet Audio Distance (FAD)
  - SSIM on spectrograms

- **Pipeline** (`disgen/pipeline/`):
  - `Evaluator`: Main orchestrator
  - `StatisticalTests`: Statistical significance testing
  - `Reporter`: Results formatting and export

## Adding Custom Components

### Custom Discriminative Model

```python
from disgen.core.base_model import DiscriminativeModel, ModelRegistry
import numpy as np

@ModelRegistry.register("my_model", tags=["custom"])
class MyModel(DiscriminativeModel):
    def load(self):
        # Load your model
        pass
    
    @property
    def is_loaded(self) -> bool:
        return True
    
    def separate(self, mixture, sample_rate, stem=None):
        # Implement separation logic
        return {'vocals': mixture}  # Example
    
    @property
    def domain(self) -> str:
        return self.DOMAIN_TIME
    
    @property
    def available_stems(self):
        return ['vocals', 'drums', 'bass', 'other']
```

### Custom Metric

```python
from disgen.core.base_metric import BaseMetric, MetricRegistry
import numpy as np

@MetricRegistry.register("my_metric")
class MyMetric(BaseMetric):
    def compute(self, estimated, reference, sample_rate, **kwargs):
        # Implement metric computation
        value = np.mean((estimated - reference) ** 2)
        return {'my_metric': float(value)}
    
    @property
    def name(self):
        return "my_metric"
    
    @property
    def higher_is_better(self):
        return False
```

## Signal Processing Flow

1. **Load mixture** from dataset
2. **Separate** using discriminative model
3. **Check domain**:
   - If frequency-domain model + vocoder-capable generative model → use vocoder mode
   - Otherwise → use refinement mode
4. **Apply generative refinement** (optional)
5. **Compute metrics** against ground truth
6. **Statistical testing** across all tracks
7. **Report results** with significance indicators

## Statistical Testing

Model-level statistical significance testing:

- **Paired t-test**: Parametric test for mean differences
- **Wilcoxon signed-rank test**: Non-parametric alternative
- **Cohen's d**: Effect size measurement
- **Confidence intervals**: 95% CI for mean improvements
- **p-value threshold**: α = 0.05

Results indicate whether generative refinement provides statistically significant improvement.

## Test Subset Enforcement

By default, evaluation runs only on test subsets to prevent data leakage:

- **Explicit test splits** (MUSDB18-HQ, Slakh2100): Uses predefined test sets
- **No explicit split** (MoisesDB): Samples 15% (configurable) as test set
- **Override**: Use `--allow-full-dataset` flag (emits warning)

## Configuration

Environment variables:

```bash
export DISGEN_CACHE_DIR=/path/to/cache
export DISGEN_OUTPUT_DIR=/path/to/output
```

## Testing

See [TESTING.md](TESTING.md) for detailed testing instructions.

```bash
# Run all tests (no GPU required)
pytest tests/ -v

# With coverage
pytest tests/ --cov=disgen --cov-report=html
```

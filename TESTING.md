# Testing Guide for disgen

This document provides comprehensive instructions for testing the disgen library.

## Table of Contents

1. [Unit Testing (No GPU Required)](#unit-testing-no-gpu-required)
2. [Integration Testing with Real Models](#integration-testing-with-real-models)
3. [Test Coverage](#test-coverage)
4. [Continuous Integration](#continuous-integration)

## Unit Testing (No GPU Required)

All unit tests use mock models and synthetic audio data, so they can run on any machine without GPU.

### Installation

```bash
cd disgen
pip install -e .[test]
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_models.py -v
pytest tests/test_datasets.py -v
pytest tests/test_metrics.py -v
pytest tests/test_statistics.py -v
pytest tests/test_pipeline.py -v

# Run specific test class
pytest tests/test_models.py::TestMockDiscriminativeModel -v

# Run specific test function
pytest tests/test_models.py::TestMockDiscriminativeModel::test_separate -v

# Run with output
pytest tests/ -v -s

# Stop on first failure
pytest tests/ -x

# Run in parallel (requires pytest-xdist)
pytest tests/ -n auto
```

### Test Coverage

```bash
# Generate coverage report
pytest tests/ --cov=disgen --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
# or
xdg-open htmlcov/index.html  # Linux
```

### Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Pytest configuration
├── test_models.py           # Model classes and registry tests
├── test_datasets.py         # Dataset classes and Track tests
├── test_metrics.py          # Metric implementations tests
├── test_statistics.py       # Statistical testing functions tests
└── test_pipeline.py         # Integration tests for evaluator
```

### What Gets Tested

#### test_models.py
- Mock model initialization and loading
- Separation and refinement logic
- Domain detection
- Model registry (register, get, list)
- Edge cases (nonexistent models, tags)

#### test_datasets.py
- Mock dataset initialization
- Track loading and properties
- Test split handling (explicit vs sampled)
- Dataset iteration
- Dataset registry

#### test_metrics.py
- BSS Eval metric computation (stereo and mono)
- FAD metric (single and batch)
- SSIM metric (perfect match, stereo)
- Metric aggregation (mean, median, with NaN)
- Metric registry

#### test_statistics.py
- Paired t-test
- Wilcoxon signed-rank test
- Cohen's d effect size
- Confidence intervals
- Model-level significance computation
- Handling of NaN values and edge cases

#### test_pipeline.py
- Evaluator initialization and model loading
- Evaluation with and without generative models
- Vocoder mode detection logic
- Statistical results inclusion
- Stem-specific evaluation
- Max tracks limiting

## Integration Testing with Real Models

Integration tests require GPU and real model installations.

### Prerequisites

```bash
# Install model dependencies
pip install torch torchaudio
# pip install demucs  # For HTDemucs
# pip install spleeter  # For Spleeter
# etc.

# Download datasets
# See dataset-specific instructions below
```

### Dataset Setup

#### MUSDB18-HQ

```bash
pip install musdb
# Download from https://zenodo.org/record/3338373
# Extract to /path/to/musdb18hq
```

#### MoisesDB

```bash
pip install moisesdb
# Use moisesdb Python API to download
```

#### Slakh2100

```bash
# Download from http://www.slakh.com/
# Extract to /path/to/slakh2100
```

### Running Integration Tests

```bash
# Example: HTDemucs on MUSDB18-HQ
disgen \
  --discriminative-model htdemucs \
  --dataset musdb18hq \
  --dataset-path /path/to/musdb18hq \
  --metrics sdr,sir,sar,ssim \
  --device cuda \
  --max-tracks 5 \
  --output test_results.json \
  --export-markdown

# Example: With generative refinement
disgen \
  --discriminative-model htdemucs \
  --generative-model hifigan \
  --dataset musdb18hq \
  --dataset-path /path/to/musdb18hq \
  --metrics sdr,fad \
  --device cuda \
  --output refined_results.json
```

### Expected Outputs

After running evaluation, you should see:

1. **Console output** with progress bar and summary
2. **JSON file** with detailed results
3. **CSV file** (if `--export-csv` specified) with per-track results
4. **Markdown file** (if `--export-markdown` specified) with formatted report

### Interpreting Results

#### Aggregated Metrics

- **SDR** (Signal-to-Distortion Ratio): Higher is better, typical range 5-15 dB
- **SIR** (Source-to-Interference Ratio): Higher is better
- **SAR** (Source-to-Artifact Ratio): Higher is better
- **FAD** (Frechet Audio Distance): Lower is better, typical range 0-10
- **SSIM**: Higher is better, range 0-1

#### Statistical Significance

- **p-value < 0.05**: Statistically significant improvement
- **Cohen's d**:
  - Small effect: d ≈ 0.2
  - Medium effect: d ≈ 0.5
  - Large effect: d ≈ 0.8
- **Confidence Interval**: 95% CI for mean improvement

#### Example Result Interpretation

```
| Metric | Mean Improvement | Cohen's d | p-value (t-test) | Significant? |
|--------|------------------|-----------|------------------|--------------|
| SDR    | 1.2 dB          | 0.65      | 0.003            | ✓            |
| FAD    | -0.8            | 0.45      | 0.021            | ✓            |
```

This indicates:
- SDR improved by 1.2 dB on average (medium-to-large effect)
- FAD decreased by 0.8 (improvement, as lower is better)
- Both improvements are statistically significant (p < 0.05)

## Test Coverage

Target coverage: > 80% for all modules

Current coverage (estimated):
- `disgen/core/`: ~90%
- `disgen/models/mock.py`: ~95%
- `disgen/datasets/mock.py`: ~90%
- `disgen/metrics/`: ~85%
- `disgen/pipeline/`: ~85%

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e .[test]
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=disgen --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## Troubleshooting

### Common Issues

#### ImportError for museval or fadtk

```bash
# Install metric dependencies
pip install museval fadtk
```

#### CUDA out of memory

```bash
# Use CPU instead
disgen --device cpu ...

# Or reduce max_tracks
disgen --max-tracks 5 ...
```

#### pytest not found

```bash
pip install pytest pytest-cov
```

### Debug Mode

```bash
# Run with verbose output and print statements
pytest tests/ -v -s

# Run single test with debugging
pytest tests/test_models.py::TestMockDiscriminativeModel::test_separate -v -s
```

## Performance Benchmarks

Expected test run times (on CPU):

- Unit tests: ~10-30 seconds
- Integration tests (5 tracks, no GPU): ~2-5 minutes
- Integration tests (50 tracks, GPU): ~10-30 minutes (depends on models)

## Contributing Tests

When adding new functionality:

1. **Write tests first** (TDD approach)
2. **Ensure coverage** of new code
3. **Test edge cases** (empty inputs, NaN values, etc.)
4. **Use fixtures** for common test data
5. **Document test purpose** in docstrings

Example test template:

```python
def test_new_feature():
    """Test that new feature works correctly.
    
    This tests:
    - Normal operation
    - Edge case X
    - Error handling
    """
    # Setup
    model = MyModel()
    
    # Execute
    result = model.new_feature(input_data)
    
    # Assert
    assert result is not None
    assert result.shape == expected_shape
```

## Questions?

For testing-related questions:
- Open an issue on GitHub
- Check existing test files for examples
- Review pytest documentation: https://docs.pytest.org/

# Testing & Benchmarking Directory

This directory contains all testing and benchmarking activities for the LEVEL3 Security Evaluation Framework.

## Directory Structure

### `unit_tests/`
- Unit tests for individual components
- Test individual functions, classes, and modules
- Fast, isolated tests that don't require external dependencies

### `integration_tests/`
- Integration tests for component interactions
- Test API integrations (OpenRouter, Ollama, HuggingFace)
- End-to-end CLI functionality tests
- Database/file I/O tests

### `benchmarks/`
- Performance benchmarking scripts
- Model inference speed tests
- Memory usage analysis
- Scalability testing

### `performance/`
- Load testing scripts
- Stress testing for high-throughput scenarios
- Resource utilization monitoring
- Comparative performance analysis

### `validation/`
- Dataset validation scripts
- Model output validation
- Security metric validation
- Configuration validation

## Usage Guidelines

1. **All test files should be placed in the appropriate subdirectory**
2. **Use descriptive naming**: `test_metric_jailbreak_sentinel.py`, `benchmark_openrouter_inference.py`
3. **Include requirements files** for any specific dependencies needed for testing
4. **Document test setup** in README files within each subdirectory
5. **Keep test data separate** from production code

## Running Tests

```bash
# Run all unit tests
cd testing_benchmarking/unit_tests
python -m pytest

# Run integration tests
cd testing_benchmarking/integration_tests
python -m pytest

# Run benchmarks
cd testing_benchmarking/benchmarks
python benchmark_script.py
```

## Test Data

- Store test datasets in `validation/test_datasets/`
- Use small, representative datasets for unit tests
- Include both positive and negative test cases
- Document data sources and licensing

## CI/CD Integration

This directory structure is designed to integrate with CI/CD pipelines:
- Unit tests run on every commit
- Integration tests run on pull requests
- Benchmarks run on scheduled builds
- Performance tests run on release candidates
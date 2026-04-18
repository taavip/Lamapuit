# Test Suite for Lamapuit CDW Detection

This directory contains the test suite for the CDW detection package.

## Structure

- `conftest.py` - Pytest configuration and shared fixtures
- `test_prepare.py` - Tests for data preparation module
- `test_detect.py` - Tests for detection module
- `test_train.py` - Tests for training module
- `test_integration.py` - Integration tests for full workflow

## Running Tests

### Run all tests:
```bash
pytest tests/
```

### Run with coverage report:
```bash
pytest tests/ --cov=src/cdw_detect --cov-report=html
```

### Run specific test file:
```bash
pytest tests/test_prepare.py
```

### Run with verbose output:
```bash
pytest tests/ -v
```

### Run only fast tests (skip slow integration tests):
```bash
pytest tests/ -m "not slow"
```

### Run in parallel (faster):
```bash
pytest tests/ -n auto
```

## Test Markers

- `@pytest.mark.slow` - Tests that take >1 second
- `@pytest.mark.integration` - Full workflow integration tests

## Coverage Goals

- Overall coverage: >70%
- Core modules (prepare, detect, train): >80%
- Edge case handling: >90%

## Adding New Tests

1. Create test file following naming convention: `test_<module>.py`
2. Use fixtures from `conftest.py` for common test data
3. Mark slow tests with `@pytest.mark.slow`
4. Add docstrings explaining what each test validates
5. Use descriptive test names: `test_<function>_<scenario>`

## Test Data

Tests use synthetic data created by fixtures in `conftest.py`:
- `sample_chm` - 1000x1000 pixel CHM with artificial features
- `sample_labels` - Vector line annotations matching CHM features
- `sample_dataset` - Complete YOLO dataset ready for training

Real data is NOT committed to the repository for testing.

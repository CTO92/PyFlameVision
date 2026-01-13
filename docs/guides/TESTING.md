# Testing Guide

> **PRE-RELEASE ALPHA SOFTWARE** - This project is currently in an early alpha stage. APIs may change without notice.

This guide covers how to run and write tests for PyFlameVision.

## Overview

PyFlameVision uses two testing frameworks:
- **GoogleTest** for C++ unit tests
- **pytest** for Python unit tests

---

## Running Tests

### Quick Start

```bash
# Run all C++ tests
cd build
ctest --output-on-failure

# Run all Python tests
pytest tests/python -v

# Run both
cd build && ctest && cd .. && pytest tests/python
```

### C++ Tests (GoogleTest)

```bash
# Build tests
cmake .. -DPYFLAME_VISION_BUILD_TESTS=ON
cmake --build .

# Run all tests
ctest

# Run with verbose output
ctest --output-on-failure

# Run specific test executable
./bin/test_transforms

# Run specific test case
./bin/test_transforms --gtest_filter="ResizeTest.*"

# Run specific test
./bin/test_transforms --gtest_filter="ResizeTest.SquareResize"

# List available tests
./bin/test_transforms --gtest_list_tests
```

#### GoogleTest Filters

```bash
# Wildcard patterns
--gtest_filter="*Resize*"           # All tests with "Resize"
--gtest_filter="*Test.Square*"      # All "Square" tests
--gtest_filter="-*Slow*"            # Exclude slow tests
--gtest_filter="Fast*:*Unit*"       # Fast OR Unit tests
```

### Python Tests (pytest)

```bash
# Run all tests
pytest tests/python

# Verbose output
pytest tests/python -v

# Very verbose (show print statements)
pytest tests/python -vv -s

# Run specific file
pytest tests/python/test_transforms.py

# Run specific class
pytest tests/python/test_transforms.py::TestResize

# Run specific test
pytest tests/python/test_transforms.py::TestResize::test_square_resize

# Run tests matching pattern
pytest tests/python -k "resize"
pytest tests/python -k "resize and not random"
```

#### pytest Options

```bash
# Stop on first failure
pytest --exitfirst  # or -x

# Run last failed tests only
pytest --lf

# Run failed tests first
pytest --ff

# Show local variables in tracebacks
pytest --showlocals  # or -l

# Generate JUnit XML report
pytest --junitxml=report.xml

# Parallel execution (requires pytest-xdist)
pytest -n auto
```

---

## Test Coverage

### Python Coverage

```bash
# Install coverage tools
pip install pytest-cov

# Run with coverage
pytest tests/python --cov=pyflame_vision

# HTML report
pytest tests/python --cov=pyflame_vision --cov-report=html
open htmlcov/index.html

# Require minimum coverage
pytest tests/python --cov=pyflame_vision --cov-fail-under=80
```

### C++ Coverage (gcov/lcov)

```bash
# Build with coverage flags
cmake .. -DCMAKE_BUILD_TYPE=Debug \
         -DCMAKE_CXX_FLAGS="--coverage"
cmake --build .

# Run tests
ctest

# Generate coverage report
lcov --capture --directory . --output-file coverage.info
lcov --remove coverage.info '/usr/*' --output-file coverage.info
genhtml coverage.info --output-directory coverage_html
open coverage_html/index.html
```

---

## Writing Tests

### C++ Test Structure

```cpp
#include <gtest/gtest.h>
#include <pyflame_vision/transforms/resize.hpp>

using namespace pyflame_vision::transforms;

// Test fixture for shared setup
class ResizeTest : public ::testing::Test {
protected:
    // Called before each test
    void SetUp() override {
        input_shape = {1, 3, 480, 640};
    }

    // Called after each test
    void TearDown() override {
        // Cleanup if needed
    }

    std::vector<int64_t> input_shape;
};

// Basic test
TEST_F(ResizeTest, SquareResize) {
    Resize resize(224);
    auto output = resize.get_output_shape(input_shape);

    EXPECT_EQ(output[0], 1);    // Batch preserved
    EXPECT_EQ(output[1], 3);    // Channels preserved
    EXPECT_EQ(output[2], 224);  // New height
    EXPECT_EQ(output[3], 224);  // New width
}

// Test with parameters
TEST_F(ResizeTest, RectangularResize) {
    Resize resize(Size(256, 512));
    auto output = resize.get_output_shape(input_shape);

    EXPECT_EQ(output[2], 256);
    EXPECT_EQ(output[3], 512);
}

// Test exception
TEST_F(ResizeTest, InvalidInputThrows) {
    Resize resize(224);
    std::vector<int64_t> bad_shape = {3, 224, 224};  // Missing batch

    EXPECT_THROW(
        resize.get_output_shape(bad_shape),
        std::runtime_error
    );
}

// Parameterized test
class ResizeParamTest : public ::testing::TestWithParam<
    std::tuple<int64_t, std::vector<int64_t>>
> {};

TEST_P(ResizeParamTest, VariousSizes) {
    auto [size, expected] = GetParam();
    Resize resize(size);

    auto output = resize.get_output_shape({1, 3, 480, 640});
    EXPECT_EQ(output, expected);
}

INSTANTIATE_TEST_SUITE_P(
    ResizeSizes,
    ResizeParamTest,
    ::testing::Values(
        std::make_tuple(224, std::vector<int64_t>{1, 3, 224, 224}),
        std::make_tuple(256, std::vector<int64_t>{1, 3, 256, 256}),
        std::make_tuple(512, std::vector<int64_t>{1, 3, 512, 512})
    )
);
```

### GoogleTest Assertions

```cpp
// Boolean assertions
EXPECT_TRUE(condition);
EXPECT_FALSE(condition);

// Equality
EXPECT_EQ(actual, expected);    // ==
EXPECT_NE(actual, expected);    // !=

// Comparison
EXPECT_LT(a, b);    // <
EXPECT_LE(a, b);    // <=
EXPECT_GT(a, b);    // >
EXPECT_GE(a, b);    // >=

// Floating point (with tolerance)
EXPECT_FLOAT_EQ(a, b);
EXPECT_DOUBLE_EQ(a, b);
EXPECT_NEAR(a, b, tolerance);

// String
EXPECT_STREQ(str1, str2);
EXPECT_STRNE(str1, str2);

// Exceptions
EXPECT_THROW(statement, exception_type);
EXPECT_NO_THROW(statement);
EXPECT_ANY_THROW(statement);

// Fatal assertions (stop test on failure)
ASSERT_EQ(actual, expected);  // Use ASSERT_ prefix
```

### Python Test Structure

```python
import pytest
from pyflame_vision.transforms import Resize, CenterCrop

class TestResize:
    """Tests for Resize transform."""

    # Fixtures (shared setup)
    @pytest.fixture
    def input_shape(self):
        return [1, 3, 480, 640]

    @pytest.fixture
    def resize(self):
        return Resize(224)

    # Basic test
    def test_square_resize(self, resize, input_shape):
        output = resize.get_output_shape(input_shape)
        assert output == [1, 3, 224, 224]

    # Test with expected exception
    def test_invalid_input_raises(self, resize):
        with pytest.raises(ValueError) as exc_info:
            resize.get_output_shape([3, 224, 224])

        assert "4D" in str(exc_info.value)

    # Parameterized test
    @pytest.mark.parametrize("size,expected", [
        (224, [1, 3, 224, 224]),
        (256, [1, 3, 256, 256]),
        ((256, 512), [1, 3, 256, 512]),
    ])
    def test_various_sizes(self, size, expected):
        resize = Resize(size)
        output = resize.get_output_shape([1, 3, 480, 640])
        assert output == expected

    # Skip test conditionally
    @pytest.mark.skipif(
        condition,
        reason="Requires feature X"
    )
    def test_advanced_feature(self):
        pass

    # Mark as slow (can be excluded with -m "not slow")
    @pytest.mark.slow
    def test_large_batch(self):
        pass
```

### pytest Fixtures

```python
# conftest.py - shared fixtures

import pytest

@pytest.fixture
def sample_rgb_shape():
    """Standard RGB image shape."""
    return [1, 3, 224, 224]

@pytest.fixture
def imagenet_mean():
    """ImageNet normalization mean."""
    return [0.485, 0.456, 0.406]

@pytest.fixture
def imagenet_std():
    """ImageNet normalization std."""
    return [0.229, 0.224, 0.225]

@pytest.fixture(scope="module")
def expensive_setup():
    """Shared across all tests in module."""
    # Setup
    resource = create_expensive_resource()
    yield resource
    # Teardown
    resource.cleanup()

@pytest.fixture(params=[224, 256, 512])
def various_sizes(request):
    """Parameterized fixture."""
    return request.param
```

### pytest Assertions

```python
# Equality
assert actual == expected
assert actual != unexpected

# Approximate equality (floats)
assert actual == pytest.approx(expected, rel=1e-5)
assert actual == pytest.approx(expected, abs=1e-10)

# Container assertions
assert item in container
assert len(container) == expected_length

# Exception assertions
with pytest.raises(ValueError):
    do_something_invalid()

with pytest.raises(ValueError, match="specific message"):
    do_something_invalid()

# Warnings
with pytest.warns(DeprecationWarning):
    deprecated_function()
```

---

## Test Organization

### Directory Structure

```
tests/
├── cpp/
│   ├── CMakeLists.txt
│   ├── test_image_tensor.cpp    # Core module tests
│   ├── test_transforms.cpp      # Transform tests
│   └── test_functional.cpp      # Functional API tests
└── python/
    ├── conftest.py              # Shared fixtures
    ├── test_transforms.py       # Transform tests
    ├── test_functional.py       # Functional API tests
    └── test_integration.py      # Integration tests
```

### Test Naming Conventions

```cpp
// C++: TestFixtureName.TestName
TEST_F(ResizeTest, SquareResize) {}
TEST_F(ResizeTest, RectangularResize) {}
TEST_F(ResizeTest, InvalidInputThrows) {}
```

```python
# Python: test_<what>_<condition/scenario>
def test_square_resize():
def test_resize_with_tuple_size():
def test_resize_invalid_input_raises():
```

---

## Integration Testing

### Testing Transform Pipelines

```python
def test_imagenet_pipeline():
    """Test complete ImageNet preprocessing pipeline."""
    from pyflame_vision.transforms import Compose, Resize, CenterCrop, Normalize

    pipeline = Compose([
        Resize(256),
        CenterCrop(224),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Test various input sizes
    for h, w in [(480, 640), (1080, 1920), (224, 224)]:
        input_shape = [1, 3, h, w]
        output_shape = pipeline.get_output_shape(input_shape)

        assert output_shape == [1, 3, 224, 224]
        assert pipeline.is_deterministic()
```

### Testing CSL Generation

```cpp
TEST(CSLGenerationTest, ResizeGeneratesValidCSL) {
    Resize resize(224);

    std::vector<int64_t> input_shape = {1, 3, 480, 640};
    auto layout = pyflame::MeshLayout::Grid(2, 2);

    std::string csl = resize.generate_csl(input_shape, layout);

    // Verify structure
    EXPECT_NE(csl.find("const TILE_H"), std::string::npos);
    EXPECT_NE(csl.find("task "), std::string::npos);

    // Verify no unsubstituted placeholders
    EXPECT_EQ(csl.find("{{"), std::string::npos);
}
```

---

## Continuous Integration

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  cpp-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Configure
        run: |
          mkdir build && cd build
          cmake .. -DPYFLAME_VISION_STANDALONE=ON

      - name: Build
        run: cmake --build build --parallel

      - name: Test
        run: cd build && ctest --output-on-failure

  python-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install pytest pytest-cov
          pip install -e .

      - name: Test
        run: pytest tests/python --cov=pyflame_vision
```

---

## Best Practices

1. **Test one thing per test** - Keep tests focused
2. **Use descriptive names** - Test name should describe what's tested
3. **Test edge cases** - Empty inputs, boundary values, errors
4. **Use fixtures** - Share setup code, avoid duplication
5. **Test both success and failure** - Verify error handling
6. **Keep tests fast** - Mark slow tests, run them separately
7. **Maintain test isolation** - Tests shouldn't depend on each other
8. **Update tests with code** - Keep tests in sync with implementation

---

## See Also

- [Contributing Guide](../CONTRIBUTING.md)
- [GoogleTest Documentation](https://google.github.io/googletest/)
- [pytest Documentation](https://docs.pytest.org/)

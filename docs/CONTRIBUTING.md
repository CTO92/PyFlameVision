# Contributing to PyFlameVision

Thank you for your interest in contributing to PyFlameVision! This guide covers everything you need to know to contribute effectively.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

---

## Code of Conduct

This project follows a standard code of conduct. Be respectful, inclusive, and constructive in all interactions.

---

## Getting Started

### Prerequisites

- C++17 compatible compiler (GCC 9+, Clang 10+, MSVC 2019+)
- CMake 3.18+
- Python 3.8+
- Git

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/PyFlameVision.git
cd PyFlameVision
git remote add upstream https://github.com/ORIGINAL_ORG/PyFlameVision.git
```

---

## Development Setup

### Build for Development

```bash
mkdir build && cd build

# Debug build with all features
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DPYFLAME_VISION_BUILD_TESTS=ON \
    -DPYFLAME_VISION_BUILD_EXAMPLES=ON \
    -DPYFLAME_VISION_STANDALONE=ON

cmake --build . --parallel
```

### Install Development Dependencies

```bash
# Python development dependencies
pip install -e ".[dev]"

# Or manually
pip install pytest pytest-cov black isort mypy
```

### IDE Setup

#### VS Code

Recommended extensions:
- C/C++ (Microsoft)
- CMake Tools
- Python
- clangd (for better C++ IntelliSense)

`.vscode/settings.json`:
```json
{
    "cmake.buildDirectory": "${workspaceFolder}/build",
    "cmake.configureArgs": ["-DPYFLAME_VISION_STANDALONE=ON"],
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/python"]
}
```

#### CLion

- Open as CMake project
- Set CMake options in Settings → Build → CMake

---

## Code Style

### C++ Style

We follow a style similar to Google C++ Style Guide with modifications:

```cpp
// Namespaces: snake_case
namespace pyflame_vision::transforms {

// Classes: PascalCase
class CenterCrop : public Transform {
public:
    // Methods: snake_case
    std::vector<int64_t> get_output_shape(
        const std::vector<int64_t>& input_shape
    ) const override;

    // Accessors: snake_case, no "get_" prefix
    const Size& size() const { return size_; }

private:
    // Members: snake_case with trailing underscore
    Size size_;
};

// Free functions: snake_case
void validate_image_shape(const std::vector<int64_t>& shape);

// Constants: SCREAMING_SNAKE_CASE
constexpr int MAX_CHANNELS = 4;

}  // namespace pyflame_vision::transforms
```

#### Formatting

Use clang-format with the project's `.clang-format` file:

```bash
# Format a file
clang-format -i src/transforms/resize.cpp

# Format all files
find src include -name "*.cpp" -o -name "*.hpp" | xargs clang-format -i
```

### Python Style

Follow PEP 8 with Black formatting:

```python
# Classes: PascalCase
class CenterCrop(Transform):
    """Center crop transform."""

    def __init__(self, size: Union[int, Tuple[int, int]]) -> None:
        # Use type hints
        self._size = _parse_size(size)

    def get_output_shape(self, input_shape: List[int]) -> List[int]:
        """Get output shape for given input."""
        _validate_image_shape(input_shape)
        return [...]

# Functions: snake_case
def _parse_size(size: Union[int, Tuple[int, int]]) -> Size:
    """Parse size argument."""
    ...

# Constants: SCREAMING_SNAKE_CASE
DEFAULT_INTERPOLATION = "bilinear"
```

#### Formatting

```bash
# Format with Black
black python/ tests/python/

# Sort imports
isort python/ tests/python/

# Type checking
mypy python/pyflame_vision/
```

### Documentation Style

- Use docstrings for all public APIs
- C++: Doxygen-style comments (`///` or `/** */`)
- Python: Google-style docstrings

```cpp
/// Resize image to target size.
///
/// @param size Target size
/// @param interpolation Interpolation method
/// @throws std::invalid_argument if size is invalid
Resize(const Size& size, InterpolationMode interpolation);
```

```python
def resize(size: int, interpolation: str = "bilinear") -> Resize:
    """Create a resize transform.

    Args:
        size: Target size for the output image.
        interpolation: Interpolation method. One of "nearest",
            "bilinear", "bicubic", "area".

    Returns:
        A Resize transform instance.

    Raises:
        ValueError: If size is not positive.

    Example:
        >>> resize = resize(224)
        >>> output_shape = resize.get_output_shape([1, 3, 480, 640])
    """
```

---

## Making Changes

### Branch Naming

```
feature/add-random-rotation
bugfix/fix-normalize-channel-check
docs/update-api-reference
refactor/simplify-compose-impl
```

### Commit Messages

Follow conventional commits:

```
feat(transforms): add RandomRotation transform

- Implement rotation with configurable angle range
- Support bilinear and nearest interpolation
- Add CSL template for WSE execution

Closes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

### Before Committing

1. Run tests: `ctest` and `pytest`
2. Format code: `clang-format` and `black`
3. Update documentation if needed
4. Add tests for new features

---

## Testing

### Running Tests

```bash
# C++ tests
cd build
ctest --output-on-failure

# Run specific test
./bin/test_transforms --gtest_filter="*Resize*"

# Python tests
pytest tests/python -v

# With coverage
pytest tests/python --cov=pyflame_vision --cov-report=html
```

### Writing Tests

#### C++ Tests (GoogleTest)

```cpp
#include <gtest/gtest.h>
#include <pyflame_vision/transforms/resize.hpp>

using namespace pyflame_vision::transforms;

class ResizeTest : public ::testing::Test {
protected:
    std::vector<int64_t> input_shape = {1, 3, 480, 640};
};

TEST_F(ResizeTest, SquareResize) {
    Resize resize(224);
    auto output = resize.get_output_shape(input_shape);

    EXPECT_EQ(output[2], 224);
    EXPECT_EQ(output[3], 224);
}

TEST_F(ResizeTest, InvalidInputThrows) {
    Resize resize(224);
    EXPECT_THROW(
        resize.get_output_shape({3, 224, 224}),  // Missing batch dim
        std::runtime_error
    );
}
```

#### Python Tests (pytest)

```python
import pytest
from pyflame_vision.transforms import Resize

class TestResize:
    def test_square_resize(self):
        resize = Resize(224)
        output = resize.get_output_shape([1, 3, 480, 640])
        assert output == [1, 3, 224, 224]

    def test_invalid_input_raises(self):
        resize = Resize(224)
        with pytest.raises(ValueError):
            resize.get_output_shape([3, 224, 224])

    @pytest.mark.parametrize("size,expected", [
        (224, [1, 3, 224, 224]),
        ((256, 512), [1, 3, 256, 512]),
    ])
    def test_various_sizes(self, size, expected):
        resize = Resize(size)
        assert resize.get_output_shape([1, 3, 480, 640]) == expected
```

### Test Coverage Requirements

- New features: Minimum 80% coverage
- Bug fixes: Add regression test
- All public API methods must be tested

---

## Documentation

### When to Update Docs

- New public API → Update API reference
- New feature → Add to Quick Start
- Breaking change → Update migration guide
- Configuration change → Update installation guide

### Building Documentation

```bash
# If using Sphinx/MkDocs (future)
cd docs
make html

# Preview
python -m http.server -d _build/html
```

### Documentation Structure

```
docs/
├── README.md              # Documentation index
├── INSTALLATION.md        # Build and install guide
├── QUICKSTART.md          # Getting started
├── ARCHITECTURE.md        # System design
├── CONTRIBUTING.md        # This file
├── api/
│   ├── CORE.md           # Core module reference
│   ├── TRANSFORMS.md     # Transforms reference
│   └── FUNCTIONAL.md     # Functional API reference
└── guides/
    ├── CSL_TEMPLATES.md  # CSL development guide
    └── TESTING.md        # Testing guide
```

---

## Pull Request Process

### 1. Prepare Your Changes

```bash
# Update your fork
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/my-feature

# Make changes, commit
git add .
git commit -m "feat: add my feature"
```

### 2. Pre-PR Checklist

- [ ] All tests pass (`ctest` and `pytest`)
- [ ] Code is formatted (`clang-format`, `black`)
- [ ] Documentation is updated
- [ ] Commit messages follow convention
- [ ] No unrelated changes included

### 3. Create Pull Request

```bash
git push origin feature/my-feature
```

Then create PR on GitHub with:
- Clear title describing the change
- Description of what and why
- Link to related issues
- Screenshots/examples if applicable

### 4. Code Review

- Respond to feedback promptly
- Make requested changes in new commits
- Squash only when asked or before merge

### 5. After Merge

```bash
git checkout main
git pull upstream main
git branch -d feature/my-feature
```

---

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue with reproduction steps
- **Security**: Email security@example.com (do not open public issue)

---

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project README

Thank you for contributing to PyFlameVision!

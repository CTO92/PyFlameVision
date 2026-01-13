# Installation Guide

This guide covers how to build and install PyFlameVision from source.

## Prerequisites

### Required

- **C++ Compiler**: GCC 9+, Clang 10+, or MSVC 2019+
- **CMake**: Version 3.18 or higher
- **Python**: Version 3.8 or higher (for Python bindings)

### Optional

- **PyFlame**: For full Cerebras WSE integration
- **CUDA Toolkit**: For GPU-accelerated fallback operations
- **Ninja**: For faster builds

## Quick Install

### From Source (Standalone Mode)

```bash
# Clone the repository
git clone https://github.com/your-org/PyFlameVision.git
cd PyFlameVision

# Create build directory
mkdir build && cd build

# Configure (standalone mode, no PyFlame dependency)
cmake .. -DPYFLAME_VISION_STANDALONE=ON

# Build
cmake --build . --parallel

# Install (optional)
cmake --install . --prefix /usr/local
```

### Python Package

```bash
# Install Python package in development mode
cd PyFlameVision
pip install -e .

# Or build wheel
pip install build
python -m build
pip install dist/pyflame_vision-*.whl
```

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `PYFLAME_VISION_BUILD_PYTHON` | ON | Build Python bindings |
| `PYFLAME_VISION_BUILD_TESTS` | ON | Build unit tests |
| `PYFLAME_VISION_BUILD_EXAMPLES` | ON | Build example programs |
| `PYFLAME_VISION_STANDALONE` | OFF | Build without PyFlame |

### Examples

```bash
# Minimal build (no Python, no tests)
cmake .. -DPYFLAME_VISION_BUILD_PYTHON=OFF \
         -DPYFLAME_VISION_BUILD_TESTS=OFF \
         -DPYFLAME_VISION_BUILD_EXAMPLES=OFF

# Debug build
cmake .. -DCMAKE_BUILD_TYPE=Debug

# With PyFlame integration
cmake .. -DPYFLAME_DIR=/path/to/PyFlame
```

## Platform-Specific Instructions

### Linux

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-dev

# Build
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### macOS

```bash
# Install dependencies
brew install cmake python

# Build
mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
```

### Windows

```powershell
# Using Visual Studio Developer Command Prompt
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

Or with Ninja:

```powershell
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja
```

## Integration with PyFlame

For full Cerebras WSE support, build with PyFlame:

```bash
# Option 1: Adjacent directory
# Place PyFlame source at ../PyFlame
cmake ..

# Option 2: Specify path
cmake .. -DPYFLAME_DIR=/path/to/PyFlame

# Option 3: Installed PyFlame
cmake .. -DPyFlame_DIR=/path/to/pyflame/cmake
```

## Verifying Installation

### C++ Library

```bash
# Run tests
cd build
ctest --output-on-failure

# Run example
./bin/imagenet_transforms
```

### Python Package

```python
import pyflame_vision
print(pyflame_vision.__version__)

from pyflame_vision.transforms import Resize, Compose
pipeline = Compose([Resize(224)])
print(pipeline)
```

```bash
# Run Python tests
pytest tests/python -v
```

## Troubleshooting

### CMake can't find Python

```bash
cmake .. -DPython3_EXECUTABLE=/path/to/python3
```

### pybind11 not found

pybind11 is automatically downloaded via FetchContent. If you have issues:

```bash
# Use system pybind11
pip install pybind11
cmake .. -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
```

### Tests fail to link

Ensure GoogleTest is properly built:

```bash
# Clean rebuild
rm -rf build
mkdir build && cd build
cmake .. -DPYFLAME_VISION_BUILD_TESTS=ON
cmake --build .
```

### Python import fails

Check that the extension module was built:

```bash
# Look for the .so/.pyd file
find build -name "_pyflame_vision*"

# Add to Python path
export PYTHONPATH=$PWD/build/python:$PYTHONPATH
```

## Next Steps

- Read the [Quick Start Guide](./QUICKSTART.md)
- Explore the [API Reference](./api/TRANSFORMS.md)
- Check out the [Examples](../examples/)

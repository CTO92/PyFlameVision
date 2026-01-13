# PyFlameVision

> **PRE-RELEASE ALPHA SOFTWARE**
>
> This project is currently in an early alpha stage of development. APIs may change without notice, features may be incomplete, and the software is not yet recommended for production use. Use at your own risk.

**Cerebras-native Computer Vision Library**

PyFlameVision is a computer vision library designed for the Cerebras Wafer Scale Engine (WSE), providing PyTorch-compatible APIs for image transforms, datasets, and vision operations.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

## Features

- **Image Transforms**: Resize, crop, normalize, flip, rotate, color jitter, blur
- **Datasets**: Map-style and iterable datasets with efficient DataLoader
- **Vision Ops**: Grid sampling, ROI align, Non-Maximum Suppression (NMS)
- **Models**: ResNet, EfficientNet architectures (C++ headers)
- **PyTorch-compatible API**: Familiar interface for torchvision users
- **Hardware Acceleration**: Optimized for Cerebras WSE via CSL templates

## Installation

### From Source

```bash
git clone https://github.com/pyflame/pyflame-vision.git
cd pyflame-vision
pip install .
```

### Development Installation

```bash
git clone https://github.com/pyflame/pyflame-vision.git
cd pyflame-vision
pip install -e ".[dev]"
```

### With PyFlame Integration

For full hardware acceleration, set the `PYFLAME_DIR` environment variable before building:

```bash
export PYFLAME_DIR=/path/to/pyflame
pip install .
```

## Quick Start

### Image Transforms

```python
import pyflame_vision.transforms as T

# Create a transform pipeline
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply to an image
from pyflame_vision.io import read_image
img = read_image("photo.jpg")
transformed = transform(img)
```

### Datasets and DataLoader

```python
from pyflame_vision.datasets import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Create dataloader with batching
dataset = MyDataset([1, 2, 3, 4, 5])
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

for batch in loader:
    print(batch)
```

### Vision Operations

```python
from pyflame_vision.ops import nms, roi_align, grid_sample

# Non-Maximum Suppression
keep = nms(boxes, scores, iou_threshold=0.5)

# ROI Align for object detection
pooled = roi_align(features, boxes, output_size=(7, 7))

# Grid sampling for spatial transformations
output = grid_sample(input, grid, mode='bilinear')
```

## Architecture

PyFlameVision is designed as a modular library:

```
pyflame_vision/
    transforms/     # Image transformation operations
    datasets/       # Dataset and DataLoader implementations
    io/             # Image I/O utilities
    ops/            # Vision-specific operations
```

### Relationship to PyFlame

PyFlameVision can operate in two modes:

1. **Standalone Mode**: Full Python functionality without PyFlame dependency
2. **Integrated Mode**: Hardware-accelerated operations via PyFlame's Cerebras backend

## Documentation

- [Installation Guide](docs/INSTALLATION.md)
- [Quick Start](docs/QUICKSTART.md)
- [API Reference](docs/api/)
- [Architecture](docs/ARCHITECTURE.md)
- [Contributing](docs/CONTRIBUTING.md)

## Requirements

- Python 3.8+
- NumPy >= 1.20.0
- Pillow >= 8.0.0
- CMake >= 3.18 (for building from source)
- C++17 compiler (for building from source)

Optional:
- PyFlame (for hardware acceleration)

## Testing

```bash
# Run Python tests
pytest tests/python/

# Run C++ tests (after building)
cd build && ctest --output-on-failure
```

## License

PyFlameVision is released under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our [Contributing Guide](docs/CONTRIBUTING.md) before submitting pull requests.

## Related Projects

- [PyFlame](https://github.com/pyflame/pyflame) - Core tensor library for Cerebras WSE
- [torchvision](https://github.com/pytorch/vision) - PyTorch's computer vision library (API inspiration)

## Citation

If you use PyFlameVision in your research, please cite:

```bibtex
@software{pyflame_vision,
  title = {PyFlameVision: Cerebras-native Computer Vision Library},
  year = {2024},
  url = {https://github.com/pyflame/pyflame-vision}
}
```

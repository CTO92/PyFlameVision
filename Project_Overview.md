# PyFlameVision

## Project Overview

PyFlameVision is a Cerebras-native computer vision library that provides PyFlame-compatible implementations of image processing transforms, vision model architectures, and dataset utilities. It serves as the Cerebras equivalent to PyTorch's torchvision library, enabling vision-based deep learning workloads to run on Cerebras Wafer-Scale Engine (WSE) hardware.

## Purpose

The torchvision library relies heavily on CUDA for GPU-accelerated image processing and model inference. PyFlameVision replaces these CUDA-dependent operations with Cerebras-optimized implementations that:

1. Generate CSL (Cerebras Software Language) code for WSE execution
2. Leverage the 2D PE mesh architecture for parallel image processing
3. Maintain API compatibility with torchvision for easy migration
4. Support lazy evaluation consistent with PyFlame's execution model

## Target Library Replacement

| Original Library | Version | CUDA Dependencies |
|------------------|---------|-------------------|
| torchvision | >= 0.21.0 | CUDA kernels for transforms, model ops |

## Core Components

### 1. Image Transforms

Cerebras-optimized preprocessing operations that build IR graph nodes for deferred execution.

#### Required Transforms

| Transform | torchvision Equivalent | Description |
|-----------|------------------------|-------------|
| `Resize` | `transforms.Resize` | Bilinear/bicubic image resizing |
| `CenterCrop` | `transforms.CenterCrop` | Center crop to target size |
| `RandomCrop` | `transforms.RandomCrop` | Random position cropping |
| `RandomHorizontalFlip` | `transforms.RandomHorizontalFlip` | Horizontal flip augmentation |
| `RandomRotation` | `transforms.RandomRotation` | Rotation augmentation |
| `Normalize` | `transforms.Normalize` | Channel-wise normalization |
| `ToTensor` | `transforms.ToTensor` | Convert image to PyFlame tensor |
| `ColorJitter` | `transforms.ColorJitter` | Brightness/contrast/saturation |
| `GaussianBlur` | `transforms.GaussianBlur` | Gaussian blur filter |
| `Compose` | `transforms.Compose` | Chain multiple transforms |

#### Functional API

Low-level functions matching `torchvision.transforms.functional`:

```python
import pyflame_vision.transforms.functional as F

F.resize(img, size, interpolation)
F.crop(img, top, left, height, width)
F.normalize(tensor, mean, std)
F.rotate(img, angle)
F.hflip(img)
F.vflip(img)
F.adjust_brightness(img, factor)
F.adjust_contrast(img, factor)
F.gaussian_blur(img, kernel_size, sigma)
```

### 2. Vision Models

Pre-built model architectures commonly used in biometric and recognition systems.

#### Required Model Families

| Model Family | Variants | Use Case in bioID |
|--------------|----------|-------------------|
| **ResNet** | ResNet18, ResNet34, ResNet50, ResNet101 | Face recognition backbone |
| **EfficientNet** | EfficientNet-B0 through B7 | Liveness detection, ear recognition |
| **Vision Transformer (ViT)** | ViT-Base, ViT-Large | Modern recognition architectures |
| **MobileNet** | MobileNetV2, MobileNetV3 | Lightweight inference |

#### Model API Design

```python
import pyflame_vision.models as models

# Load pretrained model
model = models.resnet50(pretrained=True)
model = models.efficientnet_b0(pretrained=True)

# Custom number of classes
model = models.resnet50(num_classes=512)  # Embedding dimension

# Feature extraction mode
model = models.resnet50(pretrained=True)
model.fc = pf.nn.Identity()  # Remove classification head
```

### 3. Datasets

Standard vision dataset loaders compatible with PyFlame's DataLoader.

#### Required Dataset Support

| Dataset | Purpose |
|---------|---------|
| `ImageFolder` | Load images from directory structure |
| `FaceDataset` | Face image pairs for verification training |
| Custom dataset base class | For proprietary biometric data |

### 4. Vision-Specific Operations

Operations commonly needed in vision pipelines that require Cerebras optimization.

| Operation | Description | CSL Considerations |
|-----------|-------------|-------------------|
| `roi_align` | Region of interest alignment | PE-local computation with halo exchange |
| `nms` | Non-maximum suppression | Reduction across PE mesh |
| `interpolate` | Feature map resizing | Distributed bilinear interpolation |
| `grid_sample` | Spatial transformer sampling | Coordinate-based PE routing |

## Cerebras-Specific Considerations

### PE Mesh Layout for Images

Images naturally map to Cerebras's 2D PE mesh:

```
Image Tensor [B, C, H, W]
                    │
                    ▼
┌─────────────────────────────────────┐
│  PE(0,0)  │  PE(0,1)  │  PE(0,2)   │  ← Height partitioned
├───────────┼───────────┼────────────┤     across PE rows
│  PE(1,0)  │  PE(1,1)  │  PE(1,2)   │
├───────────┼───────────┼────────────┤
│  PE(2,0)  │  PE(2,1)  │  PE(2,2)   │  ← Width partitioned
└─────────────────────────────────────┘     across PE columns
```

### Halo Exchange for Convolutions

Convolution operations require neighboring pixel data. CSL code must implement:

1. **Halo regions**: Each PE maintains border pixels from neighbors
2. **Wavelet communication**: Exchange halo data before convolution
3. **Synchronization**: Barrier before compute to ensure data arrival

### Batch Processing Strategy

- Small batches: Replicate model across PE groups
- Large batches: Distribute batch dimension across PE rows
- Single image: Full PE mesh for maximum parallelism

## Implementation Phases

### Phase 1: Core Infrastructure
- [ ] Project structure and build system (CMake + pybind11)
- [ ] Basic tensor operations for images (NCHW format)
- [ ] Resize and crop transforms with CSL code generation
- [ ] Normalize transform
- [ ] Compose transform chaining

### Phase 2: Model Architectures
- [ ] ResNet family (18, 34, 50, 101)
- [ ] EfficientNet family (B0-B4)
- [ ] Pretrained weight loading (converted from torchvision)
- [ ] Feature extraction utilities

### Phase 3: Advanced Transforms
- [ ] Data augmentation transforms (flip, rotate, color jitter)
- [ ] Gaussian blur and filtering operations
- [ ] Random transforms with deterministic seeding

### Phase 4: Specialized Operations
- [ ] ROI align for detection models
- [ ] Non-maximum suppression
- [ ] Grid sample for spatial transformers
- [ ] Interpolation modes (bilinear, bicubic, nearest)

### Phase 5: Dataset Integration
- [ ] ImageFolder dataset
- [ ] Custom dataset base classes
- [ ] Integration with PyFlame DataLoader

## API Compatibility Goals

PyFlameVision aims for high API compatibility with torchvision:

```python
# torchvision code
import torchvision.transforms as T
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# PyFlameVision equivalent
import pyflame_vision.transforms as T
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])
# Identical API, but operations build CSL-compatible graph
```

## Key Differences from torchvision

| Aspect | torchvision | PyFlameVision |
|--------|-------------|---------------|
| Execution | Eager (immediate) | Lazy (deferred to eval()) |
| Backend | CUDA kernels | CSL code generation |
| Memory | GPU VRAM | Distributed across PE SRAM |
| Parallelism | SIMT threads | 2D PE mesh wavelet routing |

## Dependencies

- **PyFlame**: Core tensor operations and IR graph
- **NumPy**: Host-side array operations
- **Pillow**: Image file I/O (host-side only)

## Success Criteria

1. All bioID vision transforms execute on Cerebras WSE
2. ResNet and EfficientNet models achieve equivalent accuracy
3. Migration from torchvision requires minimal code changes
4. Performance scales with PE mesh size

## References

- [torchvision documentation](https://pytorch.org/vision/stable/index.html)
- [PyFlame architecture documentation](../PyFlame/docs/)
- [Cerebras CSL programming guide](https://docs.cerebras.net/)

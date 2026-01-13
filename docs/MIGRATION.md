# Migration from torchvision

This guide helps you migrate existing code from torchvision to PyFlameVision.

## Overview

PyFlameVision provides a torchvision-compatible API for image transforms, making migration straightforward. Most code changes involve updating import statements and minor API adjustments.

---

## Quick Migration

### Basic Import Changes

```python
# Before (torchvision)
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# After (PyFlameVision)
from pyflame_vision.transforms import Compose, Resize, CenterCrop, Normalize
transform = Compose([
    Resize(256),
    CenterCrop(224),
    # ToTensor() not needed - PyFlameVision works with tensors
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])
])
```

### Import Mapping

| torchvision | PyFlameVision |
|-------------|---------------|
| `from torchvision import transforms` | `from pyflame_vision import transforms` |
| `transforms.Resize` | `transforms.Resize` |
| `transforms.CenterCrop` | `transforms.CenterCrop` |
| `transforms.RandomCrop` | `transforms.RandomCrop` |
| `transforms.Normalize` | `transforms.Normalize` |
| `transforms.Compose` | `transforms.Compose` |

---

## API Differences

### Resize

```python
# torchvision
transforms.Resize(256)                    # Square
transforms.Resize((256, 512))             # (height, width)
transforms.Resize(256, interpolation=InterpolationMode.BILINEAR)

# PyFlameVision - Same API
Resize(256)
Resize((256, 512))
Resize(256, interpolation="bilinear")  # String instead of enum
```

**Differences:**
- Interpolation mode is specified as string: `"nearest"`, `"bilinear"`, `"bicubic"`, `"area"`
- No PIL dependency - works directly with tensor shapes

### CenterCrop

```python
# torchvision
transforms.CenterCrop(224)
transforms.CenterCrop((200, 300))

# PyFlameVision - Same API
CenterCrop(224)
CenterCrop((200, 300))
```

**Additional in PyFlameVision:**
```python
# Get crop bounds without applying
crop = CenterCrop(224)
top, left, h, w = crop.compute_bounds(input_height=256, input_width=256)
```

### RandomCrop

```python
# torchvision
transforms.RandomCrop(224)
transforms.RandomCrop(224, padding=4)
transforms.RandomCrop(224, pad_if_needed=True)

# PyFlameVision - Same API
RandomCrop(224)
RandomCrop(224, padding=4)
RandomCrop(224, pad_if_needed=True)
```

**Additional in PyFlameVision:**
```python
# Set seed for reproducibility
crop = RandomCrop(224)
crop.set_seed(42)
```

### Normalize

```python
# torchvision
transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])

# PyFlameVision - Same API
Normalize(mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225])
```

**Additional in PyFlameVision:**
```python
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Access precomputed inverse std
inv_std = norm.inv_std()  # [1/0.229, 1/0.224, 1/0.225]
```

### Compose

```python
# torchvision
transforms.Compose([...])

# PyFlameVision - Same API
Compose([...])
```

**Additional in PyFlameVision:**
```python
pipeline = Compose([Resize(256), CenterCrop(224)])

# Check determinism
pipeline.is_deterministic()  # True

# Get output shape without running
output_shape = pipeline.get_output_shape([1, 3, 480, 640])
```

---

## Removed Transforms

These torchvision transforms have no direct equivalent in PyFlameVision Phase 1:

| Transform | Status | Alternative |
|-----------|--------|-------------|
| `ToTensor` | Not needed | Input is already tensor |
| `ToPILImage` | Not needed | Output is tensor |
| `ColorJitter` | Phase 2 | - |
| `RandomHorizontalFlip` | Phase 2 | - |
| `RandomVerticalFlip` | Phase 2 | - |
| `RandomRotation` | Phase 2 | - |
| `Pad` | Phase 2 | Use RandomCrop with padding |
| `GaussianBlur` | Phase 2 | - |

---

## Tensor Format

### torchvision

```python
# torchvision expects PIL Image or Tensor
# ToTensor converts PIL -> Tensor [C, H, W] in range [0, 1]

from PIL import Image
img = Image.open("image.jpg")  # PIL Image
tensor = transforms.ToTensor()(img)  # [3, H, W]
```

### PyFlameVision

```python
# PyFlameVision works with NCHW tensors directly
# Shape: [Batch, Channels, Height, Width]

input_shape = [1, 3, 480, 640]  # 1 image, 3 channels, 480x640

# Get output shape (no actual tensor needed for shape computation)
pipeline = Compose([Resize(256), CenterCrop(224)])
output_shape = pipeline.get_output_shape(input_shape)  # [1, 3, 224, 224]
```

---

## Common Migration Patterns

### Pattern 1: Classification Preprocessing

```python
# Before (torchvision)
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# After (PyFlameVision)
from pyflame_vision.transforms import Compose, Resize, CenterCrop, Normalize

preprocess = Compose([
    Resize(256),
    CenterCrop(224),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

### Pattern 2: Training Augmentation

```python
# Before (torchvision)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # Not yet in PyFlameVision
    transforms.RandomHorizontalFlip(),  # Not yet in PyFlameVision
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# After (PyFlameVision) - Simplified for Phase 1
train_transform = Compose([
    Resize(256),
    RandomCrop(224),  # Alternative to RandomResizedCrop
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])
```

### Pattern 3: Validation Pipeline

```python
# Before (torchvision)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# After (PyFlameVision) - Direct mapping
val_transform = Compose([
    Resize(256),
    CenterCrop(224),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])
```

---

## Feature Comparison

| Feature | torchvision | PyFlameVision |
|---------|-------------|---------------|
| Shape computation | Via forward pass | `get_output_shape()` |
| Determinism check | Manual | `is_deterministic()` |
| Cerebras WSE support | No | Yes |
| CSL code generation | No | Yes |
| Pipeline inspection | Limited | Full introspection |
| Memory estimation | No | Via ImageTensor utilities |

### PyFlameVision Advantages

```python
# Shape computation without actual tensors
pipeline = Compose([Resize(256), CenterCrop(224), Normalize(...)])
output_shape = pipeline.get_output_shape([1, 3, 480, 640])

# Determinism checking
if pipeline.is_deterministic():
    print("Safe for validation")
else:
    print("Has random transforms - use for training")

# Pipeline inspection
for i, transform in enumerate(pipeline.transforms()):
    print(f"{i}: {transform.name()}")

# Memory estimation
from pyflame_vision.core import ImageTensor
bytes_needed = ImageTensor.size_bytes(output_shape)
```

---

## Troubleshooting

### Error: "Invalid image tensor: expected 4D NCHW"

```python
# Problem: Using 3D tensor [C, H, W] instead of 4D [N, C, H, W]
resize.get_output_shape([3, 224, 224])  # Error!

# Solution: Add batch dimension
resize.get_output_shape([1, 3, 224, 224])  # OK
```

### Error: "mean/std length does not match channels"

```python
# Problem: Normalization params don't match input channels
norm = Normalize(mean=[0.5], std=[0.5])  # 1 channel
norm.get_output_shape([1, 3, 224, 224])  # Error! 3 channels

# Solution: Match channel count
norm = Normalize(
    mean=[0.5, 0.5, 0.5],  # 3 channels
    std=[0.5, 0.5, 0.5]
)
```

### Error: "Crop size larger than image"

```python
# Problem: Trying to crop more than available
crop = CenterCrop(512)
crop.get_output_shape([1, 3, 256, 256])  # Error! 512 > 256

# Solution: Resize first or use smaller crop
pipeline = Compose([
    Resize(512),      # Upscale first
    CenterCrop(512)   # Then crop
])
```

---

## Getting Help

- **Documentation**: See [API Reference](./api/TRANSFORMS.md)
- **Examples**: Check [examples/](../examples/) directory
- **Issues**: Report on GitHub

---

## See Also

- [Quick Start](./QUICKSTART.md)
- [API Reference](./api/TRANSFORMS.md)
- [Architecture](./ARCHITECTURE.md)

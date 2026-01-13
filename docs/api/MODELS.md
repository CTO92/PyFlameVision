# Models API Reference

The `pyflame_vision.models` module provides production-ready vision model architectures compatible with the torchvision API.

## Quick Reference

### Factory Functions

```python
import pyflame_vision.models as models

# ResNet Family
model = models.resnet18(num_classes=1000, pretrained=False)
model = models.resnet34(num_classes=1000, pretrained=False)
model = models.resnet50(num_classes=1000, pretrained=False)
model = models.resnet101(num_classes=1000, pretrained=False)
model = models.resnet152(num_classes=1000, pretrained=False)

# ResNeXt Variants
model = models.resnext50_32x4d(num_classes=1000, pretrained=False)
model = models.resnext101_32x8d(num_classes=1000, pretrained=False)

# Wide ResNet Variants
model = models.wide_resnet50_2(num_classes=1000, pretrained=False)
model = models.wide_resnet101_2(num_classes=1000, pretrained=False)

# EfficientNet Family
model = models.efficientnet_b0(num_classes=1000, pretrained=False)
model = models.efficientnet_b1(num_classes=1000, pretrained=False)
model = models.efficientnet_b2(num_classes=1000, pretrained=False)
model = models.efficientnet_b3(num_classes=1000, pretrained=False)
model = models.efficientnet_b4(num_classes=1000, pretrained=False)
```

---

## ResNet

Deep Residual Network for image classification.

### Factory Functions

```python
# Standard ResNet
model = models.resnet18()   # 18 layers, BasicBlock
model = models.resnet34()   # 34 layers, BasicBlock
model = models.resnet50()   # 50 layers, Bottleneck
model = models.resnet101()  # 101 layers, Bottleneck
model = models.resnet152()  # 152 layers, Bottleneck

# Custom number of classes
model = models.resnet50(num_classes=10)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_classes` | int | 1000 | Number of output classes |
| `pretrained` | bool | False | Load pretrained weights (not yet implemented) |

### ResNet Class

For custom configurations, use the ResNet class directly.

```python
from pyflame_vision.models import ResNet, ResNetBlockType

model = ResNet(
    block_type=ResNetBlockType.BOTTLENECK,
    layers=[3, 4, 6, 3],  # ResNet-50 configuration
    num_classes=1000,
    zero_init_residual=False,
    groups=1,
    width_per_group=64
)
```

**Constructor:**
```python
ResNet(
    block_type: ResNetBlockType,
    layers: List[int],
    num_classes: int = 1000,
    zero_init_residual: bool = False,
    groups: int = 1,
    width_per_group: int = 64
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_type` | ResNetBlockType | required | BASIC or BOTTLENECK |
| `layers` | List[int] | required | Blocks per stage [layer1, layer2, layer3, layer4] |
| `num_classes` | int | 1000 | Output classes (0 to disable FC) |
| `zero_init_residual` | bool | False | Zero-init last BN in residual |
| `groups` | int | 1 | Groups for 3x3 conv (>1 for ResNeXt) |
| `width_per_group` | int | 64 | Base width per group |

### ResNetBlockType

```python
from pyflame_vision.models import ResNetBlockType

ResNetBlockType.BASIC      # 2 conv layers (ResNet-18/34)
ResNetBlockType.BOTTLENECK # 3 conv layers (ResNet-50/101/152)
```

### Methods

```python
# Shape inference
output = model.get_output_shape([1, 3, 224, 224])  # [1, 1000]

# Feature extraction (intermediate outputs)
features = model.forward_features(input_spec)
# Returns: [stem_out, layer1_out, layer2_out, layer3_out, layer4_out]

# Remove FC for feature extraction
model.remove_fc()
output = model.get_output_shape([1, 3, 224, 224])  # [1, 2048, 7, 7]

# Check if FC exists
has_fc = model.has_fc()

# Get number of features before FC
num_features = model.num_features()  # 512 (Basic) or 2048 (Bottleneck)

# Access properties
block_type = model.block_type        # ResNetBlockType
layer_config = model.layer_config    # [3, 4, 6, 3]
```

### Model Configurations

| Model | Block Type | Layers | Output Features |
|-------|------------|--------|-----------------|
| ResNet-18 | Basic | [2, 2, 2, 2] | 512 |
| ResNet-34 | Basic | [3, 4, 6, 3] | 512 |
| ResNet-50 | Bottleneck | [3, 4, 6, 3] | 2048 |
| ResNet-101 | Bottleneck | [3, 4, 23, 3] | 2048 |
| ResNet-152 | Bottleneck | [3, 8, 36, 3] | 2048 |
| ResNeXt-50 (32x4d) | Bottleneck | [3, 4, 6, 3] | 2048 |
| ResNeXt-101 (32x8d) | Bottleneck | [3, 4, 23, 3] | 2048 |
| Wide ResNet-50-2 | Bottleneck | [3, 4, 6, 3] | 2048 |
| Wide ResNet-101-2 | Bottleneck | [3, 4, 23, 3] | 2048 |

### ResNet Architecture

```
Input: [N, 3, 224, 224]
    │
    ▼
┌─────────────────────────────────┐
│  Stem                           │
│  ├── Conv2d(3, 64, 7x7, s=2)   │  → [N, 64, 112, 112]
│  ├── BatchNorm2d(64)           │
│  ├── ReLU                      │
│  └── MaxPool2d(3x3, s=2)       │  → [N, 64, 56, 56]
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Layer 1: 64 channels          │  → [N, 64/256, 56, 56]
│  Layer 2: 128 channels, s=2    │  → [N, 128/512, 28, 28]
│  Layer 3: 256 channels, s=2    │  → [N, 256/1024, 14, 14]
│  Layer 4: 512 channels, s=2    │  → [N, 512/2048, 7, 7]
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Head                           │
│  ├── AdaptiveAvgPool2d(1)      │  → [N, 512/2048, 1, 1]
│  ├── Flatten                   │  → [N, 512/2048]
│  └── Linear(512/2048, classes) │  → [N, num_classes]
└─────────────────────────────────┘
```

---

## EfficientNet

Efficient convolutional network with compound scaling.

### Factory Functions

```python
model = models.efficientnet_b0()  # 224x224 input
model = models.efficientnet_b1()  # 240x240 input
model = models.efficientnet_b2()  # 260x260 input
model = models.efficientnet_b3()  # 300x300 input
model = models.efficientnet_b4()  # 380x380 input

# Custom number of classes
model = models.efficientnet_b0(num_classes=10)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_classes` | int | 1000 | Number of output classes |
| `pretrained` | bool | False | Load pretrained weights (not yet implemented) |

### Methods

```python
# Shape inference
output = model.get_output_shape([1, 3, 224, 224])  # [1, 1000]

# Feature extraction (before classifier)
features = model.forward_features(input_spec)  # [1, 1280, 7, 7]

# Remove classifier for feature extraction
model.remove_classifier()
output = model.get_output_shape([1, 3, 224, 224])  # [1, 1280, 7, 7]

# Check if classifier exists
has_clf = model.has_classifier()

# Get number of features
num_features = model.num_features()  # 1280 for B0
```

### Model Configurations

| Model | Width | Depth | Input Size | Features |
|-------|-------|-------|------------|----------|
| EfficientNet-B0 | 1.0 | 1.0 | 224 | 1280 |
| EfficientNet-B1 | 1.0 | 1.1 | 240 | 1280 |
| EfficientNet-B2 | 1.1 | 1.2 | 260 | 1408 |
| EfficientNet-B3 | 1.2 | 1.4 | 300 | 1536 |
| EfficientNet-B4 | 1.4 | 1.8 | 380 | 1792 |

### EfficientNet Architecture

```
Input: [N, 3, 224, 224] (B0)
    │
    ▼
┌─────────────────────────────────┐
│  Stem                           │
│  ├── Conv2d(3, 32, 3x3, s=2)   │  → [N, 32, 112, 112]
│  ├── BatchNorm2d(32)           │
│  └── SiLU                      │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  MBConv Blocks (7 stages)       │
│  ├── Stage 1: MBConv1, k3      │  → [N, 16, 112, 112]
│  ├── Stage 2: MBConv6, k3, s2  │  → [N, 24, 56, 56]
│  ├── Stage 3: MBConv6, k5, s2  │  → [N, 40, 28, 28]
│  ├── Stage 4: MBConv6, k3, s2  │  → [N, 80, 14, 14]
│  ├── Stage 5: MBConv6, k5      │  → [N, 112, 14, 14]
│  ├── Stage 6: MBConv6, k5, s2  │  → [N, 192, 7, 7]
│  └── Stage 7: MBConv6, k3      │  → [N, 320, 7, 7]
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Head                           │
│  ├── Conv2d(320, 1280, 1x1)    │  → [N, 1280, 7, 7]
│  ├── BatchNorm2d(1280)         │
│  ├── SiLU                      │
│  ├── AdaptiveAvgPool2d(1)      │  → [N, 1280, 1, 1]
│  ├── Flatten                   │  → [N, 1280]
│  └── Linear(1280, classes)     │  → [N, num_classes]
└─────────────────────────────────┘
```

### MBConv Block

Mobile Inverted Bottleneck Convolution:

```
Input
  │
  ├─────────────────────────────┐ (skip connection if stride=1)
  │                             │
  ▼                             │
Expand Conv (1x1) [optional]    │
  │                             │
  ▼                             │
Depthwise Conv (3x3/5x5)        │
  │                             │
  ▼                             │
Squeeze-Excitation              │
  │                             │
  ▼                             │
Project Conv (1x1)              │
  │                             │
  ▼                             │
  +─────────────────────────────┘
  │
  ▼
Output
```

---

## Common Usage Patterns

### Image Classification

```python
from pyflame_vision.transforms import Resize, CenterCrop, Normalize, Compose
import pyflame_vision.models as models

# Preprocessing
preprocess = Compose([
    Resize(256),
    CenterCrop(224),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model
model = models.resnet50(num_classes=1000)

# Verify pipeline
input_shape = [1, 3, 480, 640]
output_shape = model.get_output_shape(
    preprocess.get_output_shape(input_shape)
)
print(f"Output: {output_shape}")  # [1, 1000]
```

### Feature Extraction

```python
import pyflame_vision.models as models

# Create backbone without classifier
backbone = models.resnet50()
backbone.remove_fc()

# Get feature shapes
features = backbone.forward_features([1, 3, 224, 224])
for i, feat in enumerate(features):
    print(f"Stage {i}: {feat.shape}")

# Output:
# Stage 0: [1, 64, 56, 56]   (after stem)
# Stage 1: [1, 256, 56, 56]  (layer1)
# Stage 2: [1, 512, 28, 28]  (layer2)
# Stage 3: [1, 1024, 14, 14] (layer3)
# Stage 4: [1, 2048, 7, 7]   (layer4)
```

### Transfer Learning

```python
import pyflame_vision.models as models

# Custom classification head
model = models.resnet50(num_classes=10)  # 10 classes instead of 1000

# Or for feature extraction + custom head
backbone = models.efficientnet_b0()
backbone.remove_classifier()

num_features = backbone.num_features()  # 1280
# Add your own classifier...
```

### Model Inspection

```python
import pyflame_vision.models as models

model = models.resnet50()

# Print model structure
print(model)

# List all parameters
for param in model.parameters():
    print(f"{param.name}: {param.spec.shape}")

# Named parameters with hierarchical names
for name, param in model.named_parameters().items():
    print(f"{name}: {param.spec.shape}")
```

---

## Thread Safety

All model classes are thread-safe after construction:
- Models are immutable (weights are metadata only in lazy evaluation mode)
- Multiple threads can call `forward()` and `get_output_shape()` concurrently
- `remove_fc()` and `remove_classifier()` modify the model (call before multi-threaded use)

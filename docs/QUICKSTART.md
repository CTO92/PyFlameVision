# Quick Start Guide

> **PRE-RELEASE ALPHA SOFTWARE** - This project is currently in an early alpha stage. APIs may change without notice.

Get up and running with PyFlameVision in 5 minutes.

## Basic Usage (Python)

### Creating Transforms

```python
from pyflame_vision.transforms import Resize, CenterCrop, Normalize, Compose

# Single transform
resize = Resize(256)

# Get output shape for an input
input_shape = [1, 3, 480, 640]  # [batch, channels, height, width]
output_shape = resize.get_output_shape(input_shape)
print(f"Output: {output_shape}")  # [1, 3, 256, 256]
```

### Building Pipelines

```python
# ImageNet preprocessing pipeline
imagenet_transforms = Compose([
    Resize(256),
    CenterCrop(224),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Check the pipeline
print(imagenet_transforms)

# Get final output shape
input_shape = [8, 3, 480, 640]  # batch of 8 images
output_shape = imagenet_transforms.get_output_shape(input_shape)
print(f"Final shape: {output_shape}")  # [8, 3, 224, 224]
```

### Training vs Validation Transforms

```python
from pyflame_vision.transforms import RandomCrop

# Validation: deterministic (center crop)
val_transforms = Compose([
    Resize(256),
    CenterCrop(224),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Training: with augmentation (random crop)
train_transforms = Compose([
    Resize(256),
    RandomCrop(224),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Check determinism
print(f"Val deterministic: {val_transforms.is_deterministic()}")    # True
print(f"Train deterministic: {train_transforms.is_deterministic()}")  # False
```

## Basic Usage (C++)

### Creating Transforms

```cpp
#include <pyflame_vision/pyflame_vision.hpp>
#include <iostream>

using namespace pyflame_vision::transforms;

int main() {
    // Create a resize transform
    Resize resize(256);

    // Input shape [batch, channels, height, width]
    std::vector<int64_t> input_shape = {1, 3, 480, 640};

    // Get output shape
    auto output = resize.get_output_shape(input_shape);
    std::cout << "Output: [" << output[0] << ", " << output[1]
              << ", " << output[2] << ", " << output[3] << "]\n";

    return 0;
}
```

### Building Pipelines

```cpp
#include <pyflame_vision/pyflame_vision.hpp>

using namespace pyflame_vision::transforms;

int main() {
    // ImageNet normalization parameters
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};

    // Create pipeline
    Compose pipeline({
        std::make_shared<Resize>(256),
        std::make_shared<CenterCrop>(224),
        std::make_shared<Normalize>(mean, std)
    });

    // Print pipeline
    std::cout << pipeline.repr() << std::endl;

    // Get output shape
    std::vector<int64_t> input = {8, 3, 480, 640};
    auto output = pipeline.get_output_shape(input);

    return 0;
}
```

## Transform Reference

### Resize

Resize images to a target size.

```python
# Square resize
Resize(224)                           # 224x224

# Rectangular resize
Resize((256, 512))                    # 256 height, 512 width

# With interpolation mode
Resize(224, interpolation="nearest")  # nearest neighbor
Resize(224, interpolation="bilinear") # bilinear (default)
Resize(224, interpolation="bicubic")  # bicubic

# Disable antialiasing
Resize(224, antialias=False)
```

### CenterCrop

Crop the center of the image.

```python
# Square crop
CenterCrop(224)

# Rectangular crop
CenterCrop((200, 300))

# Get crop bounds
crop = CenterCrop(224)
top, left, h, w = crop.compute_bounds(input_height=256, input_width=256)
# Returns: (16, 16, 224, 224)
```

### RandomCrop

Randomly crop the image (for data augmentation).

```python
# Basic random crop
RandomCrop(224)

# With padding (applied before crop)
RandomCrop(224, padding=4)

# Pad if image is smaller than crop size
RandomCrop(224, pad_if_needed=True)

# Set seed for reproducibility
crop = RandomCrop(224)
crop.set_seed(42)
```

### Normalize

Normalize tensor with mean and standard deviation.

```python
# ImageNet normalization
Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Grayscale normalization
Normalize(mean=[0.5], std=[0.5])

# In-place operation
Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
```

### Compose

Chain multiple transforms into a pipeline.

```python
pipeline = Compose([
    Resize(256),
    CenterCrop(224),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Access individual transforms
first_transform = pipeline[0]
print(first_transform.name())  # "Resize"

# Get all transforms
transforms = pipeline.transforms()

# Check if empty
if not pipeline.empty():
    print(f"Pipeline has {len(pipeline)} transforms")
```

## Image Format

PyFlameVision uses **NCHW** format (Batch, Channels, Height, Width):

```
Shape: [N, C, H, W]
  N = Batch size (number of images)
  C = Channels (3 for RGB, 1 for grayscale)
  H = Height in pixels
  W = Width in pixels

Example: [8, 3, 224, 224]
  8 RGB images, each 224x224 pixels
```

## Common Patterns

### Classification Preprocessing

```python
from pyflame_vision.transforms import Resize, CenterCrop, Normalize, Compose

# Standard ImageNet preprocessing
preprocess = Compose([
    Resize(256),
    CenterCrop(224),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### Object Detection Preprocessing

```python
# Resize to fixed size (no crop)
preprocess = Compose([
    Resize((800, 1333)),  # height, width
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### Grayscale Images

```python
# Single channel normalization
preprocess = Compose([
    Resize(224),
    Normalize(mean=[0.5], std=[0.5])
])

input_shape = [1, 1, 480, 640]  # Note: 1 channel
output_shape = preprocess.get_output_shape(input_shape)
```

## Model Architectures

PyFlameVision provides production-ready model architectures for image classification.

### Creating Models

```python
import pyflame_vision.models as models

# Create ResNet-50
model = models.resnet50(num_classes=1000)

# Get output shape for an image
input_shape = [1, 3, 224, 224]  # [batch, channels, height, width]
output_shape = model.get_output_shape(input_shape)
print(f"Output: {output_shape}")  # [1, 1000]

# Create EfficientNet-B0
effnet = models.efficientnet_b0(num_classes=1000)
```

### Available Models

```python
# ResNet family
model = models.resnet18()
model = models.resnet34()
model = models.resnet50()
model = models.resnet101()
model = models.resnet152()

# ResNeXt variants
model = models.resnext50_32x4d()
model = models.resnext101_32x8d()

# Wide ResNet variants
model = models.wide_resnet50_2()
model = models.wide_resnet101_2()

# EfficientNet family
model = models.efficientnet_b0()
model = models.efficientnet_b1()
model = models.efficientnet_b2()
model = models.efficientnet_b3()
model = models.efficientnet_b4()
```

### Feature Extraction

```python
# Create model for feature extraction
backbone = models.resnet50(num_classes=1000)
backbone.remove_fc()  # Remove classification head

# Get feature shape
input_shape = [1, 3, 224, 224]
feature_shape = backbone.get_output_shape(input_shape)
print(f"Features: {feature_shape}")  # [1, 2048, 7, 7]

# Get intermediate feature shapes
features = backbone.forward_features(input_shape)
for i, feat in enumerate(features):
    print(f"Layer {i}: {feat.shape}")
```

### Custom Number of Classes

```python
# Transfer learning with custom classes
model = models.resnet50(num_classes=10)  # 10 classes instead of 1000

# Check output
output_shape = model.get_output_shape([1, 3, 224, 224])
print(f"Output: {output_shape}")  # [1, 10]
```

## Neural Network Layers

Build custom architectures using the nn module.

### Basic Layers

```python
import pyflame_vision.nn as nn

# Convolution
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
output = conv.get_output_shape([1, 3, 224, 224])  # [1, 64, 224, 224]

# Batch normalization
bn = nn.BatchNorm2d(num_features=64)
output = bn.get_output_shape([1, 64, 224, 224])  # [1, 64, 224, 224]

# Activation
relu = nn.ReLU()
silu = nn.SiLU()

# Pooling
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
output = maxpool.get_output_shape([1, 64, 224, 224])  # [1, 64, 112, 112]

avgpool = nn.AdaptiveAvgPool2d(output_size=1)
output = avgpool.get_output_shape([1, 64, 224, 224])  # [1, 64, 1, 1]

# Linear
fc = nn.Linear(in_features=512, out_features=1000)
output = fc.get_output_shape([1, 512])  # [1, 1000]
```

### Building Custom Networks

```python
import pyflame_vision.nn as nn

# Create a simple CNN using Sequential
simple_cnn = nn.Sequential([
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
])

# Check output shape
input_shape = [1, 3, 32, 32]
output_shape = simple_cnn.get_output_shape(input_shape)
print(f"Output: {output_shape}")  # [1, 64, 8, 8]
```

### Model Inspection

```python
# View model structure
model = models.resnet50()
print(model)  # Prints full model structure

# Get parameters
params = model.parameters()
for p in params:
    print(f"{p.name}: {p.spec.shape}")

# Named parameters with hierarchical names
named = model.named_parameters()
for name, param in named.items():
    print(f"{name}: {param.spec.shape}")
```

## Complete Example: Image Classification Pipeline

```python
from pyflame_vision.transforms import Resize, CenterCrop, Normalize, Compose
import pyflame_vision.models as models

# Create preprocessing pipeline
preprocess = Compose([
    Resize(256),
    CenterCrop(224),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Create model
model = models.resnet50(num_classes=1000)

# Verify shapes
input_shape = [1, 3, 480, 640]  # Raw image
preprocessed = preprocess.get_output_shape(input_shape)  # [1, 3, 224, 224]
output = model.get_output_shape(preprocessed)  # [1, 1000]

print(f"Input: {input_shape}")
print(f"After preprocessing: {preprocessed}")
print(f"Model output: {output}")
```

## Specialized Operations for Detection Models

PyFlameVision provides specialized operations commonly used in object detection and segmentation models.

### Grid Sample

Sample from input tensor at locations specified by a normalized grid.

```python
from pyflame_vision.ops import GridSample, grid_sample

# Create grid sample operation
gs = GridSample(mode="bilinear", padding_mode="zeros", align_corners=False)

# Compute output shape
input_shape = [1, 256, 64, 64]    # [N, C, H, W]
grid_shape = [1, 32, 32, 2]       # [N, H_out, W_out, 2]
output_shape = gs.get_output_shape(input_shape, grid_shape)
print(f"Output: {output_shape}")  # [1, 256, 32, 32]

# Functional API
output_shape = grid_sample(input_shape, grid_shape, mode="bilinear")
```

### ROI Align

Extract fixed-size feature maps from regions of interest.

```python
from pyflame_vision.ops import ROIAlign, ROI, roi_align

# Create ROI Align operation
roi_align_op = ROIAlign(
    output_size=(7, 7),
    spatial_scale=1.0/16,  # Feature map is 1/16 of input
    sampling_ratio=2,
    aligned=True
)

# Compute output shape for 10 ROIs
input_shape = [1, 256, 56, 56]  # Feature map shape
num_rois = 10
output_shape = roi_align_op.get_output_shape(input_shape, num_rois)
print(f"Output: {output_shape}")  # [10, 256, 7, 7]

# Functional API
output_shape = roi_align(
    input_shape=[1, 256, 56, 56],
    num_rois=10,
    output_size=(7, 7),
    spatial_scale=0.0625
)
```

### Non-Maximum Suppression (NMS)

Filter overlapping detection boxes.

```python
from pyflame_vision.ops import NMS, BatchedNMS, SoftNMS, DetectionBox

# Create detection boxes
box1 = DetectionBox(x1=10, y1=10, x2=50, y2=50, score=0.9, class_id=0)
box2 = DetectionBox(x1=15, y1=15, x2=55, y2=55, score=0.8, class_id=0)

# Compute IoU between boxes
iou = box1.iou(box2)
print(f"IoU: {iou:.3f}")  # ~0.58

# Standard NMS
nms = NMS(iou_threshold=0.5)
max_kept = nms.max_output_size(100)  # At most 100 boxes kept

# Batched NMS (class-aware)
batched_nms = BatchedNMS(iou_threshold=0.5)

# Soft NMS (score decay)
soft_nms = SoftNMS(
    sigma=0.5,
    iou_threshold=0.3,
    score_threshold=0.001,
    method="gaussian"  # or "linear"
)
```

### Complete Detection Pipeline Example

```python
from pyflame_vision.transforms import Resize, Normalize, Compose
from pyflame_vision.ops import ROIAlign, NMS
import pyflame_vision.models as models

# Create backbone model
backbone = models.resnet50(num_classes=1000)
backbone.remove_fc()  # Remove classifier for feature extraction

# Preprocessing
preprocess = Compose([
    Resize((800, 1333)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ROI Align for detection head
roi_align = ROIAlign(output_size=7, spatial_scale=1/32, sampling_ratio=2)

# NMS for post-processing
nms = NMS(iou_threshold=0.5)

# Shape inference
input_shape = [1, 3, 800, 1333]
preprocessed = preprocess.get_output_shape(input_shape)
features = backbone.get_output_shape(preprocessed)
roi_features = roi_align.get_output_shape(features, num_rois=300)

print(f"Input: {input_shape}")
print(f"Preprocessed: {preprocessed}")
print(f"Backbone features: {features}")
print(f"ROI features: {roi_features}")  # [300, 2048, 7, 7]
```

## Next Steps

- Read the full [Transforms API Reference](./api/TRANSFORMS.md)
- Read the full [Neural Network API Reference](./api/NN.md)
- Read the full [Models API Reference](./api/MODELS.md)
- Read the full [Operations API Reference](./api/OPS.md)
- Learn about [CSL Template Development](./guides/CSL_TEMPLATES.md)
- Explore the [Architecture](./ARCHITECTURE.md)

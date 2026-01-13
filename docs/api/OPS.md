# Operations (ops) Module API Reference

> **PRE-RELEASE ALPHA SOFTWARE** - This project is currently in an early alpha stage. APIs may change without notice.

The `pyflame_vision.ops` module provides specialized computer vision operations commonly used in object detection and segmentation models. These operations are compatible with `torchvision.ops`.

## Grid Sample

### GridSample

Coordinate-based spatial sampling operation. Samples from input tensor at locations specified by a normalized grid.

```python
from pyflame_vision.ops import GridSample

gs = GridSample(
    mode="bilinear",       # Interpolation mode: "bilinear" or "nearest"
    padding_mode="zeros",  # Padding mode: "zeros", "border", or "reflection"
    align_corners=False    # Whether to align corner pixels
)

# Compute output shape
input_shape = [1, 256, 64, 64]  # [N, C, H, W]
grid_shape = [1, 32, 32, 2]    # [N, H_out, W_out, 2]
output_shape = gs.get_output_shape(input_shape, grid_shape)
# Returns: [1, 256, 32, 32]
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `str` | `"bilinear"` | Interpolation mode ("bilinear" or "nearest") |
| `padding_mode` | `str` | `"zeros"` | Padding for out-of-bounds ("zeros", "border", "reflection") |
| `align_corners` | `bool` | `False` | If True, corner pixels are aligned |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `mode` | `str` | Interpolation mode |
| `padding_mode` | `str` | Padding mode |
| `align_corners` | `bool` | Corner alignment setting |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `get_output_shape(input_shape, grid_shape)` | `List[int]` | Compute output shape |
| `halo_size()` | `int` | Required halo for distributed execution |

**Grid Format:**
- Grid shape: `[N, H_out, W_out, 2]`
- Grid values: Normalized coordinates in `[-1, 1]`
- `(-1, -1)` = top-left corner, `(1, 1)` = bottom-right corner

---

## ROI Operations

### ROI

Region of Interest specification.

```python
from pyflame_vision.ops import ROI

roi = ROI(
    batch_index=0,  # Index into batch dimension
    x1=10.0,        # Left coordinate
    y1=20.0,        # Top coordinate
    x2=100.0,       # Right coordinate
    y2=150.0        # Bottom coordinate
)

print(roi.width())   # 90.0
print(roi.height())  # 130.0
print(roi.area())    # 11700.0
```

**Constructor Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `batch_index` | `int` | Index into batch dimension (must be >= 0) |
| `x1` | `float` | Left coordinate |
| `y1` | `float` | Top coordinate |
| `x2` | `float` | Right coordinate (must be >= x1) |
| `y2` | `float` | Bottom coordinate (must be >= y1) |

**Attributes:**
- `batch_index: int` - Batch index
- `x1, y1, x2, y2: float` - Box coordinates

**Methods:**
- `width() -> float` - Box width (x2 - x1)
- `height() -> float` - Box height (y2 - y1)
- `area() -> float` - Box area

### ROIAlign

Region of Interest Align operation. Extracts fixed-size feature maps from regions of interest using bilinear interpolation.

```python
from pyflame_vision.ops import ROIAlign

roi_align = ROIAlign(
    output_size=(7, 7),    # Output spatial size (H, W) or single int
    spatial_scale=0.0625,  # Scale factor (e.g., 1/16 for feature map)
    sampling_ratio=2,      # Sampling points per bin (0 = adaptive)
    aligned=True           # Use corrected alignment (recommended)
)

# Compute output shape
input_shape = [1, 256, 56, 56]  # Feature map shape
num_rois = 10
output_shape = roi_align.get_output_shape(input_shape, num_rois)
# Returns: [10, 256, 7, 7]
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_size` | `int` or `(int, int)` | - | Output spatial dimensions |
| `spatial_scale` | `float` | - | Scale from input to feature map |
| `sampling_ratio` | `int` | `0` | Sampling points per bin (0 = adaptive) |
| `aligned` | `bool` | `True` | Use pixel-accurate alignment |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `output_height` | `int` | Output height |
| `output_width` | `int` | Output width |
| `spatial_scale` | `float` | Scale factor |
| `sampling_ratio` | `int` | Sampling ratio |
| `aligned` | `bool` | Alignment mode |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `get_output_shape(input_shape, num_rois)` | `List[int]` | Compute output shape |
| `halo_size()` | `int` | Required halo (always 1) |

**Alignment Note:**
- `aligned=True` (default): Sub-pixel accurate, matches torchvision
- `aligned=False`: Legacy behavior, slight misalignment at boundaries

---

## Non-Maximum Suppression

### DetectionBox

Detection box with score and class information.

```python
from pyflame_vision.ops import DetectionBox

box1 = DetectionBox(
    x1=10, y1=10,
    x2=50, y2=50,
    score=0.9,
    class_id=0
)

box2 = DetectionBox(
    x1=15, y1=15,
    x2=55, y2=55,
    score=0.8,
    class_id=0
)

print(box1.area())      # 1600.0
print(box1.iou(box2))   # ~0.58 (Intersection over Union)
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x1` | `float` | - | Left coordinate |
| `y1` | `float` | - | Top coordinate |
| `x2` | `float` | - | Right coordinate |
| `y2` | `float` | - | Bottom coordinate |
| `score` | `float` | - | Detection confidence (must be finite) |
| `class_id` | `int` | `0` | Class label |

**Attributes:**
- `x1, y1, x2, y2: float` - Box coordinates
- `score: float` - Detection score
- `class_id: int` - Class label

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `area()` | `float` | Box area (handles negative width/height) |
| `iou(other: DetectionBox)` | `float` | Intersection over Union with another box |

### NMS

Non-Maximum Suppression for filtering overlapping detection boxes.

```python
from pyflame_vision.ops import NMS

nms = NMS(iou_threshold=0.5)

# Get maximum number of boxes that could be kept
max_kept = nms.max_output_size(num_boxes=100)  # Returns 100
```

**Constructor Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `iou_threshold` | `float` | IoU threshold in [0, 1] for suppression |

**Properties:**
- `iou_threshold: float` - IoU threshold

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `max_output_size(num_boxes)` | `int` | Maximum possible kept boxes |

**Algorithm:**
1. Sort boxes by score (descending)
2. Keep highest-scoring box
3. Remove all boxes with IoU > threshold
4. Repeat until no boxes remain

### BatchedNMS

Class-aware Non-Maximum Suppression. Performs NMS independently for each class within each batch item.

```python
from pyflame_vision.ops import BatchedNMS

batched_nms = BatchedNMS(iou_threshold=0.5)
max_kept = batched_nms.max_output_size(num_boxes=100)
```

**Constructor Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `iou_threshold` | `float` | IoU threshold in [0, 1] |

**Properties:**
- `iou_threshold: float` - IoU threshold

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `max_output_size(num_boxes)` | `int` | Maximum possible kept boxes |

### SoftNMS

Soft Non-Maximum Suppression. Instead of completely removing overlapping boxes, reduces their scores.

```python
from pyflame_vision.ops import SoftNMS

soft_nms = SoftNMS(
    sigma=0.5,               # Gaussian sigma (for gaussian method)
    iou_threshold=0.3,       # IoU threshold (for linear method)
    score_threshold=0.001,   # Minimum score to keep
    method="gaussian"        # "gaussian" or "linear"
)

max_kept = soft_nms.max_output_size(num_boxes=100)
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sigma` | `float` | `0.5` | Gaussian decay parameter (must be positive) |
| `iou_threshold` | `float` | `0.3` | Linear decay threshold |
| `score_threshold` | `float` | `0.001` | Minimum score to keep box |
| `method` | `str` | `"gaussian"` | Decay method ("gaussian" or "linear") |

**Properties:**
- `sigma: float` - Gaussian sigma
- `iou_threshold: float` - IoU threshold
- `score_threshold: float` - Score threshold

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `max_output_size(num_boxes)` | `int` | Maximum possible kept boxes |

**Score Decay Methods:**

| Method | Formula |
|--------|---------|
| Gaussian | `score *= exp(-(iou^2) / sigma)` |
| Linear | `score *= 1 - iou` if `iou > threshold` |

---

## Functional API

Stateless functions for computing output shapes.

### grid_sample

```python
from pyflame_vision.ops import grid_sample

output_shape = grid_sample(
    input_shape=[1, 256, 64, 64],
    grid_shape=[1, 32, 32, 2],
    mode="bilinear",
    padding_mode="zeros",
    align_corners=False
)
# Returns: [1, 256, 32, 32]
```

### roi_align

```python
from pyflame_vision.ops import roi_align

output_shape = roi_align(
    input_shape=[1, 256, 56, 56],
    num_rois=10,
    output_size=(7, 7),
    spatial_scale=0.0625,
    sampling_ratio=0,
    aligned=True
)
# Returns: [10, 256, 7, 7]
```

### nms

```python
from pyflame_vision.ops import nms

max_output = nms(num_boxes=100, iou_threshold=0.5)
# Returns: 100
```

### batched_nms

```python
from pyflame_vision.ops import batched_nms

max_output = batched_nms(num_boxes=100, iou_threshold=0.5)
# Returns: 100
```

---

## Security Limits

All ops validate inputs against security limits to prevent resource exhaustion:

| Limit | Value | Description |
|-------|-------|-------------|
| `MAX_ROIS` | 10,000 | Maximum number of ROIs |
| `MAX_ROI_OUTPUT_SIZE` | 256 | Maximum ROI output dimension |
| `MAX_GRID_SAMPLE_SIZE` | 4,096 | Maximum grid sample dimension |
| `MAX_NMS_BOXES` | 100,000 | Maximum boxes for NMS |
| `MAX_GRID_COORDINATE` | 1e6 | Maximum grid coordinate value |

**Validation Errors:**

```python
# These will raise ValueError:
ROIAlign(output_size=300, ...)  # Exceeds MAX_ROI_OUTPUT_SIZE
NMS(-0.1)                        # iou_threshold out of range
ROI(batch_index=-1, ...)         # Negative batch index
DetectionBox(..., score=float('inf'))  # Non-finite score
```

---

## C++ API

### Headers

```cpp
#include <pyflame_vision/ops/ops.hpp>       // All ops
#include <pyflame_vision/ops/grid_sample.hpp>
#include <pyflame_vision/ops/roi_align.hpp>
#include <pyflame_vision/ops/nms.hpp>
```

### GridSample

```cpp
using namespace pyflame_vision::ops;

GridSample gs(
    InterpolationMode::BILINEAR,
    PaddingMode::ZEROS,
    false  // align_corners
);

std::vector<int64_t> input = {1, 256, 64, 64};
std::vector<int64_t> grid = {1, 32, 32, 2};
auto output = gs.get_output_shape(input, grid);
```

### ROIAlign

```cpp
using namespace pyflame_vision::ops;

ROIAlign roi_align(
    7, 7,       // output_height, output_width
    0.0625f,    // spatial_scale
    2,          // sampling_ratio
    true        // aligned
);

std::vector<int64_t> input = {1, 256, 56, 56};
auto output = roi_align.get_output_shape(input, 10);  // 10 ROIs
```

### NMS

```cpp
using namespace pyflame_vision::ops;

NMS nms(0.5f);  // iou_threshold
int max_output = nms.max_output_size(100);

BatchedNMS batched_nms(0.5f);
SoftNMS soft_nms(0.5f, 0.3f, 0.001f, SoftNMSMethod::GAUSSIAN);

// DetectionBox with IoU
DetectionBox box1(10.0f, 10.0f, 50.0f, 50.0f, 0.9f, 0);
DetectionBox box2(15.0f, 15.0f, 55.0f, 55.0f, 0.8f, 0);
float iou = box1.iou(box2);
```

---

## Compatibility with torchvision

PyFlameVision ops are designed to be drop-in compatible with torchvision:

| PyFlameVision | torchvision |
|---------------|-------------|
| `GridSample` | `torch.nn.functional.grid_sample` |
| `ROIAlign` | `torchvision.ops.RoIAlign` |
| `NMS` | `torchvision.ops.nms` |
| `BatchedNMS` | `torchvision.ops.batched_nms` |
| `SoftNMS` | N/A (custom implementation) |

**Migration Example:**

```python
# torchvision
import torchvision.ops as ops
output = ops.roi_align(features, rois, output_size=7, spatial_scale=0.0625)

# PyFlameVision (shape inference)
from pyflame_vision.ops import roi_align
output_shape = roi_align(
    input_shape=features.shape,
    num_rois=len(rois),
    output_size=7,
    spatial_scale=0.0625
)
```

---

*See also: [Quick Start Guide](../QUICKSTART.md) | [Architecture](../ARCHITECTURE.md)*

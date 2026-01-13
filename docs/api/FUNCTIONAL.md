# Functional API Reference

The functional API provides stateless functions for transform operations. These are useful when you need direct control over transform parameters without creating transform objects.

## Python Module

```python
from pyflame_vision.transforms import functional as F

# Or import specific functions
from pyflame_vision.transforms.functional import (
    get_output_shape_resize,
    get_output_shape_center_crop,
    get_output_shape_random_crop,
    get_output_shape_normalize,
    compute_center_crop_bounds,
    validate_image_shape,
    validate_normalize_params,
)
```

---

## Shape Functions

### get_output_shape_resize

Compute output shape for resize operation.

```python
def get_output_shape_resize(
    input_shape: List[int],
    size: Union[int, Tuple[int, int]]
) -> List[int]:
    """
    Get output shape for resize operation.

    Args:
        input_shape: Input tensor shape [N, C, H, W]
        size: Target size (int for square, tuple for (height, width))

    Returns:
        Output shape [N, C, new_H, new_W]

    Raises:
        ValueError: If input_shape is not 4D NCHW
    """
```

#### Example

```python
from pyflame_vision.transforms.functional import get_output_shape_resize

# Square resize
shape = get_output_shape_resize([1, 3, 480, 640], 224)
# shape = [1, 3, 224, 224]

# Rectangular resize
shape = get_output_shape_resize([1, 3, 480, 640], (256, 512))
# shape = [1, 3, 256, 512]

# Batch is preserved
shape = get_output_shape_resize([8, 3, 480, 640], 224)
# shape = [8, 3, 224, 224]
```

---

### get_output_shape_center_crop

Compute output shape for center crop operation.

```python
def get_output_shape_center_crop(
    input_shape: List[int],
    size: Union[int, Tuple[int, int]]
) -> List[int]:
    """
    Get output shape for center crop operation.

    Args:
        input_shape: Input tensor shape [N, C, H, W]
        size: Crop size (int for square, tuple for (height, width))

    Returns:
        Output shape [N, C, crop_H, crop_W]

    Raises:
        ValueError: If input_shape is not 4D NCHW
        ValueError: If crop size is larger than input size
    """
```

#### Example

```python
from pyflame_vision.transforms.functional import get_output_shape_center_crop

shape = get_output_shape_center_crop([1, 3, 256, 256], 224)
# shape = [1, 3, 224, 224]

# Error: crop larger than input
get_output_shape_center_crop([1, 3, 256, 256], 512)
# Raises ValueError
```

---

### get_output_shape_random_crop

Compute output shape for random crop operation.

```python
def get_output_shape_random_crop(
    input_shape: List[int],
    size: Union[int, Tuple[int, int]],
    padding: int = 0
) -> List[int]:
    """
    Get output shape for random crop operation.

    Args:
        input_shape: Input tensor shape [N, C, H, W]
        size: Crop size (int for square, tuple for (height, width))
        padding: Optional padding applied before crop

    Returns:
        Output shape [N, C, crop_H, crop_W]

    Raises:
        ValueError: If input_shape is not 4D NCHW
    """
```

#### Example

```python
from pyflame_vision.transforms.functional import get_output_shape_random_crop

shape = get_output_shape_random_crop([1, 3, 256, 256], 224)
# shape = [1, 3, 224, 224]

# With padding (doesn't affect output shape)
shape = get_output_shape_random_crop([1, 3, 256, 256], 224, padding=4)
# shape = [1, 3, 224, 224]
```

---

### get_output_shape_normalize

Compute output shape for normalize operation.

```python
def get_output_shape_normalize(
    input_shape: List[int],
    mean: List[float],
    std: List[float]
) -> List[int]:
    """
    Get output shape for normalize operation (preserves shape).

    Args:
        input_shape: Input tensor shape [N, C, H, W]
        mean: Per-channel mean values
        std: Per-channel standard deviation values

    Returns:
        Output shape (same as input)

    Raises:
        ValueError: If input_shape is not 4D NCHW
        ValueError: If mean/std length doesn't match channel count
    """
```

#### Example

```python
from pyflame_vision.transforms.functional import get_output_shape_normalize

input_shape = [1, 3, 224, 224]
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

shape = get_output_shape_normalize(input_shape, mean, std)
# shape = [1, 3, 224, 224]  (unchanged)

# Error: channel mismatch
get_output_shape_normalize([1, 3, 224, 224], [0.5], [0.5])
# Raises ValueError: mean/std length (1) doesn't match channels (3)
```

---

## Utility Functions

### compute_center_crop_bounds

Compute the bounding box for a center crop operation.

```python
def compute_center_crop_bounds(
    input_height: int,
    input_width: int,
    crop_height: int,
    crop_width: int
) -> Tuple[int, int, int, int]:
    """
    Compute crop bounds for center crop.

    Args:
        input_height: Input image height
        input_width: Input image width
        crop_height: Desired crop height
        crop_width: Desired crop width

    Returns:
        Tuple of (top, left, height, width)

    Raises:
        ValueError: If crop size is larger than input size
    """
```

#### Example

```python
from pyflame_vision.transforms.functional import compute_center_crop_bounds

top, left, h, w = compute_center_crop_bounds(256, 256, 224, 224)
# top = 16, left = 16, h = 224, w = 224
# Center of 256x256 -> 224x224 crop starts at (16, 16)

# Rectangular input
top, left, h, w = compute_center_crop_bounds(480, 640, 256, 512)
# top = 112, left = 64, h = 256, w = 512
```

---

### validate_image_shape

Validate that a shape represents a valid NCHW image tensor.

```python
def validate_image_shape(shape: List[int]) -> None:
    """
    Validate that shape is a valid NCHW image tensor shape.

    Args:
        shape: Shape to validate

    Raises:
        ValueError: If shape is not 4D
        ValueError: If any dimension is <= 0
    """
```

#### Example

```python
from pyflame_vision.transforms.functional import validate_image_shape

# Valid shapes
validate_image_shape([1, 3, 224, 224])  # OK
validate_image_shape([8, 1, 480, 640])  # OK

# Invalid shapes
validate_image_shape([3, 224, 224])     # ValueError: expected 4D
validate_image_shape([1, 0, 224, 224])  # ValueError: dimensions must be positive
validate_image_shape([1, 3, -1, 224])   # ValueError: dimensions must be positive
```

---

### validate_normalize_params

Validate normalization parameters.

```python
def validate_normalize_params(
    mean: List[float],
    std: List[float],
    num_channels: Optional[int] = None
) -> None:
    """
    Validate normalization parameters.

    Args:
        mean: Per-channel mean values
        std: Per-channel standard deviation values
        num_channels: Optional expected number of channels

    Raises:
        ValueError: If mean and std have different lengths
        ValueError: If mean is empty
        ValueError: If any std value is <= 0
        ValueError: If num_channels doesn't match mean/std length
    """
```

#### Example

```python
from pyflame_vision.transforms.functional import validate_normalize_params

# Valid parameters
validate_normalize_params(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)  # OK

# With channel validation
validate_normalize_params(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    num_channels=3
)  # OK

# Invalid parameters
validate_normalize_params([0.5, 0.5], [0.5, 0.5, 0.5])  # ValueError: length mismatch
validate_normalize_params([], [])                        # ValueError: empty mean
validate_normalize_params([0.5], [0.0])                  # ValueError: std must be positive
validate_normalize_params([0.5], [0.5], num_channels=3)  # ValueError: channel mismatch
```

---

## C++ Functional API

The C++ functional API is available in the header:

```cpp
#include <pyflame_vision/transforms/functional.hpp>

namespace pyflame_vision::transforms::functional {

/// Get output shape for resize
std::vector<int64_t> resize_output_shape(
    const std::vector<int64_t>& input_shape,
    int64_t height,
    int64_t width
);

/// Get output shape for center crop
std::vector<int64_t> center_crop_output_shape(
    const std::vector<int64_t>& input_shape,
    int64_t height,
    int64_t width
);

/// Get output shape for random crop
std::vector<int64_t> random_crop_output_shape(
    const std::vector<int64_t>& input_shape,
    int64_t height,
    int64_t width
);

/// Get output shape for normalize (passthrough)
std::vector<int64_t> normalize_output_shape(
    const std::vector<int64_t>& input_shape,
    size_t num_channels
);

/// Compute center crop bounds
std::tuple<int64_t, int64_t, int64_t, int64_t> compute_center_crop_bounds(
    int64_t input_height,
    int64_t input_width,
    int64_t crop_height,
    int64_t crop_width
);

/// Validate NCHW shape
void validate_image_shape(const std::vector<int64_t>& shape);

/// Validate normalization parameters
void validate_normalize_params(
    const std::vector<float>& mean,
    const std::vector<float>& std,
    std::optional<size_t> num_channels = std::nullopt
);

}
```

### C++ Example

```cpp
#include <pyflame_vision/transforms/functional.hpp>

namespace F = pyflame_vision::transforms::functional;

// Compute shapes
auto resize_out = F::resize_output_shape({1, 3, 480, 640}, 224, 224);
auto crop_out = F::center_crop_output_shape({1, 3, 256, 256}, 224, 224);

// Get crop bounds
auto [top, left, h, w] = F::compute_center_crop_bounds(256, 256, 224, 224);

// Validation
F::validate_image_shape({1, 3, 224, 224});  // OK
F::validate_normalize_params({0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, 3);  // OK
```

---

## Use Cases

### When to Use Functional API

1. **Quick shape calculations** without creating transform objects
2. **Validation** of parameters before pipeline construction
3. **Custom transforms** that need internal utilities
4. **Testing and debugging** shape computations

### When to Use Transform Classes

1. **Standard preprocessing pipelines** (use `Compose`)
2. **Reusable transform configurations**
3. **Full API access** (repr, name, determinism checks)

---

## See Also

- [Transforms API](./TRANSFORMS.md) - Transform classes
- [Core Module](./CORE.md) - ImageTensor utilities
- [Quick Start](../QUICKSTART.md) - Getting started guide

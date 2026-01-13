# Transforms API Reference

> **PRE-RELEASE ALPHA SOFTWARE** - This project is currently in an early alpha stage. APIs may change without notice.

This document covers all transform classes in PyFlameVision.

## Overview

All transforms inherit from the base `Transform` class and implement:

- `get_output_shape(input_shape)` - Compute output tensor shape
- `name()` - Get transform name
- `is_deterministic()` - Check if transform produces consistent outputs
- `repr()` - Get string representation

---

## Transform (Base Class)

Abstract base class for all transforms.

### C++ API

```cpp
namespace pyflame_vision::transforms {

class Transform {
public:
    virtual ~Transform() = default;

    /// Get output shape for given input shape
    /// @param input_shape NCHW input tensor shape
    /// @return NCHW output tensor shape
    /// @throws std::runtime_error if input_shape is invalid
    virtual std::vector<int64_t> get_output_shape(
        const std::vector<int64_t>& input_shape
    ) const = 0;

    /// Get transform name
    virtual std::string name() const = 0;

    /// Get string representation
    virtual std::string repr() const = 0;

    /// Check if transform is deterministic
    /// @return true if same input always produces same output
    virtual bool is_deterministic() const { return true; }
};

}
```

### Python API

```python
class Transform(ABC):
    @abstractmethod
    def get_output_shape(self, input_shape: List[int]) -> List[int]:
        """Get output shape for given input shape."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Get transform name."""
        pass

    def is_deterministic(self) -> bool:
        """Check if transform is deterministic."""
        return True
```

---

## Size

Helper class for specifying dimensions.

### C++ API

```cpp
namespace pyflame_vision::transforms {

class Size {
public:
    /// Create square size
    explicit Size(int64_t size);

    /// Create rectangular size
    Size(int64_t height, int64_t width);

    /// Get height
    int64_t height() const;

    /// Get width
    int64_t width() const;

    /// Check if square
    bool is_square() const;

    /// Check if valid (positive dimensions)
    bool is_valid() const;
};

}
```

### Usage

```cpp
Size square(224);           // 224x224
Size rect(480, 640);        // 480 height, 640 width

std::cout << square.height();  // 224
std::cout << rect.is_square(); // false
```

### Python

In Python, sizes can be specified as:
- `int` - Square size (e.g., `224` → 224x224)
- `tuple` - Rectangular size (e.g., `(480, 640)` → 480 height, 640 width)

---

## Resize

Resize images to a target size using interpolation.

### C++ API

```cpp
namespace pyflame_vision::transforms {

class Resize : public Transform {
public:
    /// Create resize transform with square target size
    /// @param size Target size (height and width)
    /// @param interpolation Interpolation method (default: BILINEAR)
    /// @param antialias Apply antialiasing (default: true)
    explicit Resize(
        int64_t size,
        InterpolationMode interpolation = InterpolationMode::BILINEAR,
        bool antialias = true
    );

    /// Create resize transform with Size object
    explicit Resize(
        const Size& size,
        InterpolationMode interpolation = InterpolationMode::BILINEAR,
        bool antialias = true
    );

    // Inherited from Transform
    std::vector<int64_t> get_output_shape(
        const std::vector<int64_t>& input_shape
    ) const override;

    std::string name() const override;  // Returns "Resize"
    std::string repr() const override;
    bool is_deterministic() const override;  // Returns true

    // Accessors
    const Size& size() const;
    InterpolationMode interpolation() const;
    bool antialias() const;
};

}
```

### Python API

```python
class Resize(Transform):
    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        interpolation: str = "bilinear",
        antialias: bool = True
    ):
        """
        Create resize transform.

        Args:
            size: Target size. Int for square, tuple for (height, width).
            interpolation: One of "nearest", "bilinear", "bicubic", "area".
            antialias: Apply antialiasing filter when downsampling.
        """
```

### Examples

```python
# Python
from pyflame_vision.transforms import Resize

# Square resize
resize = Resize(224)
output = resize.get_output_shape([1, 3, 480, 640])
# output = [1, 3, 224, 224]

# Rectangular resize
resize = Resize((256, 512))
output = resize.get_output_shape([1, 3, 480, 640])
# output = [1, 3, 256, 512]

# With nearest neighbor interpolation
resize = Resize(224, interpolation="nearest")

# Disable antialiasing
resize = Resize(224, antialias=False)
```

```cpp
// C++
#include <pyflame_vision/transforms/resize.hpp>

Resize resize(224);
auto output = resize.get_output_shape({1, 3, 480, 640});
// output = {1, 3, 224, 224}

// With bicubic interpolation
Resize resize_bicubic(Size(256, 512), InterpolationMode::BICUBIC);
```

### Interpolation Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `nearest` | Nearest neighbor | Speed, pixel art |
| `bilinear` | Linear interpolation | General use (default) |
| `bicubic` | Cubic interpolation | Quality |
| `area` | Area-based | Downscaling |

---

## CenterCrop

Crop the center of the image to a specified size.

### C++ API

```cpp
namespace pyflame_vision::transforms {

class CenterCrop : public Transform {
public:
    /// Create center crop transform
    /// @param size Crop size
    /// @throws std::invalid_argument if size is invalid
    explicit CenterCrop(int64_t size);
    explicit CenterCrop(const Size& size);

    // Inherited from Transform
    std::vector<int64_t> get_output_shape(
        const std::vector<int64_t>& input_shape
    ) const override;

    std::string name() const override;  // Returns "CenterCrop"
    std::string repr() const override;
    bool is_deterministic() const override;  // Returns true

    /// Compute crop bounds
    /// @param input_height Input image height
    /// @param input_width Input image width
    /// @return Tuple of (top, left, height, width)
    std::tuple<int64_t, int64_t, int64_t, int64_t> compute_bounds(
        int64_t input_height,
        int64_t input_width
    ) const;

    // Accessors
    const Size& size() const;
};

}
```

### Python API

```python
class CenterCrop(Transform):
    def __init__(self, size: Union[int, Tuple[int, int]]):
        """
        Create center crop transform.

        Args:
            size: Crop size. Int for square, tuple for (height, width).

        Raises:
            ValueError: If size is invalid.
        """

    def compute_bounds(
        self,
        input_height: int,
        input_width: int
    ) -> Tuple[int, int, int, int]:
        """
        Compute crop bounds (top, left, height, width).
        """
```

### Examples

```python
# Python
from pyflame_vision.transforms import CenterCrop

crop = CenterCrop(224)

# Get output shape
output = crop.get_output_shape([1, 3, 256, 256])
# output = [1, 3, 224, 224]

# Get crop bounds
top, left, h, w = crop.compute_bounds(256, 256)
# top=16, left=16, h=224, w=224

# Error: crop larger than input
crop = CenterCrop(512)
crop.get_output_shape([1, 3, 256, 256])  # Raises ValueError
```

```cpp
// C++
CenterCrop crop(224);
auto [top, left, h, w] = crop.compute_bounds(256, 256);
// top=16, left=16, h=224, w=224
```

---

## RandomCrop

Randomly crop the image to a specified size.

### C++ API

```cpp
namespace pyflame_vision::transforms {

class RandomCrop : public Transform {
public:
    /// Create random crop transform
    /// @param size Crop size
    /// @param padding Padding to apply before cropping
    /// @param pad_if_needed Pad if image is smaller than crop size
    /// @param fill Fill value for padding
    explicit RandomCrop(
        int64_t size,
        int padding = 0,
        bool pad_if_needed = false,
        float fill = 0.0f
    );

    explicit RandomCrop(
        const Size& size,
        int padding = 0,
        bool pad_if_needed = false,
        float fill = 0.0f
    );

    // Inherited from Transform
    std::vector<int64_t> get_output_shape(
        const std::vector<int64_t>& input_shape
    ) const override;

    std::string name() const override;  // Returns "RandomCrop"
    std::string repr() const override;
    bool is_deterministic() const override;  // Returns FALSE

    /// Set random seed for reproducibility
    void set_seed(int64_t seed);

    // Accessors
    const Size& size() const;
    int padding() const;
    bool pad_if_needed() const;
    float fill() const;
};

}
```

### Python API

```python
class RandomCrop(Transform):
    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        padding: int = 0,
        pad_if_needed: bool = False,
        fill: float = 0.0
    ):
        """
        Create random crop transform.

        Args:
            size: Crop size.
            padding: Padding applied before cropping.
            pad_if_needed: If True, pad image if smaller than crop size.
            fill: Fill value for padding.
        """

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""

    def is_deterministic(self) -> bool:
        """Returns False (random transform)."""
        return False
```

### Examples

```python
# Python
from pyflame_vision.transforms import RandomCrop

# Basic random crop
crop = RandomCrop(224)
assert crop.is_deterministic() == False

# With padding (common for training)
crop = RandomCrop(224, padding=4)

# Reproducible crops
crop = RandomCrop(224)
crop.set_seed(42)
```

---

## Normalize

Normalize tensor with mean and standard deviation.

Formula: `output = (input - mean) / std`

### C++ API

```cpp
namespace pyflame_vision::transforms {

class Normalize : public Transform {
public:
    /// Create normalize transform
    /// @param mean Per-channel mean values
    /// @param std Per-channel standard deviation values
    /// @param inplace Perform operation in-place
    /// @throws std::invalid_argument if mean/std lengths differ or std <= 0
    Normalize(
        const std::vector<float>& mean,
        const std::vector<float>& std,
        bool inplace = false
    );

    // Inherited from Transform
    std::vector<int64_t> get_output_shape(
        const std::vector<int64_t>& input_shape
    ) const override;

    std::string name() const override;  // Returns "Normalize"
    std::string repr() const override;
    bool is_deterministic() const override;  // Returns true

    // Accessors
    const std::vector<float>& mean() const;
    const std::vector<float>& std() const;
    const std::vector<float>& inv_std() const;  // Precomputed 1/std
    bool inplace() const;
};

}
```

### Python API

```python
class Normalize(Transform):
    def __init__(
        self,
        mean: List[float],
        std: List[float],
        inplace: bool = False
    ):
        """
        Create normalize transform.

        Args:
            mean: Per-channel mean values.
            std: Per-channel standard deviation values.
            inplace: Perform operation in-place (default: False).

        Raises:
            ValueError: If mean/std lengths differ or std values <= 0.
        """

    def mean(self) -> List[float]:
        """Get mean values."""

    def std(self) -> List[float]:
        """Get std values."""

    def inv_std(self) -> List[float]:
        """Get precomputed 1/std values."""
```

### Examples

```python
# Python
from pyflame_vision.transforms import Normalize

# ImageNet normalization
norm = Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Check values
print(norm.mean())     # [0.485, 0.456, 0.406]
print(norm.inv_std())  # [4.367..., 4.464..., 4.444...]

# Shape is preserved
output = norm.get_output_shape([1, 3, 224, 224])
# output = [1, 3, 224, 224]

# Grayscale
norm_gray = Normalize(mean=[0.5], std=[0.5])
norm_gray.get_output_shape([1, 1, 224, 224])  # OK
norm_gray.get_output_shape([1, 3, 224, 224])  # ValueError: channel mismatch
```

### Common Normalization Parameters

| Dataset | Mean | Std |
|---------|------|-----|
| ImageNet | [0.485, 0.456, 0.406] | [0.229, 0.224, 0.225] |
| CIFAR-10 | [0.4914, 0.4822, 0.4465] | [0.2470, 0.2435, 0.2616] |
| Grayscale | [0.5] | [0.5] |

---

## Compose

Chain multiple transforms into a pipeline.

### C++ API

```cpp
namespace pyflame_vision::transforms {

class Compose : public Transform {
public:
    /// Create compose transform
    /// @param transforms List of transforms to chain
    /// @throws std::invalid_argument if any transform is null
    explicit Compose(std::vector<std::shared_ptr<Transform>> transforms);

    // Inherited from Transform
    std::vector<int64_t> get_output_shape(
        const std::vector<int64_t>& input_shape
    ) const override;

    std::string name() const override;  // Returns "Compose"
    std::string repr() const override;

    /// Returns true only if ALL transforms are deterministic
    bool is_deterministic() const override;

    // Container-like interface
    size_t size() const;
    bool empty() const;
    std::shared_ptr<Transform> operator[](size_t index) const;

    // Get all transforms
    const std::vector<std::shared_ptr<Transform>>& transforms() const;
};

}
```

### Python API

```python
class Compose(Transform):
    def __init__(self, transforms: List[Transform]):
        """
        Create composed transform pipeline.

        Args:
            transforms: List of transforms to apply in order.

        Raises:
            ValueError: If any transform is None.
        """

    def __len__(self) -> int:
        """Get number of transforms."""

    def __getitem__(self, index: int) -> Transform:
        """Get transform at index."""

    def transforms(self) -> List[Transform]:
        """Get all transforms."""

    def empty(self) -> bool:
        """Check if pipeline is empty."""
```

### Examples

```python
# Python
from pyflame_vision.transforms import Compose, Resize, CenterCrop, Normalize

# Create pipeline
pipeline = Compose([
    Resize(256),
    CenterCrop(224),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Get output shape
output = pipeline.get_output_shape([1, 3, 480, 640])
# Resize: [1, 3, 480, 640] -> [1, 3, 256, 256]
# CenterCrop: [1, 3, 256, 256] -> [1, 3, 224, 224]
# Normalize: [1, 3, 224, 224] -> [1, 3, 224, 224]
# output = [1, 3, 224, 224]

# Access individual transforms
print(len(pipeline))           # 3
print(pipeline[0].name())      # "Resize"
print(pipeline.is_deterministic())  # True

# Print pipeline
print(pipeline)
# Compose([
#     Resize(size=256, interpolation=bilinear),
#     CenterCrop(size=224),
#     Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
# ])
```

```cpp
// C++
Compose pipeline({
    std::make_shared<Resize>(256),
    std::make_shared<CenterCrop>(224),
    std::make_shared<Normalize>(
        std::vector<float>{0.485f, 0.456f, 0.406f},
        std::vector<float>{0.229f, 0.224f, 0.225f}
    )
});

auto output = pipeline.get_output_shape({1, 3, 480, 640});
// output = {1, 3, 224, 224}
```

---

## Error Handling

All transforms validate inputs and throw descriptive exceptions:

| Error | Cause |
|-------|-------|
| `ValueError` / `std::runtime_error` | Invalid input shape (not 4D NCHW) |
| `ValueError` / `std::runtime_error` | Crop size larger than input |
| `ValueError` / `std::invalid_argument` | Invalid parameters (e.g., std <= 0) |
| `ValueError` / `std::invalid_argument` | Channel count mismatch (Normalize) |

```python
# Python error handling
from pyflame_vision.transforms import CenterCrop

crop = CenterCrop(512)
try:
    crop.get_output_shape([1, 3, 256, 256])  # 512 > 256
except ValueError as e:
    print(e)  # "Crop size (512, 512) larger than image size (256, 256)"
```

---

## Phase 3: Data Augmentation Transforms

The following transforms are random (non-deterministic) data augmentation operations
commonly used in training pipelines. All inherit from `RandomTransform` base class
which provides thread-safe random number generation.

### RandomTransform (Base Class)

Base class for all random transforms with thread-safe RNG.

```cpp
namespace pyflame_vision::transforms {

class RandomTransform : public Transform {
public:
    /// Set seed for reproducible results
    void set_seed(uint64_t seed);

    /// Get current seed (if explicitly set)
    std::optional<uint64_t> seed() const;

    /// Always returns false (random transforms are non-deterministic)
    bool is_deterministic() const override { return false; }

protected:
    /// Thread-safe random utilities
    float random_uniform() const;              // [0, 1)
    float random_uniform(float low, float high) const;
    bool random_bool(float probability) const;
    int random_int(int low, int high) const;
};

}
```

---

## RandomHorizontalFlip

Randomly flip image horizontally with given probability.

### C++ API

```cpp
namespace pyflame_vision::transforms {

class RandomHorizontalFlip : public RandomTransform {
public:
    /// Create horizontal flip transform
    /// @param p Probability of flipping (0.0 to 1.0, default: 0.5)
    /// @throws ValidationError if p is not in [0, 1]
    explicit RandomHorizontalFlip(float p = 0.5f);

    // Inherited from Transform
    std::vector<int64_t> get_output_shape(
        const std::vector<int64_t>& input_shape
    ) const override;

    std::string name() const override;  // Returns "RandomHorizontalFlip"
    bool is_deterministic() const override;  // Returns false

    // Accessors
    float probability() const;
    bool was_flipped() const;  // Check if last call resulted in flip
};

}
```

### Python API

```python
class RandomHorizontalFlip(Transform):
    def __init__(self, p: float = 0.5):
        """
        Randomly flip image horizontally.

        Args:
            p: Probability of flipping (default: 0.5).

        Raises:
            ValueError: If p is not in [0, 1].
        """

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""

    def was_flipped(self) -> bool:
        """Check if last call resulted in a flip."""
```

### Examples

```python
from pyflame_vision.transforms import RandomHorizontalFlip, Compose

# Basic usage
flip = RandomHorizontalFlip(p=0.5)
assert flip.is_deterministic() == False

# Shape is preserved
output = flip.get_output_shape([1, 3, 224, 224])
# output = [1, 3, 224, 224]

# Reproducible results
flip.set_seed(42)
flip.get_output_shape([1, 3, 224, 224])
print(flip.was_flipped())  # True or False

# Always flip
always_flip = RandomHorizontalFlip(p=1.0)
always_flip.get_output_shape([1, 3, 224, 224])
assert always_flip.was_flipped() == True

# Never flip
never_flip = RandomHorizontalFlip(p=0.0)
never_flip.get_output_shape([1, 3, 224, 224])
assert never_flip.was_flipped() == False
```

---

## RandomVerticalFlip

Randomly flip image vertically with given probability.

### C++ API

```cpp
namespace pyflame_vision::transforms {

class RandomVerticalFlip : public RandomTransform {
public:
    /// Create vertical flip transform
    /// @param p Probability of flipping (0.0 to 1.0, default: 0.5)
    /// @throws ValidationError if p is not in [0, 1]
    explicit RandomVerticalFlip(float p = 0.5f);

    // Inherited from Transform
    std::vector<int64_t> get_output_shape(
        const std::vector<int64_t>& input_shape
    ) const override;

    std::string name() const override;  // Returns "RandomVerticalFlip"
    bool is_deterministic() const override;  // Returns false

    // Accessors
    float probability() const;
    bool was_flipped() const;
};

}
```

### Python API

```python
class RandomVerticalFlip(Transform):
    def __init__(self, p: float = 0.5):
        """
        Randomly flip image vertically.

        Args:
            p: Probability of flipping (default: 0.5).
        """

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""

    def was_flipped(self) -> bool:
        """Check if last call resulted in a flip."""
```

### Examples

```python
from pyflame_vision.transforms import RandomVerticalFlip

flip = RandomVerticalFlip(p=0.5)
flip.set_seed(42)

# Run multiple times
for _ in range(10):
    flip.get_output_shape([1, 3, 224, 224])
    print(f"Flipped: {flip.was_flipped()}")
```

---

## RandomRotation

Randomly rotate image by angle within specified range.

### C++ API

```cpp
namespace pyflame_vision::transforms {

/// Fill mode for areas outside the original image
enum class RotationFillMode : uint8_t {
    CONSTANT = 0,   // Fill with constant value
    REFLECT = 1,    // Reflect at boundary
    REPLICATE = 2,  // Replicate edge pixels
};

class RandomRotation : public RandomTransform {
public:
    /// Create rotation with symmetric range [-degrees, +degrees]
    /// @param degrees Maximum rotation angle
    /// @param interpolation Interpolation mode (default: BILINEAR)
    /// @param expand If true, expand output to fit rotated image
    /// @param center Center of rotation (default: image center)
    /// @param fill Fill values for areas outside the image
    explicit RandomRotation(
        float degrees,
        InterpolationMode interpolation = InterpolationMode::BILINEAR,
        bool expand = false,
        std::optional<std::pair<float, float>> center = std::nullopt,
        std::vector<float> fill = {0.0f}
    );

    /// Create rotation with asymmetric range [degrees_min, degrees_max]
    RandomRotation(
        float degrees_min,
        float degrees_max,
        InterpolationMode interpolation = InterpolationMode::BILINEAR,
        bool expand = false,
        std::optional<std::pair<float, float>> center = std::nullopt,
        std::vector<float> fill = {0.0f}
    );

    // Inherited from Transform
    std::vector<int64_t> get_output_shape(
        const std::vector<int64_t>& input_shape
    ) const override;

    std::string name() const override;  // Returns "RandomRotation"

    // Accessors
    std::pair<float, float> degrees() const;  // Get (min, max) range
    float last_angle() const;                 // Get last applied angle
    InterpolationMode interpolation() const;
    bool expand() const;
    const std::vector<float>& fill() const;
    int halo_size() const;  // For distributed execution
};

}
```

### Python API

```python
class RandomRotation(Transform):
    def __init__(
        self,
        degrees: Union[float, Tuple[float, float]],
        interpolation: str = "bilinear",
        expand: bool = False,
        center: Optional[Tuple[float, float]] = None,
        fill: Union[float, List[float]] = 0.0
    ):
        """
        Randomly rotate image.

        Args:
            degrees: Rotation range. Float for [-d, d], tuple for (min, max).
            interpolation: One of "nearest", "bilinear".
            expand: If True, expand output to contain full rotated image.
            center: Center of rotation. Default is image center.
            fill: Fill value(s) for areas outside the image.

        Raises:
            ValueError: If degrees exceed 360 or min > max.
        """

    def degrees(self) -> Tuple[float, float]:
        """Get rotation angle range (min, max)."""

    def last_angle(self) -> float:
        """Get last applied rotation angle."""
```

### Examples

```python
from pyflame_vision.transforms import RandomRotation

# Symmetric range: rotate between -30 and +30 degrees
rot = RandomRotation(30.0)
rot.set_seed(42)

# Check output shape (unchanged without expand)
output = rot.get_output_shape([1, 3, 224, 224])
# output = [1, 3, 224, 224]

print(f"Rotated by: {rot.last_angle()} degrees")

# Asymmetric range
rot2 = RandomRotation(degrees=(-15.0, 45.0))

# With expand (output larger to fit rotated image)
rot3 = RandomRotation(45.0, expand=True)
output = rot3.get_output_shape([1, 3, 224, 224])
# output = [1, 3, ~317, ~317]  (expanded for 45° rotation)

# Custom fill color (gray)
rot4 = RandomRotation(30.0, fill=[0.5, 0.5, 0.5])
```

---

## ColorJitter

Randomly adjust brightness, contrast, saturation, and hue.

### C++ API

```cpp
namespace pyflame_vision::transforms {

class ColorJitter : public RandomTransform {
public:
    /// Create color jitter with single values
    /// @param brightness Creates range [max(0, 1-b), 1+b]
    /// @param contrast Creates range [max(0, 1-c), 1+c]
    /// @param saturation Creates range [max(0, 1-s), 1+s]
    /// @param hue Creates range [-h, h], must be in [0, 0.5]
    ColorJitter(
        float brightness = 0.0f,
        float contrast = 0.0f,
        float saturation = 0.0f,
        float hue = 0.0f
    );

    /// Create color jitter with explicit ranges
    ColorJitter(
        std::pair<float, float> brightness,
        std::pair<float, float> contrast,
        std::pair<float, float> saturation,
        std::pair<float, float> hue
    );

    // Inherited from Transform
    std::vector<int64_t> get_output_shape(
        const std::vector<int64_t>& input_shape
    ) const override;

    std::string name() const override;  // Returns "ColorJitter"

    // Get parameter ranges
    std::pair<float, float> brightness() const;
    std::pair<float, float> contrast() const;
    std::pair<float, float> saturation() const;
    std::pair<float, float> hue() const;

    // Get last applied values
    float last_brightness_factor() const;
    float last_contrast_factor() const;
    float last_saturation_factor() const;
    float last_hue_factor() const;

    // Get last transform order (0=B, 1=C, 2=S, 3=H)
    std::array<int, 4> last_order() const;
};

}
```

### Python API

```python
class ColorJitter(Transform):
    def __init__(
        self,
        brightness: Union[float, Tuple[float, float]] = 0.0,
        contrast: Union[float, Tuple[float, float]] = 0.0,
        saturation: Union[float, Tuple[float, float]] = 0.0,
        hue: Union[float, Tuple[float, float]] = 0.0
    ):
        """
        Randomly adjust color properties.

        For single values:
            brightness=0.2 → factor in [0.8, 1.2]
            contrast=0.3 → factor in [0.7, 1.3]
            saturation=0.4 → factor in [0.6, 1.4]
            hue=0.1 → shift in [-0.1, 0.1]

        Args:
            brightness: Brightness jitter amount or (min, max) range.
            contrast: Contrast jitter amount or (min, max) range.
            saturation: Saturation jitter amount or (min, max) range.
            hue: Hue jitter amount (max 0.5) or (min, max) range.

        Raises:
            ValueError: If any factor is negative or hue > 0.5.
        """

    def brightness(self) -> Tuple[float, float]:
        """Get brightness factor range."""

    def contrast(self) -> Tuple[float, float]:
        """Get contrast factor range."""

    def saturation(self) -> Tuple[float, float]:
        """Get saturation factor range."""

    def hue(self) -> Tuple[float, float]:
        """Get hue shift range."""

    def last_brightness_factor(self) -> float:
        """Get last applied brightness factor."""

    def last_contrast_factor(self) -> float:
        """Get last applied contrast factor."""

    def last_saturation_factor(self) -> float:
        """Get last applied saturation factor."""

    def last_hue_factor(self) -> float:
        """Get last applied hue shift."""

    def last_order(self) -> List[int]:
        """Get last transform application order."""
```

### Examples

```python
from pyflame_vision.transforms import ColorJitter

# Basic usage
jitter = ColorJitter(
    brightness=0.2,   # ±20% brightness
    contrast=0.2,     # ±20% contrast
    saturation=0.2,   # ±20% saturation
    hue=0.1           # ±10% hue (0.1 of color wheel)
)

jitter.set_seed(42)
output = jitter.get_output_shape([1, 3, 224, 224])
# output = [1, 3, 224, 224] (shape unchanged)

print(f"Brightness: {jitter.last_brightness_factor()}")  # e.g., 1.15
print(f"Contrast: {jitter.last_contrast_factor()}")      # e.g., 0.92
print(f"Saturation: {jitter.last_saturation_factor()}")  # e.g., 1.08
print(f"Hue: {jitter.last_hue_factor()}")                # e.g., -0.05
print(f"Order: {jitter.last_order()}")                   # e.g., [2, 0, 3, 1]

# With explicit ranges
jitter2 = ColorJitter(
    brightness=(0.8, 1.2),
    contrast=(0.9, 1.1),
    saturation=(0.7, 1.3),
    hue=(-0.05, 0.05)
)
```

### Color Jitter Order

ColorJitter randomly permutes the order of operations each time. The order
is indicated by `last_order()` which returns indices:
- 0 = Brightness
- 1 = Contrast
- 2 = Saturation
- 3 = Hue

For example, `[2, 0, 3, 1]` means: Saturation → Brightness → Hue → Contrast.

---

## GaussianBlur

Apply Gaussian blur with configurable kernel size and sigma.

### C++ API

```cpp
namespace pyflame_vision::transforms {

class GaussianBlur : public RandomTransform {
public:
    /// Create Gaussian blur with fixed kernel size
    /// @param kernel_size Blur kernel size (must be positive odd integer)
    /// @param sigma Sigma range [min, max] (default: [0.1, 2.0])
    explicit GaussianBlur(
        int kernel_size,
        std::pair<float, float> sigma = {0.1f, 2.0f}
    );

    /// Create Gaussian blur with kernel size range
    /// @param kernel_size_min Minimum kernel size (must be positive odd)
    /// @param kernel_size_max Maximum kernel size (must be positive odd)
    /// @param sigma Sigma range [min, max]
    GaussianBlur(
        int kernel_size_min,
        int kernel_size_max,
        std::pair<float, float> sigma = {0.1f, 2.0f}
    );

    // Inherited from Transform
    std::vector<int64_t> get_output_shape(
        const std::vector<int64_t>& input_shape
    ) const override;

    std::string name() const override;  // Returns "GaussianBlur"

    // Accessors
    std::pair<int, int> kernel_size() const;    // Get (min, max) kernel sizes
    std::pair<float, float> sigma() const;       // Get sigma range
    int last_kernel_size() const;                // Get last kernel size used
    float last_sigma() const;                    // Get last sigma used
    int halo_size() const;                       // For distributed execution
    std::vector<float> get_kernel_weights() const; // Get 1D kernel weights
};

}
```

### Python API

```python
class GaussianBlur(Transform):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        sigma: Union[float, Tuple[float, float]] = (0.1, 2.0)
    ):
        """
        Apply Gaussian blur.

        Args:
            kernel_size: Kernel size (must be odd). Int or (min, max) tuple.
            sigma: Sigma value(s). Float or (min, max) tuple.

        Raises:
            ValueError: If kernel_size is not positive odd or exceeds 31.
            ValueError: If sigma is not positive or exceeds 10.0.
        """

    def kernel_size(self) -> Tuple[int, int]:
        """Get kernel size range (min, max)."""

    def sigma(self) -> Tuple[float, float]:
        """Get sigma range (min, max)."""

    def last_kernel_size(self) -> int:
        """Get last applied kernel size."""

    def last_sigma(self) -> float:
        """Get last applied sigma."""

    def get_kernel_weights(self) -> List[float]:
        """Get 1D Gaussian kernel weights (normalized to sum to 1)."""

    def halo_size(self) -> int:
        """Get halo size for distributed execution."""
```

### Examples

```python
from pyflame_vision.transforms import GaussianBlur

# Fixed kernel size
blur = GaussianBlur(kernel_size=5)
blur.set_seed(42)

output = blur.get_output_shape([1, 3, 224, 224])
# output = [1, 3, 224, 224] (shape unchanged)

print(f"Kernel size: {blur.last_kernel_size()}")  # 5
print(f"Sigma: {blur.last_sigma()}")              # e.g., 1.23

# Get kernel weights (for debugging)
weights = blur.get_kernel_weights()
print(f"Weights: {weights}")  # e.g., [0.06, 0.24, 0.40, 0.24, 0.06]

# Variable kernel size
blur2 = GaussianBlur(kernel_size=(3, 7))
for _ in range(10):
    blur2.get_output_shape([1, 3, 224, 224])
    print(f"Kernel: {blur2.last_kernel_size()}")  # 3, 5, or 7

# Custom sigma range
blur3 = GaussianBlur(kernel_size=5, sigma=(0.5, 1.5))
```

### Security Limits

| Parameter | Limit | Reason |
|-----------|-------|--------|
| `kernel_size` | max 31 | Prevent excessive computation |
| `kernel_size` | must be odd | Ensures symmetric kernel |
| `sigma` | max 10.0 | Practical limit for blur effect |

---

## Data Augmentation Pipeline Example

Combining Phase 3 transforms for typical training augmentation:

```python
from pyflame_vision.transforms import (
    Compose, Resize, RandomCrop, RandomHorizontalFlip,
    RandomRotation, ColorJitter, GaussianBlur, Normalize
)

# ImageNet training augmentation
train_transform = Compose([
    Resize(256),
    RandomCrop(224),
    RandomHorizontalFlip(p=0.5),
    RandomRotation(15.0),
    ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0)),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Check pipeline
print(f"Deterministic: {train_transform.is_deterministic()}")  # False
output = train_transform.get_output_shape([1, 3, 480, 640])
print(f"Output shape: {output}")  # [1, 3, 224, 224]

# Reproducible training
for transform in train_transform.transforms():
    if hasattr(transform, 'set_seed'):
        transform.set_seed(42)
```

---

## See Also

- [Core Module](./CORE.md) - ImageTensor utilities
- [Functional API](./FUNCTIONAL.md) - Stateless functions
- [Quick Start](../QUICKSTART.md) - Usage examples
- [Phase 3 Implementation Guide](../PHASE3_IMPLEMENTATION_GUIDE.md) - Technical details

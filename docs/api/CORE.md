# Core Module API Reference

The core module provides fundamental types and utilities for image tensor operations.

## ImageFormat

Constants defining the NCHW tensor format.

### C++ API

```cpp
namespace pyflame_vision::core {

struct ImageFormat {
    static constexpr int BATCH_DIM = 0;
    static constexpr int CHANNEL_DIM = 1;
    static constexpr int HEIGHT_DIM = 2;
    static constexpr int WIDTH_DIM = 3;
    static constexpr int NUM_DIMS = 4;
};

}
```

### Usage

```cpp
#include <pyflame_vision/core/image_tensor.hpp>

std::vector<int64_t> shape = {1, 3, 224, 224};
int64_t batch = shape[ImageFormat::BATCH_DIM];      // 1
int64_t channels = shape[ImageFormat::CHANNEL_DIM]; // 3
int64_t height = shape[ImageFormat::HEIGHT_DIM];    // 224
int64_t width = shape[ImageFormat::WIDTH_DIM];      // 224
```

---

## ColorSpace

Enumeration of supported color spaces.

### C++ API

```cpp
namespace pyflame_vision::core {

enum class ColorSpace : uint8_t {
    RGB = 0,   // Red, Green, Blue
    BGR = 1,   // Blue, Green, Red (OpenCV default)
    GRAY = 2,  // Grayscale (single channel)
    HSV = 3,   // Hue, Saturation, Value
    LAB = 4,   // CIE L*a*b*
};

// Get color space name as string
std::string colorspace_name(ColorSpace cs);

}
```

### Usage

```cpp
ColorSpace cs = ColorSpace::RGB;
std::cout << colorspace_name(cs);  // "RGB"
```

---

## InterpolationMode

Enumeration of interpolation methods for resize operations.

### C++ API

```cpp
namespace pyflame_vision::core {

enum class InterpolationMode {
    NEAREST,   // Nearest neighbor interpolation
    BILINEAR,  // Bilinear interpolation (default)
    BICUBIC,   // Bicubic interpolation
    AREA,      // Area-based resampling (for downscaling)
};

// Get interpolation mode name
std::string interpolation_name(InterpolationMode mode);

// Parse string to interpolation mode
InterpolationMode parse_interpolation(const std::string& name);

// Check if mode is valid for upscaling
bool is_valid_for_upscale(InterpolationMode mode);

// Check if mode is valid for downscaling
bool is_valid_for_downscale(InterpolationMode mode);

}
```

### Usage

```cpp
#include <pyflame_vision/core/interpolation.hpp>

// Create from enum
InterpolationMode mode = InterpolationMode::BILINEAR;
std::cout << interpolation_name(mode);  // "bilinear"

// Parse from string
InterpolationMode parsed = parse_interpolation("bicubic");

// Validation
bool ok = is_valid_for_upscale(InterpolationMode::AREA);  // false
```

### Python API

```python
# Interpolation is specified as a string
from pyflame_vision.transforms import Resize

Resize(224, interpolation="nearest")
Resize(224, interpolation="bilinear")  # default
Resize(224, interpolation="bicubic")
Resize(224, interpolation="area")
```

---

## ImageTensor

Utility class for NCHW image tensor operations.

### C++ API

```cpp
namespace pyflame_vision::core {

class ImageTensor {
public:
    // ===== Shape Validation =====

    /// Check if shape is valid NCHW format
    static bool is_valid_shape(const std::vector<int64_t>& shape);

    /// Validate shape, throw if invalid
    static void validate_shape(const std::vector<int64_t>& shape);

    // ===== Dimension Accessors =====

    /// Get dimensions as tuple (batch, channels, height, width)
    static std::tuple<int64_t, int64_t, int64_t, int64_t>
    get_dimensions(const std::vector<int64_t>& shape);

    /// Get batch size (dimension 0)
    static int64_t batch_size(const std::vector<int64_t>& shape);

    /// Get number of channels (dimension 1)
    static int64_t num_channels(const std::vector<int64_t>& shape);

    /// Get height (dimension 2)
    static int64_t height(const std::vector<int64_t>& shape);

    /// Get width (dimension 3)
    static int64_t width(const std::vector<int64_t>& shape);

    // ===== Output Shape Computation =====

    /// Compute output shape for resize operation
    static std::vector<int64_t> resize_output_shape(
        const std::vector<int64_t>& input_shape,
        int64_t target_height,
        int64_t target_width
    );

    /// Compute output shape for crop operation
    static std::vector<int64_t> crop_output_shape(
        const std::vector<int64_t>& input_shape,
        int64_t crop_height,
        int64_t crop_width
    );

    // ===== Layout Planning =====

    /// Compute optimal mesh layout for given image size
    static pyflame::MeshLayout optimal_layout(
        int64_t height,
        int64_t width,
        size_t element_size = 4
    );

    /// Compute tile shape for a specific PE in the mesh
    static std::tuple<int64_t, int64_t> tile_shape(
        int64_t height,
        int64_t width,
        const pyflame::MeshLayout& layout,
        int pe_row,
        int pe_col
    );

    // ===== Halo Operations =====

    /// Check if operation needs halo exchange
    static bool needs_halo(int kernel_size);

    /// Compute halo size for a kernel
    static int halo_size(int kernel_size);

    // ===== Memory Calculations =====

    /// Calculate total number of elements
    static int64_t numel(const std::vector<int64_t>& shape);

    /// Calculate memory size in bytes
    static size_t size_bytes(
        const std::vector<int64_t>& shape,
        size_t element_size = 4
    );
};

}
```

### Usage Examples

```cpp
#include <pyflame_vision/core/image_tensor.hpp>

using namespace pyflame_vision::core;

// Validate shape
std::vector<int64_t> shape = {1, 3, 224, 224};
if (ImageTensor::is_valid_shape(shape)) {
    // Shape is valid NCHW
}

// Get dimensions
auto [batch, channels, height, width] = ImageTensor::get_dimensions(shape);

// Compute output shape for resize
auto output = ImageTensor::resize_output_shape(shape, 256, 256);
// output = {1, 3, 256, 256}

// Memory estimation
size_t bytes = ImageTensor::size_bytes(shape);  // 1*3*224*224*4 bytes

// Layout planning for distributed processing
auto layout = ImageTensor::optimal_layout(1024, 1024);
if (layout.type == pyflame::MeshLayout::Type::GRID) {
    std::cout << "Using " << layout.pe_rows << "x" << layout.pe_cols << " grid\n";
}

// Halo calculation for convolution
int kernel = 3;
if (ImageTensor::needs_halo(kernel)) {
    int halo = ImageTensor::halo_size(kernel);  // 1
    // Need to exchange 'halo' rows/columns with neighbors
}
```

### Layout Types

The `optimal_layout` function returns a `MeshLayout` with one of these types:

| Type | Description | When Used |
|------|-------------|-----------|
| `SINGLE_PE` | Single processing element | Small images (<32KB) |
| `ROW_PARTITION` | Partition by rows | Tall images |
| `COL_PARTITION` | Partition by columns | Wide images |
| `GRID` | 2D grid of PEs | Large images |
| `BLOCK_CYCLIC` | Block-cyclic distribution | Load balancing |
| `CUSTOM` | User-defined layout | Special cases |

### Memory Layout Strategy

```cpp
// Example: Determine layout for 4K image
int64_t h = 2160, w = 3840;
auto layout = ImageTensor::optimal_layout(h, w);

std::cout << "Total PEs: " << layout.total_pes() << "\n";
std::cout << "Grid: " << layout.pe_rows << " x " << layout.pe_cols << "\n";

// Get tile size for PE at position (0, 0)
auto [tile_h, tile_w] = ImageTensor::tile_shape(h, w, layout, 0, 0);
std::cout << "Tile size: " << tile_h << " x " << tile_w << "\n";
```

---

## Error Handling

All validation functions throw `std::runtime_error` with descriptive messages:

```cpp
try {
    std::vector<int64_t> bad_shape = {3, 224, 224};  // Missing batch dim
    ImageTensor::validate_shape(bad_shape);
} catch (const std::runtime_error& e) {
    // "Invalid image tensor: expected 4D NCHW format, got shape [3, 224, 224]"
    std::cerr << e.what() << std::endl;
}
```

---

## See Also

- [Transforms API](./TRANSFORMS.md) - Transform classes that use ImageTensor
- [Functional API](./FUNCTIONAL.md) - Stateless functions
- [Architecture](../ARCHITECTURE.md) - System design overview

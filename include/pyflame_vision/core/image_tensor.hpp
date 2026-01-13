#pragma once

#include <vector>
#include <tuple>
#include <cstdint>
#include <stdexcept>
#include <cmath>
#include <algorithm>

#include "pyflame_vision/core/exceptions.hpp"
#include "pyflame_vision/core/security.hpp"

// Forward declarations for PyFlame types
// These will be replaced with actual includes when integrated with PyFlame
namespace pyflame {
    class Tensor;
    enum class DType : uint8_t;

    namespace ir {
        struct TensorSpec;
    }

    struct MeshLayout {
        enum class Type { SINGLE_PE, ROW_PARTITION, COL_PARTITION, GRID, BLOCK_CYCLIC, CUSTOM };
        Type type;
        int32_t pe_rows;
        int32_t pe_cols;

        static MeshLayout SinglePE() {
            return MeshLayout{Type::SINGLE_PE, 1, 1};
        }

        static MeshLayout Grid(int32_t rows, int32_t cols) {
            return MeshLayout{Type::GRID, rows, cols};
        }

        static MeshLayout RowPartition(int32_t num_pes) {
            return MeshLayout{Type::ROW_PARTITION, num_pes, 1};
        }

        static MeshLayout ColPartition(int32_t num_pes) {
            return MeshLayout{Type::COL_PARTITION, 1, num_pes};
        }

        int32_t total_pes() const { return pe_rows * pe_cols; }
    };
}

namespace pyflame_vision::core {

/// Image format constants (NCHW = Batch, Channels, Height, Width)
struct ImageFormat {
    static constexpr int BATCH_DIM = 0;
    static constexpr int CHANNEL_DIM = 1;
    static constexpr int HEIGHT_DIM = 2;
    static constexpr int WIDTH_DIM = 3;
    static constexpr int NUM_DIMS = 4;
};

/// Color space definitions
enum class ColorSpace : uint8_t {
    RGB = 0,
    BGR = 1,
    GRAY = 2,
    HSV = 3,
    LAB = 4,
};

/// Get color space name
inline std::string colorspace_name(ColorSpace cs) {
    switch (cs) {
        case ColorSpace::RGB: return "RGB";
        case ColorSpace::BGR: return "BGR";
        case ColorSpace::GRAY: return "GRAY";
        case ColorSpace::HSV: return "HSV";
        case ColorSpace::LAB: return "LAB";
        default: return "unknown";
    }
}

/// Image tensor utilities for NCHW format
/// @note Thread Safety: All static methods are thread-safe as they have no shared state.
class ImageTensor {
public:
    /// Check if a shape represents a valid image (4D NCHW)
    static bool is_valid_shape(const std::vector<int64_t>& shape) {
        if (shape.size() != ImageFormat::NUM_DIMS) return false;
        for (auto dim : shape) {
            if (dim <= 0) return false;
        }
        return true;
    }

    /// Validate shape and throw if invalid
    /// Also validates against security limits to prevent resource exhaustion
    /// @throws ValidationError if shape is not 4D NCHW with positive dimensions
    /// @throws ResourceError if dimensions exceed security limits
    static void validate_shape(const std::vector<int64_t>& shape) {
        if (shape.size() != ImageFormat::NUM_DIMS) {
            throw ValidationError(
                "Invalid image tensor: expected 4D NCHW format, got " +
                std::to_string(shape.size()) + "D tensor"
            );
        }

        for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] <= 0) {
                throw ValidationError(
                    "Invalid image tensor: dimension " + std::to_string(i) +
                    " must be positive, got " + std::to_string(shape[i])
                );
            }
        }

        // Validate against security limits
        int64_t batch = shape[ImageFormat::BATCH_DIM];
        int64_t channels = shape[ImageFormat::CHANNEL_DIM];
        int64_t h = shape[ImageFormat::HEIGHT_DIM];
        int64_t w = shape[ImageFormat::WIDTH_DIM];

        if (batch > SecurityLimits::MAX_BATCH_SIZE) {
            throw ResourceError(
                "Batch size (" + std::to_string(batch) + ") exceeds maximum (" +
                std::to_string(SecurityLimits::MAX_BATCH_SIZE) + ")"
            );
        }
        if (channels > SecurityLimits::MAX_CHANNELS) {
            throw ResourceError(
                "Channel count (" + std::to_string(channels) + ") exceeds maximum (" +
                std::to_string(SecurityLimits::MAX_CHANNELS) + ")"
            );
        }
        if (h > SecurityLimits::MAX_DIMENSION) {
            throw ResourceError(
                "Height (" + std::to_string(h) + ") exceeds maximum (" +
                std::to_string(SecurityLimits::MAX_DIMENSION) + ")"
            );
        }
        if (w > SecurityLimits::MAX_DIMENSION) {
            throw ResourceError(
                "Width (" + std::to_string(w) + ") exceeds maximum (" +
                std::to_string(SecurityLimits::MAX_DIMENSION) + ")"
            );
        }

        // Validate total elements won't overflow or exceed limits
        validate_total_elements(batch, channels, h, w);
    }

    /// Get image dimensions as tuple (batch, channels, height, width)
    static std::tuple<int64_t, int64_t, int64_t, int64_t>
    get_dimensions(const std::vector<int64_t>& shape) {
        validate_shape(shape);
        return {shape[0], shape[1], shape[2], shape[3]};
    }

    /// Get batch size
    static int64_t batch_size(const std::vector<int64_t>& shape) {
        validate_shape(shape);
        return shape[ImageFormat::BATCH_DIM];
    }

    /// Get number of channels
    static int64_t num_channels(const std::vector<int64_t>& shape) {
        validate_shape(shape);
        return shape[ImageFormat::CHANNEL_DIM];
    }

    /// Get height
    static int64_t height(const std::vector<int64_t>& shape) {
        validate_shape(shape);
        return shape[ImageFormat::HEIGHT_DIM];
    }

    /// Get width
    static int64_t width(const std::vector<int64_t>& shape) {
        validate_shape(shape);
        return shape[ImageFormat::WIDTH_DIM];
    }

    /// Create output shape for resize operation
    static std::vector<int64_t> resize_output_shape(
        const std::vector<int64_t>& input_shape,
        int64_t target_height,
        int64_t target_width
    ) {
        validate_shape(input_shape);
        return {
            input_shape[ImageFormat::BATCH_DIM],
            input_shape[ImageFormat::CHANNEL_DIM],
            target_height,
            target_width
        };
    }

    /// Create output shape for crop operation
    static std::vector<int64_t> crop_output_shape(
        const std::vector<int64_t>& input_shape,
        int64_t crop_height,
        int64_t crop_width
    ) {
        validate_shape(input_shape);
        return {
            input_shape[ImageFormat::BATCH_DIM],
            input_shape[ImageFormat::CHANNEL_DIM],
            crop_height,
            crop_width
        };
    }

    /// Compute optimal layout for image processing
    /// @throws ValidationError if height or width is non-positive
    /// @throws ResourceError if dimensions exceed security limits
    static pyflame::MeshLayout optimal_layout(
        int64_t height,
        int64_t width,
        size_t element_size = 4
    ) {
        // Validate dimensions first
        validate_dimension(height, "height");
        validate_dimension(width, "width");

        constexpr size_t MAX_PE_MEMORY = 32 * 1024;  // 32KB usable per PE

        // Use safe multiplication to prevent overflow
        int64_t total_pixels = safe_multiply(height, width, "image dimensions");
        size_t image_bytes = static_cast<size_t>(total_pixels) * element_size;

        // If fits on single PE, use single PE layout
        if (image_bytes <= MAX_PE_MEMORY) {
            return pyflame::MeshLayout::SinglePE();
        }

        // Need to tile across PEs
        // Aim for ~16KB tiles to leave room for halos and scratch
        constexpr size_t TARGET_TILE_BYTES = 16 * 1024;
        size_t target_tile_elements = TARGET_TILE_BYTES / element_size;

        // Calculate number of tiles needed
        size_t total_elements = static_cast<size_t>(total_pixels);
        int num_tiles = static_cast<int>(
            std::ceil(static_cast<double>(total_elements) / target_tile_elements)
        );

        // Find grid dimensions, prefer square tiles
        int grid_rows = static_cast<int>(std::sqrt(static_cast<double>(num_tiles)));
        int grid_cols = (num_tiles + grid_rows - 1) / grid_rows;

        // Adjust based on aspect ratio (guard against division by zero)
        if (width > 0) {
            double aspect = static_cast<double>(height) / static_cast<double>(width);
            if (aspect > 1.5) {
                // Tall image: more rows
                grid_rows = std::min(grid_rows * 2, static_cast<int>(height));
                grid_cols = (num_tiles + grid_rows - 1) / grid_rows;
            } else if (aspect < 0.67) {
                // Wide image: more columns
                grid_cols = std::min(grid_cols * 2, static_cast<int>(width));
                grid_rows = (num_tiles + grid_cols - 1) / grid_cols;
            }
        }

        // Ensure at least 1x1
        grid_rows = std::max(1, grid_rows);
        grid_cols = std::max(1, grid_cols);

        return pyflame::MeshLayout::Grid(grid_rows, grid_cols);
    }

    /// Compute tile shape for a specific PE
    static std::tuple<int64_t, int64_t> tile_shape(
        int64_t height,
        int64_t width,
        const pyflame::MeshLayout& layout,
        int pe_row,
        int pe_col
    ) {
        if (layout.type == pyflame::MeshLayout::Type::SINGLE_PE) {
            return {height, width};
        }

        int64_t tile_h = (height + layout.pe_rows - 1) / layout.pe_rows;
        int64_t tile_w = (width + layout.pe_cols - 1) / layout.pe_cols;

        // Last row/column may be smaller
        int64_t start_h = pe_row * tile_h;
        int64_t start_w = pe_col * tile_w;
        int64_t actual_h = std::min(tile_h, height - start_h);
        int64_t actual_w = std::min(tile_w, width - start_w);

        return {std::max(int64_t(0), actual_h), std::max(int64_t(0), actual_w)};
    }

    /// Check if operation needs halo exchange
    static bool needs_halo(int kernel_size) {
        return kernel_size > 1;
    }

    /// Compute halo size for a kernel
    static int halo_size(int kernel_size) {
        return kernel_size / 2;
    }

    /// Calculate total number of elements
    /// @throws OverflowError if multiplication would overflow
    static int64_t numel(const std::vector<int64_t>& shape) {
        int64_t total = 1;
        for (auto dim : shape) {
            total = safe_multiply(total, dim, "numel calculation");
        }
        return total;
    }

    /// Calculate memory size in bytes
    /// @throws OverflowError if calculation would overflow
    static size_t size_bytes(const std::vector<int64_t>& shape, size_t element_size = 4) {
        int64_t elements = numel(shape);
        int64_t bytes = safe_multiply(elements, static_cast<int64_t>(element_size), "size_bytes");
        return static_cast<size_t>(bytes);
    }
};

}  // namespace pyflame_vision::core

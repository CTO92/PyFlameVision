#pragma once

#include "pyflame_vision/core/image_tensor.hpp"
#include "pyflame_vision/core/interpolation.hpp"
#include "pyflame_vision/core/exceptions.hpp"
#include "pyflame_vision/core/security.hpp"
#include <vector>
#include <tuple>

namespace pyflame_vision::transforms::functional {

/// Compute output shape for resize operation
/// @param input_shape Input image shape [B, C, H, W]
/// @param size Target size (height, width)
/// @return Output shape [B, C, target_height, target_width]
inline std::vector<int64_t> resize_output_shape(
    const std::vector<int64_t>& input_shape,
    std::tuple<int64_t, int64_t> size
) {
    core::ImageTensor::validate_shape(input_shape);
    auto [height, width] = size;
    return core::ImageTensor::resize_output_shape(input_shape, height, width);
}

/// Compute output shape for crop operation
/// @param input_shape Input image shape [B, C, H, W]
/// @param top Top coordinate of crop region
/// @param left Left coordinate of crop region
/// @param height Height of crop region
/// @param width Width of crop region
/// @return Output shape [B, C, height, width]
/// @throws ValidationError if crop parameters are invalid
/// @throws BoundsError if crop region is out of bounds
inline std::vector<int64_t> crop_output_shape(
    const std::vector<int64_t>& input_shape,
    int64_t top,
    int64_t left,
    int64_t height,
    int64_t width
) {
    core::ImageTensor::validate_shape(input_shape);

    // Validate crop dimensions are positive
    if (height <= 0 || width <= 0) {
        throw ValidationError(
            "Crop dimensions must be positive, got height=" +
            std::to_string(height) + ", width=" + std::to_string(width)
        );
    }

    // Validate crop position is non-negative
    if (top < 0 || left < 0) {
        throw ValidationError(
            "Crop position must be non-negative, got top=" +
            std::to_string(top) + ", left=" + std::to_string(left)
        );
    }

    int64_t in_height = core::ImageTensor::height(input_shape);
    int64_t in_width = core::ImageTensor::width(input_shape);

    // Use safe arithmetic for bounds check to prevent overflow
    int64_t crop_bottom = core::safe_add(top, height, "crop bottom");
    int64_t crop_right = core::safe_add(left, width, "crop right");

    // Validate crop bounds
    if (crop_bottom > in_height || crop_right > in_width) {
        throw BoundsError(
            "Crop region (" + std::to_string(top) + ", " + std::to_string(left) +
            ", " + std::to_string(height) + ", " + std::to_string(width) +
            ") out of bounds for image size (" +
            std::to_string(in_height) + ", " + std::to_string(in_width) + ")"
        );
    }

    return core::ImageTensor::crop_output_shape(input_shape, height, width);
}

/// Compute output shape for center crop
/// @param input_shape Input image shape [B, C, H, W]
/// @param size Target size (height, width)
/// @return Output shape [B, C, target_height, target_width]
/// @throws ValidationError if crop size is invalid
/// @throws BoundsError if crop size exceeds image size
inline std::vector<int64_t> center_crop_output_shape(
    const std::vector<int64_t>& input_shape,
    std::tuple<int64_t, int64_t> size
) {
    core::ImageTensor::validate_shape(input_shape);
    auto [height, width] = size;

    // Validate crop dimensions are positive
    if (height <= 0 || width <= 0) {
        throw ValidationError(
            "Center crop dimensions must be positive, got height=" +
            std::to_string(height) + ", width=" + std::to_string(width)
        );
    }

    int64_t in_height = core::ImageTensor::height(input_shape);
    int64_t in_width = core::ImageTensor::width(input_shape);

    if (height > in_height || width > in_width) {
        throw BoundsError(
            "Center crop size (" + std::to_string(height) + ", " + std::to_string(width) +
            ") larger than image size (" +
            std::to_string(in_height) + ", " + std::to_string(in_width) + ")"
        );
    }

    return core::ImageTensor::crop_output_shape(input_shape, height, width);
}

/// Compute output shape for normalize (same as input)
/// @param input_shape Input tensor shape [B, C, H, W]
/// @param mean Per-channel mean values
/// @param std Per-channel std values
/// @return Output shape (same as input)
/// @throws ValidationError if mean/std sizes don't match channels or contain invalid values
inline std::vector<int64_t> normalize_output_shape(
    const std::vector<int64_t>& input_shape,
    const std::vector<float>& mean,
    const std::vector<float>& std
) {
    core::ImageTensor::validate_shape(input_shape);

    int64_t num_channels = core::ImageTensor::num_channels(input_shape);

    if (static_cast<int64_t>(mean.size()) != num_channels) {
        throw ValidationError(
            "Mean size (" + std::to_string(mean.size()) +
            ") does not match number of channels (" +
            std::to_string(num_channels) + ")"
        );
    }

    if (static_cast<int64_t>(std.size()) != num_channels) {
        throw ValidationError(
            "Std size (" + std::to_string(std.size()) +
            ") does not match number of channels (" +
            std::to_string(num_channels) + ")"
        );
    }

    // Validate std values are positive (prevent division by zero)
    for (size_t i = 0; i < std.size(); ++i) {
        if (std[i] <= 0.0f) {
            throw ValidationError(
                "Std values must be positive, got " +
                std::to_string(std[i]) + " at index " + std::to_string(i)
            );
        }
    }

    // Normalize preserves shape
    return input_shape;
}

/// Compute output shape for horizontal flip (same as input)
inline std::vector<int64_t> hflip_output_shape(
    const std::vector<int64_t>& input_shape
) {
    core::ImageTensor::validate_shape(input_shape);
    return input_shape;
}

/// Compute output shape for vertical flip (same as input)
inline std::vector<int64_t> vflip_output_shape(
    const std::vector<int64_t>& input_shape
) {
    core::ImageTensor::validate_shape(input_shape);
    return input_shape;
}

/// Compute center crop bounds
/// @return tuple of (top, left, height, width)
inline std::tuple<int64_t, int64_t, int64_t, int64_t> compute_center_crop_bounds(
    int64_t input_height,
    int64_t input_width,
    int64_t crop_height,
    int64_t crop_width
) {
    int64_t top = (input_height - crop_height) / 2;
    int64_t left = (input_width - crop_width) / 2;
    return {top, left, crop_height, crop_width};
}

/// Compute scale factors for resize
/// @param src_height Source image height
/// @param src_width Source image width
/// @param dst_height Destination image height (must be positive)
/// @param dst_width Destination image width (must be positive)
/// @return tuple of (scale_y, scale_x)
/// @throws ValidationError if any dimension is non-positive
inline std::tuple<float, float> compute_resize_scale(
    int64_t src_height,
    int64_t src_width,
    int64_t dst_height,
    int64_t dst_width
) {
    // Guard against division by zero and invalid dimensions
    if (src_height <= 0 || src_width <= 0) {
        throw ValidationError(
            "Source dimensions must be positive, got height=" +
            std::to_string(src_height) + ", width=" + std::to_string(src_width)
        );
    }
    if (dst_height <= 0 || dst_width <= 0) {
        throw ValidationError(
            "Destination dimensions must be positive, got height=" +
            std::to_string(dst_height) + ", width=" + std::to_string(dst_width)
        );
    }

    float scale_y = static_cast<float>(src_height) / static_cast<float>(dst_height);
    float scale_x = static_cast<float>(src_width) / static_cast<float>(dst_width);
    return {scale_y, scale_x};
}

}  // namespace pyflame_vision::transforms::functional

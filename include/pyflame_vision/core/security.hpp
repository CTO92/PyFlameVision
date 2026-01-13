#pragma once

/// @file security.hpp
/// @brief Security utilities for PyFlame Vision
///
/// This header provides security-related utilities including:
/// - Overflow-safe arithmetic operations
/// - Resource limit validation
/// - Secure random number generation
/// - CSL template parameter sanitization
///
/// @note Thread Safety: All functions in this header are thread-safe.
///       They have no shared mutable state and use only local variables.
///       The generate_secure_seed() function uses thread-safe random_device
///       and chrono facilities.

#include <cstdint>
#include <cmath>
#include <limits>
#include <string>
#include <regex>
#include <chrono>
#include <random>

#include "pyflame_vision/core/exceptions.hpp"

namespace pyflame_vision::core {

/// Security-related constants and limits
/// @note Thread Safety: These are compile-time constants, inherently thread-safe.
struct SecurityLimits {
    /// Maximum allowed dimension for height or width
    static constexpr int64_t MAX_DIMENSION = 65536;  // 64K pixels

    /// Maximum allowed total elements in an image
    static constexpr int64_t MAX_TOTAL_ELEMENTS = 1LL << 32;  // ~4 billion

    /// Maximum allowed batch size
    static constexpr int64_t MAX_BATCH_SIZE = 1024;

    /// Maximum allowed number of channels
    static constexpr int64_t MAX_CHANNELS = 256;

    /// Maximum allowed padding value
    static constexpr int MAX_PADDING = 1024;

    /// Maximum mean/std vector length
    static constexpr size_t MAX_NORM_CHANNELS = 256;

    /// Maximum allowed features for Linear layers (prevents memory exhaustion)
    /// 1 million features allows ~4GB weight matrix in worst case (1M x 1M x 4 bytes)
    static constexpr int64_t MAX_FEATURES = 1LL << 20;  // ~1 million features

    /// Maximum allowed blocks per ResNet layer (prevents DoS via deep networks)
    static constexpr int64_t MAX_RESNET_BLOCKS_PER_LAYER = 1000;

    // ========================================================================
    // Phase 3: Advanced Transform Limits
    // ========================================================================

    /// Maximum allowed rotation angle in degrees
    static constexpr float MAX_ROTATION_ANGLE = 360.0f;

    /// Maximum allowed blur kernel size (must be odd)
    static constexpr int MAX_BLUR_KERNEL_SIZE = 31;

    /// Maximum allowed blur sigma
    static constexpr float MAX_BLUR_SIGMA = 10.0f;

    /// Maximum allowed color jitter factor (brightness, contrast, saturation)
    static constexpr float MAX_COLOR_FACTOR = 2.0f;

    /// Maximum allowed hue shift (in fraction of color wheel, 0.5 = 180 degrees)
    static constexpr float MAX_HUE_SHIFT = 0.5f;

    // ========================================================================
    // Phase 4: Specialized Operations Limits
    // ========================================================================

    /// Maximum number of ROIs per batch for ROI Align
    static constexpr int64_t MAX_ROIS = 10000;

    /// Maximum output size for ROI Align
    static constexpr int64_t MAX_ROI_OUTPUT_SIZE = 256;

    /// Maximum spatial size for grid sample output
    static constexpr int64_t MAX_GRID_SAMPLE_SIZE = 4096;

    /// Maximum number of boxes for NMS
    static constexpr int64_t MAX_NMS_BOXES = 100000;

    /// Maximum coordinate value for grid sample (prevents overflow)
    static constexpr float MAX_GRID_COORDINATE = 1e6f;
};

/// Check for integer overflow in addition
/// @param a First operand
/// @param b Second operand
/// @return true if a + b would overflow
inline bool would_overflow_add(int64_t a, int64_t b) {
    if (b > 0 && a > std::numeric_limits<int64_t>::max() - b) {
        return true;
    }
    if (b < 0 && a < std::numeric_limits<int64_t>::min() - b) {
        return true;
    }
    return false;
}

/// Check for integer overflow in multiplication
/// @param a First operand
/// @param b Second operand
/// @return true if a * b would overflow
inline bool would_overflow_multiply(int64_t a, int64_t b) {
    if (a == 0 || b == 0) return false;

    if (a > 0) {
        if (b > 0) {
            return a > std::numeric_limits<int64_t>::max() / b;
        } else {
            return b < std::numeric_limits<int64_t>::min() / a;
        }
    } else {
        if (b > 0) {
            return a < std::numeric_limits<int64_t>::min() / b;
        } else {
            return a < std::numeric_limits<int64_t>::max() / b;
        }
    }
}

/// Safe addition with overflow check
/// @throws OverflowError if overflow would occur
inline int64_t safe_add(int64_t a, int64_t b, const std::string& context = "") {
    if (would_overflow_add(a, b)) {
        throw OverflowError(
            "Integer overflow in addition" +
            (context.empty() ? "" : " (" + context + ")")
        );
    }
    return a + b;
}

/// Safe multiplication with overflow check
/// @throws OverflowError if overflow would occur
inline int64_t safe_multiply(int64_t a, int64_t b, const std::string& context = "") {
    if (would_overflow_multiply(a, b)) {
        throw OverflowError(
            "Integer overflow in multiplication" +
            (context.empty() ? "" : " (" + context + ")")
        );
    }
    return a * b;
}

/// Validate dimension is within allowed limits
/// @throws ResourceError if dimension exceeds limits
inline void validate_dimension(int64_t value, const std::string& name = "dimension") {
    if (value <= 0) {
        throw ValidationError(name + " must be positive, got " + std::to_string(value));
    }
    if (value > SecurityLimits::MAX_DIMENSION) {
        throw ResourceError(
            name + " (" + std::to_string(value) + ") exceeds maximum allowed (" +
            std::to_string(SecurityLimits::MAX_DIMENSION) + ")"
        );
    }
}

/// Validate total image elements are within limits
/// @throws ResourceError if total exceeds limits
inline void validate_total_elements(int64_t batch, int64_t channels, int64_t height, int64_t width) {
    // Check individual limits first
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

    // Check for overflow in total calculation
    int64_t hw = safe_multiply(height, width, "height * width");
    int64_t chw = safe_multiply(channels, hw, "channels * height * width");
    int64_t total = safe_multiply(batch, chw, "total elements");

    if (total > SecurityLimits::MAX_TOTAL_ELEMENTS) {
        throw ResourceError(
            "Total elements (" + std::to_string(total) + ") exceeds maximum (" +
            std::to_string(SecurityLimits::MAX_TOTAL_ELEMENTS) + ")"
        );
    }
}

/// Validate padding value
/// @throws ValidationError if padding is invalid
inline void validate_padding(int padding) {
    if (padding < 0) {
        throw ValidationError("Padding must be non-negative, got " + std::to_string(padding));
    }
    if (padding > SecurityLimits::MAX_PADDING) {
        throw ResourceError(
            "Padding (" + std::to_string(padding) + ") exceeds maximum (" +
            std::to_string(SecurityLimits::MAX_PADDING) + ")"
        );
    }
}

// ============================================================================
// Phase 3: Advanced Transform Validation Functions
// ============================================================================

/// Validate rotation angle
/// @throws ValidationError if angle is NaN/Inf or exceeds maximum
inline void validate_rotation_angle(float angle) {
    if (!std::isfinite(angle)) {
        throw ValidationError("Rotation angle must be finite, got NaN or Inf");
    }
    if (std::abs(angle) > SecurityLimits::MAX_ROTATION_ANGLE) {
        throw ValidationError(
            "Rotation angle (" + std::to_string(angle) +
            ") exceeds maximum allowed (" +
            std::to_string(SecurityLimits::MAX_ROTATION_ANGLE) + ")"
        );
    }
}

/// Validate blur kernel size (must be positive odd integer)
/// @throws ValidationError if size is not positive odd integer
/// @throws ResourceError if size exceeds maximum
inline void validate_blur_kernel_size(int size) {
    if (size <= 0) {
        throw ValidationError("Blur kernel size must be positive, got " + std::to_string(size));
    }
    if (size % 2 == 0) {
        throw ValidationError("Blur kernel size must be odd, got " + std::to_string(size));
    }
    if (size > SecurityLimits::MAX_BLUR_KERNEL_SIZE) {
        throw ResourceError(
            "Blur kernel size (" + std::to_string(size) +
            ") exceeds maximum allowed (" +
            std::to_string(SecurityLimits::MAX_BLUR_KERNEL_SIZE) + ")"
        );
    }
}

/// Validate blur sigma
/// @throws ValidationError if sigma is NaN/Inf or not positive
/// @throws ResourceError if sigma exceeds maximum
inline void validate_blur_sigma(float sigma) {
    if (!std::isfinite(sigma)) {
        throw ValidationError("Blur sigma must be finite, got NaN or Inf");
    }
    if (sigma <= 0.0f) {
        throw ValidationError("Blur sigma must be positive, got " + std::to_string(sigma));
    }
    if (sigma > SecurityLimits::MAX_BLUR_SIGMA) {
        throw ResourceError(
            "Blur sigma (" + std::to_string(sigma) +
            ") exceeds maximum allowed (" +
            std::to_string(SecurityLimits::MAX_BLUR_SIGMA) + ")"
        );
    }
}

/// Validate color jitter factor (brightness, contrast, saturation)
/// @throws ValidationError if factor is NaN/Inf or negative
/// @throws ResourceError if factor exceeds maximum
inline void validate_color_factor(float factor, const std::string& name) {
    if (!std::isfinite(factor)) {
        throw ValidationError(name + " factor must be finite, got NaN or Inf");
    }
    if (factor < 0.0f) {
        throw ValidationError(name + " factor must be non-negative, got " + std::to_string(factor));
    }
    if (factor > SecurityLimits::MAX_COLOR_FACTOR) {
        throw ResourceError(
            name + " factor (" + std::to_string(factor) +
            ") exceeds maximum allowed (" +
            std::to_string(SecurityLimits::MAX_COLOR_FACTOR) + ")"
        );
    }
}

/// Validate hue shift factor
/// @throws ValidationError if factor is NaN/Inf
/// @throws ResourceError if factor exceeds maximum
inline void validate_hue_factor(float factor) {
    if (!std::isfinite(factor)) {
        throw ValidationError("Hue factor must be finite, got NaN or Inf");
    }
    if (std::abs(factor) > SecurityLimits::MAX_HUE_SHIFT) {
        throw ResourceError(
            "Hue factor (" + std::to_string(factor) +
            ") exceeds maximum allowed (" +
            std::to_string(SecurityLimits::MAX_HUE_SHIFT) + ")"
        );
    }
}

// ============================================================================
// Phase 4: Specialized Operations Validation Functions
// ============================================================================

/// Validate ROI Align parameters
/// @throws ValidationError if parameters are invalid
/// @throws ResourceError if parameters exceed limits
inline void validate_roi_align_params(
    int64_t output_height,
    int64_t output_width,
    float spatial_scale,
    int sampling_ratio
) {
    if (output_height <= 0 || output_width <= 0) {
        throw ValidationError(
            "ROI Align output size must be positive, got (" +
            std::to_string(output_height) + ", " + std::to_string(output_width) + ")"
        );
    }
    if (output_height > SecurityLimits::MAX_ROI_OUTPUT_SIZE ||
        output_width > SecurityLimits::MAX_ROI_OUTPUT_SIZE) {
        throw ResourceError(
            "ROI Align output size exceeds maximum (" +
            std::to_string(SecurityLimits::MAX_ROI_OUTPUT_SIZE) + ")"
        );
    }
    if (!std::isfinite(spatial_scale) || spatial_scale <= 0.0f) {
        throw ValidationError(
            "ROI Align spatial_scale must be positive finite, got " +
            std::to_string(spatial_scale)
        );
    }
    if (sampling_ratio < 0) {
        throw ValidationError(
            "ROI Align sampling_ratio must be non-negative, got " +
            std::to_string(sampling_ratio)
        );
    }
}

/// Validate grid sample coordinates
/// @throws ValidationError if coordinates are NaN/Inf or exceed limits
inline void validate_grid_coordinate(float coord, const std::string& name) {
    if (!std::isfinite(coord)) {
        throw ValidationError(name + " must be finite, got NaN or Inf");
    }
    if (std::abs(coord) > SecurityLimits::MAX_GRID_COORDINATE) {
        throw ResourceError(
            name + " (" + std::to_string(coord) +
            ") exceeds maximum allowed (" +
            std::to_string(SecurityLimits::MAX_GRID_COORDINATE) + ")"
        );
    }
}

/// Validate NMS parameters
/// @throws ValidationError if parameters are invalid
inline void validate_nms_params(float iou_threshold, float score_threshold) {
    if (!std::isfinite(iou_threshold)) {
        throw ValidationError("NMS iou_threshold must be finite, got NaN or Inf");
    }
    if (iou_threshold < 0.0f || iou_threshold > 1.0f) {
        throw ValidationError(
            "NMS iou_threshold must be in [0, 1], got " + std::to_string(iou_threshold)
        );
    }
    if (!std::isfinite(score_threshold)) {
        throw ValidationError("NMS score_threshold must be finite, got NaN or Inf");
    }
    if (score_threshold < 0.0f) {
        throw ValidationError(
            "NMS score_threshold must be non-negative, got " + std::to_string(score_threshold)
        );
    }
}

/// Validate number of ROIs
/// @throws ResourceError if count exceeds limit
inline void validate_roi_count(int64_t count) {
    if (count < 0) {
        throw ValidationError("ROI count must be non-negative, got " + std::to_string(count));
    }
    if (count > SecurityLimits::MAX_ROIS) {
        throw ResourceError(
            "ROI count (" + std::to_string(count) +
            ") exceeds maximum allowed (" +
            std::to_string(SecurityLimits::MAX_ROIS) + ")"
        );
    }
}

/// Validate number of boxes for NMS
/// @throws ResourceError if count exceeds limit
inline void validate_nms_box_count(int64_t count) {
    if (count < 0) {
        throw ValidationError("Box count must be non-negative, got " + std::to_string(count));
    }
    if (count > SecurityLimits::MAX_NMS_BOXES) {
        throw ResourceError(
            "Box count (" + std::to_string(count) +
            ") exceeds maximum allowed (" +
            std::to_string(SecurityLimits::MAX_NMS_BOXES) + ")"
        );
    }
}

/// Secure random seed generation using multiple entropy sources
inline uint64_t generate_secure_seed() {
    std::random_device rd;

    // Combine multiple entropy sources
    uint64_t seed = rd();

    // Add high-resolution time
    auto time_point = std::chrono::high_resolution_clock::now();
    auto time_since_epoch = time_point.time_since_epoch();
    uint64_t time_seed = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(time_since_epoch).count()
    );

    // Mix the seeds
    seed ^= (time_seed << 32) | (time_seed >> 32);

    // Add another random_device call for more entropy
    seed ^= (static_cast<uint64_t>(rd()) << 32);

    return seed;
}

/// CSL template parameter sanitization
namespace template_security {

/// Maximum allowed string length for template parameters (defense-in-depth)
static constexpr size_t MAX_TEMPLATE_STRING_LENGTH = 1024;

/// Allowed characters in template parameter names
inline bool is_valid_param_name_char(char c) {
    return (c >= 'A' && c <= 'Z') ||
           (c >= 'a' && c <= 'z') ||
           (c >= '0' && c <= '9') ||
           c == '_';
}

/// Validate template parameter name
/// @throws TemplateError if name is empty, too long, or contains invalid characters
inline void validate_param_name(const std::string& name) {
    if (name.empty()) {
        throw TemplateError("Template parameter name cannot be empty");
    }
    if (name.length() > MAX_TEMPLATE_STRING_LENGTH) {
        throw TemplateError("Template parameter name too long (max " +
            std::to_string(MAX_TEMPLATE_STRING_LENGTH) + " characters)");
    }
    for (char c : name) {
        if (!is_valid_param_name_char(c)) {
            throw TemplateError(
                "Invalid character in template parameter name: '" +
                std::string(1, c) + "'"
            );
        }
    }
}

/// Allowed characters for numeric template values
inline bool is_valid_numeric_char(char c) {
    return (c >= '0' && c <= '9') ||
           c == '.' || c == '-' || c == '+' ||
           c == 'e' || c == 'E' ||
           c == 'f' || c == 'F';  // For float literals
}

/// Validate numeric template value
/// @throws TemplateError if value is empty, too long, or contains invalid characters
inline void validate_numeric_value(const std::string& value) {
    if (value.empty()) {
        throw TemplateError("Template numeric value cannot be empty");
    }
    if (value.length() > MAX_TEMPLATE_STRING_LENGTH) {
        throw TemplateError("Template numeric value too long (max " +
            std::to_string(MAX_TEMPLATE_STRING_LENGTH) + " characters)");
    }
    for (char c : value) {
        if (!is_valid_numeric_char(c)) {
            throw TemplateError(
                "Invalid character in template numeric value: '" +
                std::string(1, c) + "'"
            );
        }
    }
}

/// Validate string template value (for CSL identifiers)
/// Only allows alphanumeric and underscore
/// @throws TemplateError if value is empty, too long, or contains invalid characters
inline void validate_identifier_value(const std::string& value) {
    if (value.empty()) {
        throw TemplateError("Template identifier value cannot be empty");
    }
    if (value.length() > MAX_TEMPLATE_STRING_LENGTH) {
        throw TemplateError("Template identifier value too long (max " +
            std::to_string(MAX_TEMPLATE_STRING_LENGTH) + " characters)");
    }
    // First character must be letter or underscore
    if (!((value[0] >= 'A' && value[0] <= 'Z') ||
          (value[0] >= 'a' && value[0] <= 'z') ||
          value[0] == '_')) {
        throw TemplateError("Template identifier must start with letter or underscore");
    }
    for (char c : value) {
        if (!is_valid_param_name_char(c)) {
            throw TemplateError(
                "Invalid character in template identifier: '" +
                std::string(1, c) + "'"
            );
        }
    }
}

/// Validate CSL type value (f32, f16, i32, etc.)
inline void validate_dtype_value(const std::string& value) {
    static const std::vector<std::string> valid_types = {
        "f32", "f16", "bf16", "i32", "i16", "i8", "u32", "u16", "u8", "bool"
    };
    for (const auto& valid : valid_types) {
        if (value == valid) {
            return;
        }
    }
    throw TemplateError("Invalid CSL data type: '" + value + "'");
}

/// Validate array of numeric values (e.g., for mean/std arrays)
/// Format: "0.485, 0.456, 0.406"
inline void validate_numeric_array(const std::string& value) {
    if (value.empty()) {
        throw TemplateError("Template numeric array cannot be empty");
    }
    for (char c : value) {
        if (!is_valid_numeric_char(c) && c != ',' && c != ' ') {
            throw TemplateError(
                "Invalid character in template numeric array: '" +
                std::string(1, c) + "'"
            );
        }
    }
}

/// Sanitize a string for CSL comments (removes newlines and dangerous chars)
inline std::string sanitize_comment(const std::string& input) {
    std::string result;
    result.reserve(input.size());
    for (char c : input) {
        if (c == '\n' || c == '\r') {
            result += ' ';
        } else if (c >= 32 && c < 127) {  // Printable ASCII only
            result += c;
        }
    }
    return result;
}

}  // namespace template_security

}  // namespace pyflame_vision::core

#pragma once

/// @file interpolation.hpp
/// @brief Interpolation and padding mode utilities for spatial operations
///
/// Provides enums and utility functions for interpolation modes used in
/// resize, grid sample, ROI align, and other spatial operations.

#include <cstdint>
#include <string>
#include <cmath>
#include "pyflame_vision/core/exceptions.hpp"

namespace pyflame_vision::core {

/// Interpolation modes for image resizing and spatial sampling
enum class InterpolationMode : uint8_t {
    NEAREST = 0,    // Nearest neighbor interpolation
    BILINEAR = 1,   // Bilinear interpolation
    BICUBIC = 2,    // Bicubic interpolation
    AREA = 3,       // Area-based resampling (for downscaling)
};

/// Padding modes for out-of-bounds sampling
enum class PaddingMode : uint8_t {
    ZEROS = 0,       // Pad with zeros
    BORDER = 1,      // Clamp to border values
    REFLECTION = 2,  // Reflect at boundaries
};

/// Get string name of interpolation mode
inline std::string interpolation_name(InterpolationMode mode) {
    switch (mode) {
        case InterpolationMode::NEAREST: return "nearest";
        case InterpolationMode::BILINEAR: return "bilinear";
        case InterpolationMode::BICUBIC: return "bicubic";
        case InterpolationMode::AREA: return "area";
        default: throw ValidationError("Unknown interpolation mode");
    }
}

/// Parse interpolation mode from string
/// @throws ValidationError if mode is unknown
inline InterpolationMode interpolation_from_string(const std::string& name) {
    if (name == "nearest") return InterpolationMode::NEAREST;
    if (name == "bilinear") return InterpolationMode::BILINEAR;
    if (name == "bicubic") return InterpolationMode::BICUBIC;
    if (name == "area") return InterpolationMode::AREA;
    throw ValidationError("Unknown interpolation mode: " + name);
}

/// Get string name of padding mode
inline std::string padding_mode_name(PaddingMode mode) {
    switch (mode) {
        case PaddingMode::ZEROS: return "zeros";
        case PaddingMode::BORDER: return "border";
        case PaddingMode::REFLECTION: return "reflection";
        default: throw ValidationError("Unknown padding mode");
    }
}

/// Parse padding mode from string
/// @throws ValidationError if mode is unknown
inline PaddingMode padding_mode_from_string(const std::string& name) {
    if (name == "zeros") return PaddingMode::ZEROS;
    if (name == "border") return PaddingMode::BORDER;
    if (name == "reflection") return PaddingMode::REFLECTION;
    throw ValidationError("Unknown padding mode: " + name);
}

/// Get the kernel size needed for interpolation
inline int interpolation_kernel_size(InterpolationMode mode) {
    switch (mode) {
        case InterpolationMode::NEAREST: return 1;
        case InterpolationMode::BILINEAR: return 2;
        case InterpolationMode::BICUBIC: return 4;
        case InterpolationMode::AREA: return 1;  // Variable, depends on scale
        default: return 1;
    }
}

/// Get the halo size needed for distributed interpolation
inline int interpolation_halo_size(InterpolationMode mode) {
    switch (mode) {
        case InterpolationMode::NEAREST: return 0;
        case InterpolationMode::BILINEAR: return 1;
        case InterpolationMode::BICUBIC: return 2;
        case InterpolationMode::AREA: return 1;
        default: return 0;
    }
}

/// Calculate bicubic interpolation weight
/// Uses Catmull-Rom spline (a = -0.5) for smooth interpolation
/// @param t Distance from sample point (typically in [-2, 2])
/// @return Weight contribution for the sample
inline float bicubic_weight(float t) {
    constexpr float a = -0.5f;  // Catmull-Rom coefficient
    float abs_t = std::abs(t);
    if (abs_t <= 1.0f) {
        return ((a + 2.0f) * abs_t - (a + 3.0f)) * abs_t * abs_t + 1.0f;
    } else if (abs_t < 2.0f) {
        return ((a * abs_t - 5.0f * a) * abs_t + 8.0f * a) * abs_t - 4.0f * a;
    }
    return 0.0f;
}

/// Calculate all four bicubic weights for a fractional position
/// @param frac Fractional position in [0, 1)
/// @param weights Output array of 4 weights
inline void bicubic_weights(float frac, float weights[4]) {
    weights[0] = bicubic_weight(frac + 1.0f);
    weights[1] = bicubic_weight(frac);
    weights[2] = bicubic_weight(frac - 1.0f);
    weights[3] = bicubic_weight(frac - 2.0f);
}

}  // namespace pyflame_vision::core

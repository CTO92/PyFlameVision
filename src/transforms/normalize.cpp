#include "pyflame_vision/transforms/normalize.hpp"
#include "pyflame_vision/core/security.hpp"
#include <sstream>
#include <cmath>

namespace pyflame_vision::transforms {

// Static helper to validate params before construction
static void validate_normalize_params_static(
    const std::vector<float>& mean,
    const std::vector<float>& std
) {
    if (mean.empty()) {
        throw ValidationError("Normalize: mean cannot be empty");
    }

    if (std.empty()) {
        throw ValidationError("Normalize: std cannot be empty");
    }

    if (mean.size() != std.size()) {
        throw ValidationError(
            "Normalize: mean and std must have the same length, got " +
            std::to_string(mean.size()) + " and " + std::to_string(std.size())
        );
    }

    if (mean.size() > core::SecurityLimits::MAX_NORM_CHANNELS) {
        throw ResourceError(
            "Normalize: mean/std length (" + std::to_string(mean.size()) +
            ") exceeds maximum (" + std::to_string(core::SecurityLimits::MAX_NORM_CHANNELS) + ")"
        );
    }

    // Check for invalid std values
    for (size_t i = 0; i < std.size(); ++i) {
        if (std[i] <= 0.0f) {
            throw ValidationError(
                "Normalize: std values must be positive, got " +
                std::to_string(std[i]) + " at index " + std::to_string(i)
            );
        }
        if (!std::isfinite(std[i])) {
            throw ValidationError(
                "Normalize: std values must be finite, got invalid value at index " +
                std::to_string(i)
            );
        }
    }

    // Check mean values are finite
    for (size_t i = 0; i < mean.size(); ++i) {
        if (!std::isfinite(mean[i])) {
            throw ValidationError(
                "Normalize: mean values must be finite, got invalid value at index " +
                std::to_string(i)
            );
        }
    }
}

Normalize::Normalize(std::vector<float> mean, std::vector<float> std, bool inplace)
    : mean_()  // Initialize empty, will be set after validation
    , std_()
    , inplace_(inplace)
{
    // Validate BEFORE storing to ensure object is never in invalid state
    validate_normalize_params_static(mean, std);

    // Now safe to store
    mean_ = std::move(mean);
    std_ = std::move(std);

    // Precompute 1/std for efficiency
    inv_std_.reserve(std_.size());
    for (float s : std_) {
        inv_std_.push_back(1.0f / s);
    }
}

std::vector<int64_t> Normalize::get_output_shape(
    const std::vector<int64_t>& input_shape
) const {
    validate_input(input_shape);

    int64_t num_channels = core::ImageTensor::num_channels(input_shape);
    validate_channels(num_channels);

    // Normalize preserves shape
    return input_shape;
}

std::string Normalize::repr() const {
    std::ostringstream ss;
    ss << "Normalize(mean=[";
    for (size_t i = 0; i < mean_.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << mean_[i];
    }
    ss << "], std=[";
    for (size_t i = 0; i < std_.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << std_[i];
    }
    ss << "]";
    if (inplace_) {
        ss << ", inplace=True";
    }
    ss << ")";
    return ss.str();
}

void Normalize::validate_params() const {
    // Already validated at construction, but re-validate for safety
    validate_normalize_params_static(mean_, std_);
}

void Normalize::validate_channels(int64_t num_channels) const {
    if (static_cast<size_t>(num_channels) != mean_.size()) {
        throw ValidationError(
            "Normalize: mean/std length (" + std::to_string(mean_.size()) +
            ") does not match number of channels (" + std::to_string(num_channels) + ")"
        );
    }
}

}  // namespace pyflame_vision::transforms

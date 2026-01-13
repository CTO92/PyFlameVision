#pragma once

/// @file rotation.hpp
/// @brief Random rotation transform for data augmentation
///
/// Provides RandomRotation transform equivalent to
/// torchvision.transforms.RandomRotation.

#include "pyflame_vision/transforms/random_transform.hpp"
#include "pyflame_vision/core/interpolation.hpp"
#include <cmath>
#include <optional>
#include <utility>
#include <sstream>

namespace pyflame_vision::transforms {

/// Fill mode for areas outside the original image during rotation
enum class RotationFillMode : uint8_t {
    CONSTANT = 0,   // Fill with constant value
    REFLECT = 1,    // Reflect at boundary
    REPLICATE = 2,  // Replicate edge pixels
};

/// Random rotation transform
///
/// Rotates image by a random angle within specified range.
/// Equivalent to torchvision.transforms.RandomRotation
///
/// @note Thread Safety: Thread-safe via inherited mutex-protected RNG.
class RandomRotation : public RandomTransform {
public:
    /// Create rotation transform with symmetric angle range [-degrees, +degrees]
    /// @param degrees Maximum rotation angle in degrees
    /// @param interpolation Interpolation mode for resampling
    /// @param expand If true, expand output to fit rotated image
    /// @param center Center of rotation (default: image center)
    /// @param fill Fill value for areas outside the image
    explicit RandomRotation(
        float degrees,
        core::InterpolationMode interpolation = core::InterpolationMode::BILINEAR,
        bool expand = false,
        std::optional<std::pair<float, float>> center = std::nullopt,
        std::vector<float> fill = {0.0f}
    )
        : degrees_min_(-std::abs(degrees))
        , degrees_max_(std::abs(degrees))
        , interpolation_(interpolation)
        , expand_(expand)
        , center_(center)
        , fill_(std::move(fill))
    {
        validate_params();
    }

    /// Create rotation transform with asymmetric angle range [degrees_min, degrees_max]
    /// @param degrees_min Minimum rotation angle in degrees
    /// @param degrees_max Maximum rotation angle in degrees
    /// @param interpolation Interpolation mode for resampling
    /// @param expand If true, expand output to fit rotated image
    /// @param center Center of rotation (default: image center)
    /// @param fill Fill value for areas outside the image
    RandomRotation(
        float degrees_min,
        float degrees_max,
        core::InterpolationMode interpolation = core::InterpolationMode::BILINEAR,
        bool expand = false,
        std::optional<std::pair<float, float>> center = std::nullopt,
        std::vector<float> fill = {0.0f}
    )
        : degrees_min_(degrees_min)
        , degrees_max_(degrees_max)
        , interpolation_(interpolation)
        , expand_(expand)
        , center_(center)
        , fill_(std::move(fill))
    {
        validate_params();
    }

    std::vector<int64_t> get_output_shape(
        const std::vector<int64_t>& input_shape
    ) const override {
        validate_input(input_shape);

        int64_t batch = input_shape[0];
        int64_t channels = input_shape[1];
        int64_t height = input_shape[2];
        int64_t width = input_shape[3];

        // Generate random angle
        last_angle_ = random_uniform(degrees_min_, degrees_max_);

        if (!expand_) {
            // Output size same as input
            return input_shape;
        }

        // Compute expanded size
        auto [new_h, new_w] = compute_expanded_size(height, width, last_angle_);
        return {batch, channels, new_h, new_w};
    }

    std::string name() const override { return "RandomRotation"; }

    std::string repr() const override {
        std::ostringstream ss;
        ss << "RandomRotation(degrees=(" << degrees_min_ << ", " << degrees_max_ << ")";
        if (interpolation_ != core::InterpolationMode::BILINEAR) {
            ss << ", interpolation=" << core::interpolation_name(interpolation_);
        }
        if (expand_) {
            ss << ", expand=True";
        }
        ss << ")";
        return ss.str();
    }

    /// Get angle range
    std::pair<float, float> degrees() const { return {degrees_min_, degrees_max_}; }

    /// Get last applied rotation angle
    float last_angle() const { return last_angle_; }

    /// Get interpolation mode
    core::InterpolationMode interpolation() const { return interpolation_; }

    /// Check if output is expanded
    bool expand() const { return expand_; }

    /// Get fill values
    const std::vector<float>& fill() const { return fill_; }

    /// Get rotation center (if set)
    std::optional<std::pair<float, float>> center() const { return center_; }

    /// Get halo size needed for distributed execution
    int halo_size() const {
        return core::interpolation_halo_size(interpolation_);
    }

private:
    float degrees_min_;
    float degrees_max_;
    core::InterpolationMode interpolation_;
    bool expand_;
    std::optional<std::pair<float, float>> center_;
    std::vector<float> fill_;
    mutable float last_angle_ = 0.0f;

    void validate_params() const {
        core::validate_rotation_angle(degrees_min_);
        core::validate_rotation_angle(degrees_max_);

        if (degrees_min_ > degrees_max_) {
            throw ValidationError(
                "degrees_min (" + std::to_string(degrees_min_) +
                ") must be <= degrees_max (" + std::to_string(degrees_max_) + ")"
            );
        }

        // Validate fill values
        for (float f : fill_) {
            if (!std::isfinite(f)) {
                throw ValidationError("Fill values must be finite");
            }
        }
    }

    /// Compute output size for expanded rotation
    std::pair<int64_t, int64_t> compute_expanded_size(
        int64_t height, int64_t width, float angle
    ) const {
        // Convert to radians
        constexpr float PI = 3.14159265358979323846f;
        float rad = angle * PI / 180.0f;

        float cos_a = std::abs(std::cos(rad));
        float sin_a = std::abs(std::sin(rad));

        // New dimensions to contain rotated image
        float new_w = static_cast<float>(width) * cos_a + static_cast<float>(height) * sin_a;
        float new_h = static_cast<float>(width) * sin_a + static_cast<float>(height) * cos_a;

        return {
            static_cast<int64_t>(std::ceil(new_h)),
            static_cast<int64_t>(std::ceil(new_w))
        };
    }
};

}  // namespace pyflame_vision::transforms

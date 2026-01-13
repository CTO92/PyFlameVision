#pragma once

#include "pyflame_vision/transforms/transform_base.hpp"
#include "pyflame_vision/core/interpolation.hpp"

namespace pyflame_vision::transforms {

/// Resize transform
/// Resizes an image to the given size using the specified interpolation mode.
/// Equivalent to torchvision.transforms.Resize
///
/// Example usage:
///   auto resize = Resize(Size(224, 224), InterpolationMode::BILINEAR);
///   auto output_shape = resize.get_output_shape(input_shape);
///
/// @note Thread Safety: This class is thread-safe. All member variables are
///       initialized during construction and never modified afterwards.
///       Multiple threads can safely call get_output_shape() and accessors
///       concurrently on the same instance.
class Resize : public SizeTransform {
public:
    /// Create resize transform
    /// @param size Target size (height, width) or single int for square
    /// @param interpolation Interpolation algorithm to use
    /// @param antialias Use antialiasing (recommended for downscaling)
    explicit Resize(
        Size size,
        core::InterpolationMode interpolation = core::InterpolationMode::BILINEAR,
        bool antialias = true
    );

    /// Get the output shape after resize
    std::vector<int64_t> get_output_shape(
        const std::vector<int64_t>& input_shape
    ) const override;

    std::string name() const override { return "Resize"; }
    std::string repr() const override;

    /// Get interpolation mode
    core::InterpolationMode interpolation() const { return interpolation_; }

    /// Get antialias setting
    bool antialias() const { return antialias_; }

    /// Get the halo size needed for this resize operation
    int halo_size() const {
        return core::interpolation_halo_size(interpolation_);
    }

private:
    core::InterpolationMode interpolation_;
    bool antialias_;
};

}  // namespace pyflame_vision::transforms

#pragma once

#include "pyflame_vision/transforms/transform_base.hpp"
#include <vector>

namespace pyflame_vision::transforms {

/// Normalize transform
/// Normalizes a tensor image with mean and standard deviation.
/// output = (input - mean) / std
/// Equivalent to torchvision.transforms.Normalize
///
/// Example usage (ImageNet normalization):
///   auto normalize = Normalize(
///       {0.485f, 0.456f, 0.406f},  // mean
///       {0.229f, 0.224f, 0.225f}   // std
///   );
///
/// @note Thread Safety: This class is thread-safe. All member variables are
///       initialized during construction and never modified afterwards.
///       Multiple threads can safely call get_output_shape() and accessors
///       concurrently on the same instance.
class Normalize : public Transform {
public:
    /// Create normalize transform
    /// @param mean Per-channel mean values
    /// @param std Per-channel standard deviation values
    /// @param inplace If true, modify tensor in-place (not supported in lazy mode)
    Normalize(
        std::vector<float> mean,
        std::vector<float> std,
        bool inplace = false
    );

    std::vector<int64_t> get_output_shape(
        const std::vector<int64_t>& input_shape
    ) const override;

    std::string name() const override { return "Normalize"; }
    std::string repr() const override;

    /// Get mean values
    const std::vector<float>& mean() const { return mean_; }

    /// Get std values
    const std::vector<float>& std() const { return std_; }

    /// Get inplace setting
    bool inplace() const { return inplace_; }

    /// Get precomputed 1/std values for efficient computation
    const std::vector<float>& inv_std() const { return inv_std_; }

private:
    std::vector<float> mean_;
    std::vector<float> std_;
    std::vector<float> inv_std_;  // Precomputed 1/std
    bool inplace_;

    void validate_params() const;
    void validate_channels(int64_t num_channels) const;
};

}  // namespace pyflame_vision::transforms

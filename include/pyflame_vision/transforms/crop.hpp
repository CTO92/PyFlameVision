#pragma once

#include "pyflame_vision/transforms/transform_base.hpp"
#include <random>
#include <tuple>

namespace pyflame_vision::transforms {

/// Center crop transform
/// Crops the center of an image to the given size.
/// Equivalent to torchvision.transforms.CenterCrop
///
/// Example usage:
///   auto crop = CenterCrop(Size(224, 224));
///   auto output_shape = crop.get_output_shape(input_shape);
///
/// @note Thread Safety: This class is thread-safe. All member variables are
///       initialized during construction and never modified afterwards.
///       Multiple threads can safely call get_output_shape() and compute_bounds()
///       concurrently on the same instance.
class CenterCrop : public SizeTransform {
public:
    /// Create center crop transform
    /// @param size Target crop size (height, width)
    explicit CenterCrop(Size size);

    std::vector<int64_t> get_output_shape(
        const std::vector<int64_t>& input_shape
    ) const override;

    std::string name() const override { return "CenterCrop"; }
    std::string repr() const override;

    /// Compute crop bounds (top, left, height, width)
    std::tuple<int64_t, int64_t, int64_t, int64_t> compute_bounds(
        int64_t input_height,
        int64_t input_width
    ) const;

private:
    void validate_crop_size(int64_t input_height, int64_t input_width) const;
};

/// Random crop transform
/// Crops a random region of an image to the given size.
/// Equivalent to torchvision.transforms.RandomCrop
///
/// Example usage:
///   auto crop = RandomCrop(Size(224, 224));
///   crop.set_seed(42);  // For reproducibility
///   auto output_shape = crop.get_output_shape(input_shape);
///
/// @note Thread Safety: This class is NOT thread-safe. The random number
///       generator (rng_) is mutable and modified on each call to
///       get_random_params(). Do not share instances across threads without
///       external synchronization. For multi-threaded usage, create separate
///       RandomCrop instances per thread or protect access with a mutex.
class RandomCrop : public SizeTransform {
public:
    /// Create random crop transform
    /// @param size Target crop size
    /// @param padding Padding to apply before cropping
    /// @param pad_if_needed If true, pad image if smaller than crop size
    /// @param fill Fill value for padding
    explicit RandomCrop(
        Size size,
        int padding = 0,
        bool pad_if_needed = false,
        float fill = 0.0f
    );

    std::vector<int64_t> get_output_shape(
        const std::vector<int64_t>& input_shape
    ) const override;

    std::string name() const override { return "RandomCrop"; }
    std::string repr() const override;
    bool is_deterministic() const override { return false; }

    /// Set random seed for reproducibility
    void set_seed(uint64_t seed);

    /// Get current random parameters (top, left)
    /// These are computed lazily and stored for the transform
    std::tuple<int64_t, int64_t> get_random_params(
        int64_t input_height,
        int64_t input_width
    ) const;

    /// Get padding amount
    int padding() const { return padding_; }

    /// Check if padding is applied when needed
    bool pad_if_needed() const { return pad_if_needed_; }

private:
    int padding_;
    bool pad_if_needed_;
    float fill_;
    mutable std::mt19937_64 rng_;
    mutable bool seeded_ = false;

    std::tuple<int64_t, int64_t> compute_padded_size(
        int64_t input_height,
        int64_t input_width
    ) const;
};

}  // namespace pyflame_vision::transforms

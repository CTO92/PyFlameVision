#pragma once

/// @file flip.hpp
/// @brief Random flip transforms for data augmentation
///
/// Provides RandomHorizontalFlip and RandomVerticalFlip transforms
/// equivalent to torchvision.transforms.RandomHorizontalFlip and
/// torchvision.transforms.RandomVerticalFlip.

#include "pyflame_vision/transforms/random_transform.hpp"
#include <sstream>
#include <cmath>

namespace pyflame_vision::transforms {

/// Random horizontal flip transform
///
/// Flips image horizontally with given probability.
/// Equivalent to torchvision.transforms.RandomHorizontalFlip
///
/// @note Thread Safety: Thread-safe via inherited mutex-protected RNG.
class RandomHorizontalFlip : public RandomTransform {
public:
    /// Create horizontal flip transform
    /// @param p Probability of flipping (0.0 to 1.0)
    /// @throws ValidationError if p is not in [0, 1]
    explicit RandomHorizontalFlip(float p = 0.5f)
        : p_(p)
    {
        validate_probability();
    }

    std::vector<int64_t> get_output_shape(
        const std::vector<int64_t>& input_shape
    ) const override {
        validate_input(input_shape);

        // Decide whether to flip (lazy - this sets metadata for CSL generation)
        last_flipped_ = random_bool(p_);

        // Shape is unchanged
        return input_shape;
    }

    std::string name() const override { return "RandomHorizontalFlip"; }

    std::string repr() const override {
        std::ostringstream ss;
        ss << "RandomHorizontalFlip(p=" << p_ << ")";
        return ss.str();
    }

    /// Get flip probability
    float probability() const { return p_; }

    /// Check if last call resulted in a flip
    bool was_flipped() const { return last_flipped_; }

private:
    float p_;
    mutable bool last_flipped_ = false;

    void validate_probability() const {
        if (!std::isfinite(p_)) {
            throw ValidationError("Flip probability must be finite, got NaN or Inf");
        }
        if (p_ < 0.0f || p_ > 1.0f) {
            throw ValidationError(
                "Flip probability must be in [0, 1], got " + std::to_string(p_)
            );
        }
    }
};

/// Random vertical flip transform
///
/// Flips image vertically with given probability.
/// Equivalent to torchvision.transforms.RandomVerticalFlip
///
/// @note Thread Safety: Thread-safe via inherited mutex-protected RNG.
class RandomVerticalFlip : public RandomTransform {
public:
    /// Create vertical flip transform
    /// @param p Probability of flipping (0.0 to 1.0)
    /// @throws ValidationError if p is not in [0, 1]
    explicit RandomVerticalFlip(float p = 0.5f)
        : p_(p)
    {
        validate_probability();
    }

    std::vector<int64_t> get_output_shape(
        const std::vector<int64_t>& input_shape
    ) const override {
        validate_input(input_shape);

        // Decide whether to flip
        last_flipped_ = random_bool(p_);

        // Shape is unchanged
        return input_shape;
    }

    std::string name() const override { return "RandomVerticalFlip"; }

    std::string repr() const override {
        std::ostringstream ss;
        ss << "RandomVerticalFlip(p=" << p_ << ")";
        return ss.str();
    }

    /// Get flip probability
    float probability() const { return p_; }

    /// Check if last call resulted in a flip
    bool was_flipped() const { return last_flipped_; }

private:
    float p_;
    mutable bool last_flipped_ = false;

    void validate_probability() const {
        if (!std::isfinite(p_)) {
            throw ValidationError("Flip probability must be finite, got NaN or Inf");
        }
        if (p_ < 0.0f || p_ > 1.0f) {
            throw ValidationError(
                "Flip probability must be in [0, 1], got " + std::to_string(p_)
            );
        }
    }
};

}  // namespace pyflame_vision::transforms

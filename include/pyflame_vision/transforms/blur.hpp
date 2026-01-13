#pragma once

/// @file blur.hpp
/// @brief Gaussian blur transform for data augmentation
///
/// Provides GaussianBlur transform equivalent to
/// torchvision.transforms.GaussianBlur.

#include "pyflame_vision/transforms/random_transform.hpp"
#include <cmath>
#include <numeric>
#include <sstream>
#include <limits>

namespace pyflame_vision::transforms {

/// Gaussian blur transform
///
/// Applies Gaussian blur with configurable kernel size and sigma.
/// Equivalent to torchvision.transforms.GaussianBlur
///
/// @note Thread Safety: Thread-safe via inherited mutex-protected RNG.
class GaussianBlur : public RandomTransform {
public:
    /// Create Gaussian blur with fixed kernel size
    /// @param kernel_size Blur kernel size (must be positive odd integer)
    /// @param sigma Sigma range for Gaussian kernel [min, max]
    explicit GaussianBlur(
        int kernel_size,
        std::pair<float, float> sigma = {0.1f, 2.0f}
    )
        : kernel_size_min_(kernel_size)
        , kernel_size_max_(kernel_size)
        , sigma_(sigma)
    {
        validate_params();
    }

    /// Create Gaussian blur with kernel size range
    /// @param kernel_size_min Minimum kernel size (must be positive odd integer)
    /// @param kernel_size_max Maximum kernel size (must be positive odd integer)
    /// @param sigma Sigma range for Gaussian kernel [min, max]
    GaussianBlur(
        int kernel_size_min,
        int kernel_size_max,
        std::pair<float, float> sigma = {0.1f, 2.0f}
    )
        : kernel_size_min_(kernel_size_min)
        , kernel_size_max_(kernel_size_max)
        , sigma_(sigma)
    {
        validate_params();
    }

    std::vector<int64_t> get_output_shape(
        const std::vector<int64_t>& input_shape
    ) const override {
        validate_input(input_shape);

        // Generate random kernel size (must be odd)
        if (kernel_size_min_ == kernel_size_max_) {
            last_kernel_size_ = kernel_size_min_;
        } else {
            // Generate random odd number in range
            int range = (kernel_size_max_ - kernel_size_min_) / 2 + 1;
            int idx = random_int(0, range - 1);
            last_kernel_size_ = kernel_size_min_ + idx * 2;
        }

        // Generate random sigma
        last_sigma_ = random_uniform(sigma_.first, sigma_.second);

        // Shape unchanged (blur is a same-size convolution)
        return input_shape;
    }

    std::string name() const override { return "GaussianBlur"; }

    std::string repr() const override {
        std::ostringstream ss;
        ss << "GaussianBlur(kernel_size=";
        if (kernel_size_min_ == kernel_size_max_) {
            ss << kernel_size_min_;
        } else {
            ss << "(" << kernel_size_min_ << ", " << kernel_size_max_ << ")";
        }
        ss << ", sigma=(" << sigma_.first << ", " << sigma_.second << "))";
        return ss.str();
    }

    /// Get kernel size range
    std::pair<int, int> kernel_size() const { return {kernel_size_min_, kernel_size_max_}; }

    /// Get sigma range
    std::pair<float, float> sigma() const { return sigma_; }

    /// Get last applied kernel size
    int last_kernel_size() const { return last_kernel_size_; }

    /// Get last applied sigma
    float last_sigma() const { return last_sigma_; }

    /// Get halo size for distributed execution
    int halo_size() const {
        return last_kernel_size_ / 2;
    }

    /// Generate Gaussian kernel weights
    /// @return 1D kernel weights (apply separably for 2D Gaussian blur)
    /// @throws ConfigurationError if get_output_shape has not been called
    std::vector<float> get_kernel_weights() const {
        if (last_kernel_size_ == 0) {
            throw ConfigurationError("GaussianBlur: must call get_output_shape first");
        }
        return compute_kernel(last_kernel_size_, last_sigma_);
    }

private:
    int kernel_size_min_;
    int kernel_size_max_;
    std::pair<float, float> sigma_;

    mutable int last_kernel_size_ = 0;
    mutable float last_sigma_ = 0.0f;

    void validate_params() const {
        core::validate_blur_kernel_size(kernel_size_min_);
        core::validate_blur_kernel_size(kernel_size_max_);

        if (kernel_size_min_ > kernel_size_max_) {
            throw ValidationError(
                "kernel_size_min (" + std::to_string(kernel_size_min_) +
                ") must be <= kernel_size_max (" + std::to_string(kernel_size_max_) + ")"
            );
        }

        core::validate_blur_sigma(sigma_.first);
        core::validate_blur_sigma(sigma_.second);

        if (sigma_.first > sigma_.second) {
            throw ValidationError(
                "sigma_min (" + std::to_string(sigma_.first) +
                ") must be <= sigma_max (" + std::to_string(sigma_.second) + ")"
            );
        }
    }

    /// Compute Gaussian kernel weights for given size and sigma
    /// @throws ValidationError if kernel sum underflows to zero
    static std::vector<float> compute_kernel(int size, float sigma) {
        std::vector<float> kernel(size);
        int half = size / 2;
        float sum = 0.0f;

        // Compute unnormalized Gaussian values
        for (int i = 0; i < size; ++i) {
            float x = static_cast<float>(i - half);
            kernel[i] = std::exp(-x * x / (2.0f * sigma * sigma));
            sum += kernel[i];
        }

        // Check for underflow before normalization to prevent division by zero
        if (sum < std::numeric_limits<float>::min()) {
            throw ValidationError(
                "Gaussian kernel sum underflowed to zero - sigma (" +
                std::to_string(sigma) + ") is too small for kernel size (" +
                std::to_string(size) + ")"
            );
        }

        // Normalize so weights sum to 1
        for (float& k : kernel) {
            k /= sum;
        }

        return kernel;
    }
};

}  // namespace pyflame_vision::transforms

#pragma once

/// @file pooling.hpp
/// @brief Pooling layers
///
/// Provides MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, etc.

#include "pyflame_vision/nn/module.hpp"
#include <array>
#include <sstream>
#include <cmath>

namespace pyflame_vision::nn {

/// 2D Max Pooling
///
/// Applies max pooling over an input tensor.
/// Input shape: [N, C, H, W]
/// Output shape: [N, C, H_out, W_out]
///
/// @note Thread Safety: MaxPool2d is thread-safe (immutable after construction).
class MaxPool2d : public Module {
public:
    /// Full constructor
    /// @param kernel_size Pooling window size (height, width)
    /// @param stride Stride (height, width). If {0,0}, defaults to kernel_size
    /// @param padding Padding (height, width)
    /// @param dilation Dilation (height, width)
    /// @param ceil_mode Use ceil instead of floor for output size
    MaxPool2d(
        std::array<int64_t, 2> kernel_size,
        std::array<int64_t, 2> stride = {0, 0},
        std::array<int64_t, 2> padding = {0, 0},
        std::array<int64_t, 2> dilation = {1, 1},
        bool ceil_mode = false
    ) : kernel_size_(kernel_size)
      , stride_(stride[0] == 0 && stride[1] == 0 ? kernel_size : stride)
      , padding_(padding)
      , dilation_(dilation)
      , ceil_mode_(ceil_mode)
    {
        validate_params();
    }

    /// Square kernel convenience constructor
    MaxPool2d(
        int64_t kernel_size,
        int64_t stride = 0,
        int64_t padding = 0,
        int64_t dilation = 1,
        bool ceil_mode = false
    ) : MaxPool2d(
            {kernel_size, kernel_size},
            {stride == 0 ? kernel_size : stride, stride == 0 ? kernel_size : stride},
            {padding, padding},
            {dilation, dilation},
            ceil_mode
        ) {}

    TensorSpec forward(const TensorSpec& input) const override {
        validate_input_dims(input, 4);

        auto output_shape = compute_output_shape(input.shape);
        return {output_shape, input.dtype};
    }

    std::string name() const override { return "MaxPool2d"; }

    std::string repr() const override {
        std::ostringstream ss;
        ss << "MaxPool2d(kernel_size=(" << kernel_size_[0] << ", " << kernel_size_[1] << ")";
        ss << ", stride=(" << stride_[0] << ", " << stride_[1] << ")";
        if (padding_[0] != 0 || padding_[1] != 0) {
            ss << ", padding=(" << padding_[0] << ", " << padding_[1] << ")";
        }
        if (dilation_[0] != 1 || dilation_[1] != 1) {
            ss << ", dilation=(" << dilation_[0] << ", " << dilation_[1] << ")";
        }
        if (ceil_mode_) {
            ss << ", ceil_mode=True";
        }
        ss << ")";
        return ss.str();
    }

    // Accessors
    const std::array<int64_t, 2>& kernel_size() const { return kernel_size_; }
    const std::array<int64_t, 2>& stride() const { return stride_; }
    const std::array<int64_t, 2>& padding() const { return padding_; }
    const std::array<int64_t, 2>& dilation() const { return dilation_; }
    bool ceil_mode() const { return ceil_mode_; }

    /// Get halo size for distributed execution
    int halo_size() const {
        int64_t h_halo = dilation_[0] * ((kernel_size_[0] - 1) / 2);
        int64_t w_halo = dilation_[1] * ((kernel_size_[1] - 1) / 2);
        return static_cast<int>(std::max(h_halo, w_halo));
    }

private:
    std::array<int64_t, 2> kernel_size_;
    std::array<int64_t, 2> stride_;
    std::array<int64_t, 2> padding_;
    std::array<int64_t, 2> dilation_;
    bool ceil_mode_;

    void validate_params() const {
        if (kernel_size_[0] <= 0 || kernel_size_[1] <= 0) {
            throw ValidationError("MaxPool2d: kernel_size must be positive");
        }
        if (stride_[0] <= 0 || stride_[1] <= 0) {
            throw ValidationError("MaxPool2d: stride must be positive");
        }
        if (padding_[0] < 0 || padding_[1] < 0) {
            throw ValidationError("MaxPool2d: padding must be non-negative");
        }
        if (dilation_[0] <= 0 || dilation_[1] <= 0) {
            throw ValidationError("MaxPool2d: dilation must be positive");
        }
    }

    std::vector<int64_t> compute_output_shape(const std::vector<int64_t>& input_shape) const {
        int64_t batch = input_shape[0];
        int64_t channels = input_shape[1];
        int64_t h_in = input_shape[2];
        int64_t w_in = input_shape[3];

        int64_t h_out, w_out;
        if (ceil_mode_) {
            h_out = static_cast<int64_t>(std::ceil(
                static_cast<double>(h_in + 2 * padding_[0] - dilation_[0] * (kernel_size_[0] - 1) - 1) / stride_[0]
            )) + 1;
            w_out = static_cast<int64_t>(std::ceil(
                static_cast<double>(w_in + 2 * padding_[1] - dilation_[1] * (kernel_size_[1] - 1) - 1) / stride_[1]
            )) + 1;
        } else {
            h_out = (h_in + 2 * padding_[0] - dilation_[0] * (kernel_size_[0] - 1) - 1) / stride_[0] + 1;
            w_out = (w_in + 2 * padding_[1] - dilation_[1] * (kernel_size_[1] - 1) - 1) / stride_[1] + 1;
        }

        if (h_out <= 0 || w_out <= 0) {
            throw ValidationError("MaxPool2d: computed output size is non-positive");
        }

        return {batch, channels, h_out, w_out};
    }
};

/// 2D Average Pooling
///
/// @note Thread Safety: AvgPool2d is thread-safe (immutable after construction).
class AvgPool2d : public Module {
public:
    AvgPool2d(
        std::array<int64_t, 2> kernel_size,
        std::array<int64_t, 2> stride = {0, 0},
        std::array<int64_t, 2> padding = {0, 0},
        bool ceil_mode = false,
        bool count_include_pad = true
    ) : kernel_size_(kernel_size)
      , stride_(stride[0] == 0 && stride[1] == 0 ? kernel_size : stride)
      , padding_(padding)
      , ceil_mode_(ceil_mode)
      , count_include_pad_(count_include_pad)
    {
        validate_params();
    }

    AvgPool2d(
        int64_t kernel_size,
        int64_t stride = 0,
        int64_t padding = 0,
        bool ceil_mode = false,
        bool count_include_pad = true
    ) : AvgPool2d(
            {kernel_size, kernel_size},
            {stride == 0 ? kernel_size : stride, stride == 0 ? kernel_size : stride},
            {padding, padding},
            ceil_mode,
            count_include_pad
        ) {}

    TensorSpec forward(const TensorSpec& input) const override {
        validate_input_dims(input, 4);

        auto output_shape = compute_output_shape(input.shape);
        return {output_shape, input.dtype};
    }

    std::string name() const override { return "AvgPool2d"; }

    std::string repr() const override {
        std::ostringstream ss;
        ss << "AvgPool2d(kernel_size=(" << kernel_size_[0] << ", " << kernel_size_[1] << ")";
        ss << ", stride=(" << stride_[0] << ", " << stride_[1] << ")";
        if (padding_[0] != 0 || padding_[1] != 0) {
            ss << ", padding=(" << padding_[0] << ", " << padding_[1] << ")";
        }
        if (ceil_mode_) {
            ss << ", ceil_mode=True";
        }
        if (!count_include_pad_) {
            ss << ", count_include_pad=False";
        }
        ss << ")";
        return ss.str();
    }

    const std::array<int64_t, 2>& kernel_size() const { return kernel_size_; }
    const std::array<int64_t, 2>& stride() const { return stride_; }
    const std::array<int64_t, 2>& padding() const { return padding_; }
    bool ceil_mode() const { return ceil_mode_; }
    bool count_include_pad() const { return count_include_pad_; }

private:
    std::array<int64_t, 2> kernel_size_;
    std::array<int64_t, 2> stride_;
    std::array<int64_t, 2> padding_;
    bool ceil_mode_;
    bool count_include_pad_;

    void validate_params() const {
        if (kernel_size_[0] <= 0 || kernel_size_[1] <= 0) {
            throw ValidationError("AvgPool2d: kernel_size must be positive");
        }
        if (stride_[0] <= 0 || stride_[1] <= 0) {
            throw ValidationError("AvgPool2d: stride must be positive");
        }
        if (padding_[0] < 0 || padding_[1] < 0) {
            throw ValidationError("AvgPool2d: padding must be non-negative");
        }
    }

    std::vector<int64_t> compute_output_shape(const std::vector<int64_t>& input_shape) const {
        int64_t batch = input_shape[0];
        int64_t channels = input_shape[1];
        int64_t h_in = input_shape[2];
        int64_t w_in = input_shape[3];

        int64_t h_out, w_out;
        if (ceil_mode_) {
            h_out = static_cast<int64_t>(std::ceil(
                static_cast<double>(h_in + 2 * padding_[0] - kernel_size_[0]) / stride_[0]
            )) + 1;
            w_out = static_cast<int64_t>(std::ceil(
                static_cast<double>(w_in + 2 * padding_[1] - kernel_size_[1]) / stride_[1]
            )) + 1;
        } else {
            h_out = (h_in + 2 * padding_[0] - kernel_size_[0]) / stride_[0] + 1;
            w_out = (w_in + 2 * padding_[1] - kernel_size_[1]) / stride_[1] + 1;
        }

        if (h_out <= 0 || w_out <= 0) {
            throw ValidationError("AvgPool2d: computed output size is non-positive");
        }

        return {batch, channels, h_out, w_out};
    }
};

/// Adaptive Average Pooling - outputs fixed size regardless of input
///
/// Dynamically computes kernel size to achieve target output size.
///
/// @note Thread Safety: AdaptiveAvgPool2d is thread-safe (immutable after construction).
class AdaptiveAvgPool2d : public Module {
public:
    /// @param output_size Target output size (height, width)
    explicit AdaptiveAvgPool2d(std::array<int64_t, 2> output_size)
        : output_size_(output_size)
    {
        validate_params();
    }

    /// Square output convenience constructor
    explicit AdaptiveAvgPool2d(int64_t output_size)
        : AdaptiveAvgPool2d({output_size, output_size}) {}

    TensorSpec forward(const TensorSpec& input) const override {
        validate_input_dims(input, 4);

        int64_t batch = input.shape[0];
        int64_t channels = input.shape[1];

        return {{batch, channels, output_size_[0], output_size_[1]}, input.dtype};
    }

    std::string name() const override { return "AdaptiveAvgPool2d"; }

    std::string repr() const override {
        std::ostringstream ss;
        ss << "AdaptiveAvgPool2d(output_size=(" << output_size_[0] << ", " << output_size_[1] << "))";
        return ss.str();
    }

    const std::array<int64_t, 2>& output_size() const { return output_size_; }

private:
    std::array<int64_t, 2> output_size_;

    void validate_params() const {
        if (output_size_[0] <= 0 || output_size_[1] <= 0) {
            throw ValidationError("AdaptiveAvgPool2d: output_size must be positive");
        }
    }
};

/// Adaptive Max Pooling
class AdaptiveMaxPool2d : public Module {
public:
    explicit AdaptiveMaxPool2d(std::array<int64_t, 2> output_size)
        : output_size_(output_size)
    {
        if (output_size_[0] <= 0 || output_size_[1] <= 0) {
            throw ValidationError("AdaptiveMaxPool2d: output_size must be positive");
        }
    }

    explicit AdaptiveMaxPool2d(int64_t output_size)
        : AdaptiveMaxPool2d({output_size, output_size}) {}

    TensorSpec forward(const TensorSpec& input) const override {
        validate_input_dims(input, 4);
        return {{input.shape[0], input.shape[1], output_size_[0], output_size_[1]}, input.dtype};
    }

    std::string name() const override { return "AdaptiveMaxPool2d"; }

    std::string repr() const override {
        std::ostringstream ss;
        ss << "AdaptiveMaxPool2d(output_size=(" << output_size_[0] << ", " << output_size_[1] << "))";
        return ss.str();
    }

    const std::array<int64_t, 2>& output_size() const { return output_size_; }

private:
    std::array<int64_t, 2> output_size_;
};

/// Global Average Pooling (AdaptiveAvgPool2d with output_size=1)
class GlobalAvgPool2d : public Module {
public:
    TensorSpec forward(const TensorSpec& input) const override {
        validate_input_dims(input, 4);
        return {{input.shape[0], input.shape[1], 1, 1}, input.dtype};
    }

    std::string name() const override { return "GlobalAvgPool2d"; }
    std::string repr() const override { return "GlobalAvgPool2d()"; }
};

}  // namespace pyflame_vision::nn

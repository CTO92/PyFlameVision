#pragma once

/// @file conv.hpp
/// @brief Convolution layers
///
/// Provides Conv2d and related convolution modules.

#include "pyflame_vision/nn/module.hpp"
#include <array>
#include <sstream>

namespace pyflame_vision::nn {

/// Padding mode for convolution
enum class PaddingMode : uint8_t {
    ZEROS = 0,      ///< Zero padding (default)
    REFLECT = 1,    ///< Reflect padding
    REPLICATE = 2,  ///< Replicate border
    CIRCULAR = 3    ///< Circular padding
};

/// Get string representation of padding mode
inline std::string padding_mode_name(PaddingMode mode) {
    switch (mode) {
        case PaddingMode::ZEROS: return "zeros";
        case PaddingMode::REFLECT: return "reflect";
        case PaddingMode::REPLICATE: return "replicate";
        case PaddingMode::CIRCULAR: return "circular";
        default: return "unknown";
    }
}

/// 2D Convolution layer
///
/// Applies a 2D convolution over an input tensor.
/// Input shape: [N, C_in, H, W]
/// Output shape: [N, C_out, H_out, W_out]
///
/// where:
///   H_out = floor((H + 2*padding - dilation*(kernel-1) - 1) / stride + 1)
///   W_out = floor((W + 2*padding - dilation*(kernel-1) - 1) / stride + 1)
///
/// @note Thread Safety: Conv2d is thread-safe (immutable after construction).
class Conv2d : public Module {
public:
    /// Full constructor with array parameters
    /// @param in_channels Number of input channels
    /// @param out_channels Number of output channels
    /// @param kernel_size Kernel size (height, width)
    /// @param stride Stride (height, width)
    /// @param padding Padding (height, width)
    /// @param dilation Dilation (height, width)
    /// @param groups Number of groups for grouped convolution
    /// @param bias Whether to include bias term
    /// @param padding_mode Padding mode
    Conv2d(
        int64_t in_channels,
        int64_t out_channels,
        std::array<int64_t, 2> kernel_size,
        std::array<int64_t, 2> stride = {1, 1},
        std::array<int64_t, 2> padding = {0, 0},
        std::array<int64_t, 2> dilation = {1, 1},
        int64_t groups = 1,
        bool bias = true,
        PaddingMode padding_mode = PaddingMode::ZEROS
    ) : in_channels_(in_channels)
      , out_channels_(out_channels)
      , kernel_size_(kernel_size)
      , stride_(stride)
      , padding_(padding)
      , dilation_(dilation)
      , groups_(groups)
      , bias_(bias)
      , padding_mode_(padding_mode)
    {
        validate_params();
    }

    /// Square kernel convenience constructor
    Conv2d(
        int64_t in_channels,
        int64_t out_channels,
        int64_t kernel_size,
        int64_t stride = 1,
        int64_t padding = 0,
        int64_t dilation = 1,
        int64_t groups = 1,
        bool bias = true,
        PaddingMode padding_mode = PaddingMode::ZEROS
    ) : Conv2d(
            in_channels,
            out_channels,
            {kernel_size, kernel_size},
            {stride, stride},
            {padding, padding},
            {dilation, dilation},
            groups,
            bias,
            padding_mode
        ) {}

    TensorSpec forward(const TensorSpec& input) const override {
        validate_input_dims(input, 4);
        validate_input_channels(input, in_channels_);

        auto output_shape = compute_output_shape(input.shape);
        return {output_shape, input.dtype};
    }

    std::string name() const override { return "Conv2d"; }

    std::string repr() const override {
        std::ostringstream ss;
        ss << "Conv2d(" << in_channels_ << ", " << out_channels_;
        ss << ", kernel_size=(" << kernel_size_[0] << ", " << kernel_size_[1] << ")";
        if (stride_[0] != 1 || stride_[1] != 1) {
            ss << ", stride=(" << stride_[0] << ", " << stride_[1] << ")";
        }
        if (padding_[0] != 0 || padding_[1] != 0) {
            ss << ", padding=(" << padding_[0] << ", " << padding_[1] << ")";
        }
        if (dilation_[0] != 1 || dilation_[1] != 1) {
            ss << ", dilation=(" << dilation_[0] << ", " << dilation_[1] << ")";
        }
        if (groups_ != 1) {
            ss << ", groups=" << groups_;
        }
        if (!bias_) {
            ss << ", bias=False";
        }
        if (padding_mode_ != PaddingMode::ZEROS) {
            ss << ", padding_mode='" << padding_mode_name(padding_mode_) << "'";
        }
        ss << ")";
        return ss.str();
    }

    std::vector<Parameter> parameters() const override {
        std::vector<Parameter> params;

        // Weight shape: [out_channels, in_channels/groups, kernel_h, kernel_w]
        params.push_back({
            "weight",
            {{out_channels_, in_channels_ / groups_, kernel_size_[0], kernel_size_[1]}, "float32"},
            true
        });

        if (bias_) {
            params.push_back({
                "bias",
                {{out_channels_}, "float32"},
                true
            });
        }

        return params;
    }

    // Accessors
    int64_t in_channels() const { return in_channels_; }
    int64_t out_channels() const { return out_channels_; }
    const std::array<int64_t, 2>& kernel_size() const { return kernel_size_; }
    const std::array<int64_t, 2>& stride() const { return stride_; }
    const std::array<int64_t, 2>& padding() const { return padding_; }
    const std::array<int64_t, 2>& dilation() const { return dilation_; }
    int64_t groups() const { return groups_; }
    bool has_bias() const { return bias_; }
    PaddingMode padding_mode() const { return padding_mode_; }

    /// Get halo size needed for distributed execution
    int halo_size() const {
        // Maximum halo needed based on kernel and dilation
        int64_t h_halo = dilation_[0] * ((kernel_size_[0] - 1) / 2);
        int64_t w_halo = dilation_[1] * ((kernel_size_[1] - 1) / 2);
        return static_cast<int>(std::max(h_halo, w_halo));
    }

    /// Check if this is a depthwise convolution
    bool is_depthwise() const {
        return groups_ == in_channels_ && groups_ == out_channels_;
    }

    /// Check if this is a pointwise (1x1) convolution
    bool is_pointwise() const {
        return kernel_size_[0] == 1 && kernel_size_[1] == 1;
    }

private:
    int64_t in_channels_;
    int64_t out_channels_;
    std::array<int64_t, 2> kernel_size_;
    std::array<int64_t, 2> stride_;
    std::array<int64_t, 2> padding_;
    std::array<int64_t, 2> dilation_;
    int64_t groups_;
    bool bias_;
    PaddingMode padding_mode_;

    void validate_params() const {
        if (in_channels_ <= 0) {
            throw ValidationError("Conv2d: in_channels must be positive, got " +
                                  std::to_string(in_channels_));
        }
        if (out_channels_ <= 0) {
            throw ValidationError("Conv2d: out_channels must be positive, got " +
                                  std::to_string(out_channels_));
        }
        if (kernel_size_[0] <= 0 || kernel_size_[1] <= 0) {
            throw ValidationError("Conv2d: kernel_size must be positive");
        }
        if (stride_[0] <= 0 || stride_[1] <= 0) {
            throw ValidationError("Conv2d: stride must be positive");
        }
        if (padding_[0] < 0 || padding_[1] < 0) {
            throw ValidationError("Conv2d: padding must be non-negative");
        }
        if (dilation_[0] <= 0 || dilation_[1] <= 0) {
            throw ValidationError("Conv2d: dilation must be positive");
        }
        if (groups_ <= 0) {
            throw ValidationError("Conv2d: groups must be positive, got " +
                                  std::to_string(groups_));
        }
        if (in_channels_ % groups_ != 0) {
            throw ValidationError(
                "Conv2d: in_channels (" + std::to_string(in_channels_) +
                ") must be divisible by groups (" + std::to_string(groups_) + ")"
            );
        }
        if (out_channels_ % groups_ != 0) {
            throw ValidationError(
                "Conv2d: out_channels (" + std::to_string(out_channels_) +
                ") must be divisible by groups (" + std::to_string(groups_) + ")"
            );
        }

        // Check against security limits
        if (in_channels_ > core::SecurityLimits::MAX_CHANNELS) {
            throw ResourceError("Conv2d: in_channels exceeds maximum");
        }
        if (out_channels_ > core::SecurityLimits::MAX_CHANNELS) {
            throw ResourceError("Conv2d: out_channels exceeds maximum");
        }
    }

    std::vector<int64_t> compute_output_shape(const std::vector<int64_t>& input_shape) const {
        int64_t batch = input_shape[0];
        int64_t h_in = input_shape[2];
        int64_t w_in = input_shape[3];

        // Output size formula:
        // H_out = floor((H_in + 2*padding - dilation*(kernel-1) - 1) / stride + 1)
        int64_t h_out = (h_in + 2 * padding_[0] - dilation_[0] * (kernel_size_[0] - 1) - 1) / stride_[0] + 1;
        int64_t w_out = (w_in + 2 * padding_[1] - dilation_[1] * (kernel_size_[1] - 1) - 1) / stride_[1] + 1;

        if (h_out <= 0 || w_out <= 0) {
            throw ValidationError(
                "Conv2d: computed output size is non-positive. "
                "Input: " + std::to_string(h_in) + "x" + std::to_string(w_in) +
                ", kernel: " + std::to_string(kernel_size_[0]) + "x" + std::to_string(kernel_size_[1]) +
                ", padding: " + std::to_string(padding_[0]) + "x" + std::to_string(padding_[1]) +
                ", stride: " + std::to_string(stride_[0]) + "x" + std::to_string(stride_[1])
            );
        }

        return {batch, out_channels_, h_out, w_out};
    }
};

}  // namespace pyflame_vision::nn

#pragma once

/// @file batchnorm.hpp
/// @brief Batch Normalization layers
///
/// Provides BatchNorm2d and related normalization modules.

#include "pyflame_vision/nn/module.hpp"
#include <sstream>

namespace pyflame_vision::nn {

/// 2D Batch Normalization
///
/// Normalizes each channel across batch and spatial dimensions:
///   output = (input - running_mean) / sqrt(running_var + eps) * weight + bias
///
/// During training, running_mean and running_var are updated.
/// During evaluation, the stored running statistics are used.
///
/// @note Thread Safety: BatchNorm2d is thread-safe in eval mode (immutable).
///       In training mode, running statistics would be updated (not supported
///       in lazy evaluation mode).
class BatchNorm2d : public Module {
public:
    /// Create BatchNorm2d layer
    /// @param num_features Number of input channels (C in NCHW)
    /// @param eps Small constant for numerical stability
    /// @param momentum Momentum for running statistics update
    /// @param affine Whether to include learnable affine parameters (weight, bias)
    /// @param track_running_stats Whether to track running mean/var
    BatchNorm2d(
        int64_t num_features,
        double eps = 1e-5,
        double momentum = 0.1,
        bool affine = true,
        bool track_running_stats = true
    ) : num_features_(num_features)
      , eps_(eps)
      , momentum_(momentum)
      , affine_(affine)
      , track_running_stats_(track_running_stats)
    {
        validate_params();
    }

    TensorSpec forward(const TensorSpec& input) const override {
        validate_input_dims(input, 4);
        validate_input_channels(input, num_features_);
        return input;  // BatchNorm preserves shape
    }

    std::string name() const override { return "BatchNorm2d"; }

    std::string repr() const override {
        std::ostringstream ss;
        ss << "BatchNorm2d(" << num_features_;
        ss << ", eps=" << eps_;
        ss << ", momentum=" << momentum_;
        if (!affine_) {
            ss << ", affine=False";
        }
        if (!track_running_stats_) {
            ss << ", track_running_stats=False";
        }
        ss << ")";
        return ss.str();
    }

    std::vector<Parameter> parameters() const override {
        std::vector<Parameter> params;

        if (affine_) {
            // Learnable scale (gamma)
            params.push_back({
                "weight",
                {{num_features_}, "float32"},
                true
            });
            // Learnable shift (beta)
            params.push_back({
                "bias",
                {{num_features_}, "float32"},
                true
            });
        }

        if (track_running_stats_) {
            // Running statistics (not learnable)
            params.push_back({
                "running_mean",
                {{num_features_}, "float32"},
                false
            });
            params.push_back({
                "running_var",
                {{num_features_}, "float32"},
                false
            });
            params.push_back({
                "num_batches_tracked",
                {{1}, "int64"},
                false
            });
        }

        return params;
    }

    // Accessors
    int64_t num_features() const { return num_features_; }
    double eps() const { return eps_; }
    double momentum() const { return momentum_; }
    bool affine() const { return affine_; }
    bool track_running_stats() const { return track_running_stats_; }

private:
    int64_t num_features_;
    double eps_;
    double momentum_;
    bool affine_;
    bool track_running_stats_;

    void validate_params() const {
        if (num_features_ <= 0) {
            throw ValidationError("BatchNorm2d: num_features must be positive, got " +
                                  std::to_string(num_features_));
        }
        if (eps_ <= 0) {
            throw ValidationError("BatchNorm2d: eps must be positive");
        }
        if (momentum_ < 0 || momentum_ > 1) {
            throw ValidationError("BatchNorm2d: momentum must be in [0, 1]");
        }
        if (num_features_ > core::SecurityLimits::MAX_CHANNELS) {
            throw ResourceError("BatchNorm2d: num_features exceeds maximum");
        }
    }
};

/// 1D Batch Normalization (for sequence/temporal data)
class BatchNorm1d : public Module {
public:
    BatchNorm1d(
        int64_t num_features,
        double eps = 1e-5,
        double momentum = 0.1,
        bool affine = true,
        bool track_running_stats = true
    ) : num_features_(num_features)
      , eps_(eps)
      , momentum_(momentum)
      , affine_(affine)
      , track_running_stats_(track_running_stats)
    {
        if (num_features_ <= 0) {
            throw ValidationError("BatchNorm1d: num_features must be positive");
        }
    }

    TensorSpec forward(const TensorSpec& input) const override {
        // Accept both 2D [N, C] and 3D [N, C, L] input
        if (input.shape.size() != 2 && input.shape.size() != 3) {
            throw ValidationError("BatchNorm1d: expected 2D or 3D input");
        }
        if (input.shape[1] != num_features_) {
            throw ValidationError(
                "BatchNorm1d: expected " + std::to_string(num_features_) +
                " features, got " + std::to_string(input.shape[1])
            );
        }
        return input;
    }

    std::string name() const override { return "BatchNorm1d"; }

    std::string repr() const override {
        std::ostringstream ss;
        ss << "BatchNorm1d(" << num_features_;
        ss << ", eps=" << eps_ << ", momentum=" << momentum_;
        if (!affine_) ss << ", affine=False";
        ss << ")";
        return ss.str();
    }

    std::vector<Parameter> parameters() const override {
        std::vector<Parameter> params;
        if (affine_) {
            params.push_back({"weight", {{num_features_}, "float32"}, true});
            params.push_back({"bias", {{num_features_}, "float32"}, true});
        }
        if (track_running_stats_) {
            params.push_back({"running_mean", {{num_features_}, "float32"}, false});
            params.push_back({"running_var", {{num_features_}, "float32"}, false});
        }
        return params;
    }

    int64_t num_features() const { return num_features_; }

private:
    int64_t num_features_;
    double eps_;
    double momentum_;
    bool affine_;
    bool track_running_stats_;
};

/// Layer Normalization
///
/// Normalizes across the last dimensions (channels for images).
class LayerNorm : public Module {
public:
    /// @param normalized_shape Shape of the dimensions to normalize over
    /// @param eps Small constant for numerical stability
    /// @param elementwise_affine Whether to include learnable affine parameters
    LayerNorm(
        std::vector<int64_t> normalized_shape,
        double eps = 1e-5,
        bool elementwise_affine = true
    ) : normalized_shape_(std::move(normalized_shape))
      , eps_(eps)
      , elementwise_affine_(elementwise_affine)
    {
        if (normalized_shape_.empty()) {
            throw ValidationError("LayerNorm: normalized_shape cannot be empty");
        }
    }

    TensorSpec forward(const TensorSpec& input) const override {
        // Check that input ends with normalized_shape
        if (input.shape.size() < normalized_shape_.size()) {
            throw ValidationError("LayerNorm: input has fewer dimensions than normalized_shape");
        }
        size_t start = input.shape.size() - normalized_shape_.size();
        for (size_t i = 0; i < normalized_shape_.size(); ++i) {
            if (input.shape[start + i] != normalized_shape_[i]) {
                throw ValidationError("LayerNorm: input shape doesn't match normalized_shape");
            }
        }
        return input;
    }

    std::string name() const override { return "LayerNorm"; }

    std::string repr() const override {
        std::ostringstream ss;
        ss << "LayerNorm([";
        for (size_t i = 0; i < normalized_shape_.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << normalized_shape_[i];
        }
        ss << "], eps=" << eps_;
        if (!elementwise_affine_) ss << ", elementwise_affine=False";
        ss << ")";
        return ss.str();
    }

    std::vector<Parameter> parameters() const override {
        std::vector<Parameter> params;
        if (elementwise_affine_) {
            params.push_back({"weight", {normalized_shape_, "float32"}, true});
            params.push_back({"bias", {normalized_shape_, "float32"}, true});
        }
        return params;
    }

    const std::vector<int64_t>& normalized_shape() const { return normalized_shape_; }

private:
    std::vector<int64_t> normalized_shape_;
    double eps_;
    bool elementwise_affine_;
};

}  // namespace pyflame_vision::nn

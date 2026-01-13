#pragma once

/// @file linear.hpp
/// @brief Linear (fully connected) layer
///
/// Provides Linear and related dense layers.

#include "pyflame_vision/nn/module.hpp"
#include "pyflame_vision/core/security.hpp"
#include <sstream>

namespace pyflame_vision::nn {

/// Linear (fully connected) layer
///
/// Applies a linear transformation: y = x @ W^T + b
///
/// Input shape: [*, in_features]
/// Output shape: [*, out_features]
///
/// @note Thread Safety: Linear is thread-safe (immutable after construction).
class Linear : public Module {
public:
    /// Create Linear layer
    /// @param in_features Size of each input sample
    /// @param out_features Size of each output sample
    /// @param bias If True, adds a learnable bias to the output
    Linear(int64_t in_features, int64_t out_features, bool bias = true)
        : in_features_(in_features)
        , out_features_(out_features)
        , bias_(bias)
    {
        validate_params();
    }

    TensorSpec forward(const TensorSpec& input) const override {
        if (input.shape.empty()) {
            throw ValidationError("Linear: input cannot be 0-dimensional");
        }

        // Check last dimension matches in_features
        if (input.shape.back() != in_features_) {
            throw ValidationError(
                "Linear: expected input with last dimension " + std::to_string(in_features_) +
                ", got " + std::to_string(input.shape.back())
            );
        }

        // Output shape: same as input except last dimension becomes out_features
        std::vector<int64_t> output_shape = input.shape;
        output_shape.back() = out_features_;

        return {output_shape, input.dtype};
    }

    std::string name() const override { return "Linear"; }

    std::string repr() const override {
        std::ostringstream ss;
        ss << "Linear(in_features=" << in_features_;
        ss << ", out_features=" << out_features_;
        if (!bias_) {
            ss << ", bias=False";
        }
        ss << ")";
        return ss.str();
    }

    std::vector<Parameter> parameters() const override {
        std::vector<Parameter> params;

        // Weight shape: [out_features, in_features]
        params.push_back({
            "weight",
            {{out_features_, in_features_}, "float32"},
            true
        });

        if (bias_) {
            params.push_back({
                "bias",
                {{out_features_}, "float32"},
                true
            });
        }

        return params;
    }

    // Accessors
    int64_t in_features() const { return in_features_; }
    int64_t out_features() const { return out_features_; }
    bool has_bias() const { return bias_; }

private:
    int64_t in_features_;
    int64_t out_features_;
    bool bias_;

    void validate_params() const {
        if (in_features_ <= 0) {
            throw ValidationError("Linear: in_features must be positive, got " +
                                  std::to_string(in_features_));
        }
        if (out_features_ <= 0) {
            throw ValidationError("Linear: out_features must be positive, got " +
                                  std::to_string(out_features_));
        }
        // Security: Enforce maximum feature limits to prevent memory exhaustion
        if (in_features_ > core::SecurityLimits::MAX_FEATURES) {
            throw ResourceError("Linear: in_features (" + std::to_string(in_features_) +
                               ") exceeds maximum allowed (" +
                               std::to_string(core::SecurityLimits::MAX_FEATURES) + ")");
        }
        if (out_features_ > core::SecurityLimits::MAX_FEATURES) {
            throw ResourceError("Linear: out_features (" + std::to_string(out_features_) +
                               ") exceeds maximum allowed (" +
                               std::to_string(core::SecurityLimits::MAX_FEATURES) + ")");
        }
    }
};

}  // namespace pyflame_vision::nn

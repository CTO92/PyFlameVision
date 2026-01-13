#pragma once

/// @file activation.hpp
/// @brief Activation function modules
///
/// Provides common activation functions as Module classes.

#include "pyflame_vision/nn/module.hpp"

namespace pyflame_vision::nn {

/// ReLU activation: max(0, x)
///
/// @note Thread Safety: ReLU is thread-safe (immutable after construction).
class ReLU : public Module {
public:
    /// Create ReLU activation
    /// @param inplace If true, modify tensor in-place (hint for execution)
    explicit ReLU(bool inplace = false) : inplace_(inplace) {}

    TensorSpec forward(const TensorSpec& input) const override {
        validate_positive_dims(input);
        return input;  // Shape unchanged
    }

    std::string name() const override { return "ReLU"; }

    std::string repr() const override {
        return inplace_ ? "ReLU(inplace=True)" : "ReLU()";
    }

    bool inplace() const { return inplace_; }

private:
    bool inplace_;
};

/// ReLU6 activation: min(max(0, x), 6)
///
/// Used in MobileNet architectures.
class ReLU6 : public Module {
public:
    explicit ReLU6(bool inplace = false) : inplace_(inplace) {}

    TensorSpec forward(const TensorSpec& input) const override {
        validate_positive_dims(input);
        return input;
    }

    std::string name() const override { return "ReLU6"; }

    std::string repr() const override {
        return inplace_ ? "ReLU6(inplace=True)" : "ReLU6()";
    }

    bool inplace() const { return inplace_; }

private:
    bool inplace_;
};

/// SiLU (Swish) activation: x * sigmoid(x)
///
/// Used in EfficientNet and modern architectures.
/// @note Thread Safety: SiLU is thread-safe (immutable after construction).
class SiLU : public Module {
public:
    explicit SiLU(bool inplace = false) : inplace_(inplace) {}

    TensorSpec forward(const TensorSpec& input) const override {
        validate_positive_dims(input);
        return input;  // Shape unchanged
    }

    std::string name() const override { return "SiLU"; }

    std::string repr() const override {
        return inplace_ ? "SiLU(inplace=True)" : "SiLU()";
    }

    bool inplace() const { return inplace_; }

private:
    bool inplace_;
};

/// Sigmoid activation: 1 / (1 + exp(-x))
///
/// @note Thread Safety: Sigmoid is thread-safe (stateless).
class Sigmoid : public Module {
public:
    TensorSpec forward(const TensorSpec& input) const override {
        validate_positive_dims(input);
        return input;  // Shape unchanged
    }

    std::string name() const override { return "Sigmoid"; }
    std::string repr() const override { return "Sigmoid()"; }
};

/// Tanh activation
class Tanh : public Module {
public:
    TensorSpec forward(const TensorSpec& input) const override {
        validate_positive_dims(input);
        return input;
    }

    std::string name() const override { return "Tanh"; }
    std::string repr() const override { return "Tanh()"; }
};

/// GELU activation: x * Phi(x) where Phi is CDF of standard normal
///
/// Used in Transformer architectures.
class GELU : public Module {
public:
    /// @param approximate Use tanh approximation
    explicit GELU(bool approximate = false) : approximate_(approximate) {}

    TensorSpec forward(const TensorSpec& input) const override {
        validate_positive_dims(input);
        return input;
    }

    std::string name() const override { return "GELU"; }

    std::string repr() const override {
        return approximate_ ? "GELU(approximate='tanh')" : "GELU()";
    }

    bool approximate() const { return approximate_; }

private:
    bool approximate_;
};

/// Hardswish activation: x * relu6(x + 3) / 6
///
/// Used in MobileNetV3.
class Hardswish : public Module {
public:
    explicit Hardswish(bool inplace = false) : inplace_(inplace) {}

    TensorSpec forward(const TensorSpec& input) const override {
        validate_positive_dims(input);
        return input;
    }

    std::string name() const override { return "Hardswish"; }

    std::string repr() const override {
        return inplace_ ? "Hardswish(inplace=True)" : "Hardswish()";
    }

    bool inplace() const { return inplace_; }

private:
    bool inplace_;
};

/// Hardsigmoid activation: relu6(x + 3) / 6
class Hardsigmoid : public Module {
public:
    explicit Hardsigmoid(bool inplace = false) : inplace_(inplace) {}

    TensorSpec forward(const TensorSpec& input) const override {
        validate_positive_dims(input);
        return input;
    }

    std::string name() const override { return "Hardsigmoid"; }

    std::string repr() const override {
        return inplace_ ? "Hardsigmoid(inplace=True)" : "Hardsigmoid()";
    }

    bool inplace() const { return inplace_; }

private:
    bool inplace_;
};

/// Identity - passes input through unchanged
///
/// Useful as a placeholder or to replace layers.
/// @note Thread Safety: Identity is thread-safe (stateless).
class Identity : public Module {
public:
    TensorSpec forward(const TensorSpec& input) const override {
        return input;  // Pass through unchanged
    }

    std::string name() const override { return "Identity"; }
    std::string repr() const override { return "Identity()"; }
};

/// Flatten - flattens dimensions from start_dim to end_dim
class Flatten : public Module {
public:
    /// @param start_dim First dimension to flatten (default: 1, after batch)
    /// @param end_dim Last dimension to flatten (default: -1, last dim)
    explicit Flatten(int64_t start_dim = 1, int64_t end_dim = -1)
        : start_dim_(start_dim), end_dim_(end_dim) {}

    TensorSpec forward(const TensorSpec& input) const override {
        validate_positive_dims(input);

        int64_t ndim = static_cast<int64_t>(input.shape.size());
        if (ndim == 0) {
            return input;
        }

        // Handle negative indices
        int64_t start = start_dim_ >= 0 ? start_dim_ : ndim + start_dim_;
        int64_t end = end_dim_ >= 0 ? end_dim_ : ndim + end_dim_;

        if (start < 0 || start >= ndim || end < 0 || end >= ndim || start > end) {
            throw ValidationError(
                "Flatten: invalid start_dim=" + std::to_string(start_dim_) +
                " or end_dim=" + std::to_string(end_dim_) +
                " for input with " + std::to_string(ndim) + " dimensions"
            );
        }

        // Compute flattened shape
        std::vector<int64_t> output_shape;

        // Dimensions before start_dim
        for (int64_t i = 0; i < start; ++i) {
            output_shape.push_back(input.shape[i]);
        }

        // Flattened dimension
        int64_t flat_size = 1;
        for (int64_t i = start; i <= end; ++i) {
            flat_size = core::safe_multiply(flat_size, input.shape[i], "Flatten");
        }
        output_shape.push_back(flat_size);

        // Dimensions after end_dim
        for (int64_t i = end + 1; i < ndim; ++i) {
            output_shape.push_back(input.shape[i]);
        }

        return {output_shape, input.dtype};
    }

    std::string name() const override { return "Flatten"; }

    std::string repr() const override {
        std::ostringstream ss;
        ss << "Flatten(start_dim=" << start_dim_ << ", end_dim=" << end_dim_ << ")";
        return ss.str();
    }

    int64_t start_dim() const { return start_dim_; }
    int64_t end_dim() const { return end_dim_; }

private:
    int64_t start_dim_;
    int64_t end_dim_;
};

}  // namespace pyflame_vision::nn

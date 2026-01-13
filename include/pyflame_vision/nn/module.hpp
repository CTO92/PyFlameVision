#pragma once

/// @file module.hpp
/// @brief Base Module class for neural network layers
///
/// This header provides the foundational abstraction for all neural network
/// modules in PyFlameVision. Modules compute output shapes lazily and store
/// parameter metadata for weight loading.
///
/// @note Thread Safety: Module instances are immutable after construction,
///       making them thread-safe for concurrent forward() calls. The training_
///       flag uses std::atomic for thread-safe reads/writes.

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <cstdint>
#include <atomic>
#include "pyflame_vision/core/exceptions.hpp"
#include "pyflame_vision/core/security.hpp"

namespace pyflame_vision::nn {

/// Tensor specification for shape inference
/// @note Thread Safety: TensorSpec is a simple data structure with no mutable state.
struct TensorSpec {
    std::vector<int64_t> shape;
    std::string dtype = "float32";

    /// Default constructor
    TensorSpec() = default;

    /// Constructor with shape and dtype
    TensorSpec(std::vector<int64_t> s, std::string d = "float32")
        : shape(std::move(s)), dtype(std::move(d)) {}

    /// Get total number of elements
    /// @throws OverflowError if multiplication would overflow
    int64_t numel() const {
        if (shape.empty()) return 0;
        int64_t total = 1;
        for (auto d : shape) {
            total = core::safe_multiply(total, d, "TensorSpec::numel");
        }
        return total;
    }

    /// Get size in bytes
    /// @throws OverflowError if calculation would overflow
    size_t size_bytes() const {
        int64_t elem_size = 4;  // float32 default
        if (dtype == "float16" || dtype == "bfloat16") elem_size = 2;
        else if (dtype == "float64") elem_size = 8;
        else if (dtype == "int8" || dtype == "uint8") elem_size = 1;
        else if (dtype == "int16" || dtype == "uint16") elem_size = 2;
        else if (dtype == "int64" || dtype == "uint64") elem_size = 8;

        // Use safe_multiply to check for overflow before computing size
        int64_t total = numel();
        int64_t bytes = core::safe_multiply(total, elem_size, "TensorSpec::size_bytes");
        return static_cast<size_t>(bytes);
    }

    /// Check equality
    bool operator==(const TensorSpec& other) const {
        return shape == other.shape && dtype == other.dtype;
    }

    bool operator!=(const TensorSpec& other) const {
        return !(*this == other);
    }

    /// Get number of dimensions
    size_t ndim() const { return shape.size(); }

    /// Check if empty (no shape)
    bool empty() const { return shape.empty(); }
};

/// Parameter metadata (shape info only, actual data stored externally)
struct Parameter {
    std::string name;
    TensorSpec spec;
    bool requires_grad = true;

    Parameter() = default;
    Parameter(std::string n, TensorSpec s, bool grad = true)
        : name(std::move(n)), spec(std::move(s)), requires_grad(grad) {}
};

/// Base class for all neural network modules
///
/// Modules are the building blocks of neural networks. They:
/// - Compute output shapes given input shapes (lazy evaluation)
/// - Store parameter metadata for weight loading
/// - Support hierarchical composition via children()
///
/// @note Thread Safety: Modules are immutable after construction.
///       Multiple threads can safely call forward() and accessors
///       concurrently on the same instance.
class Module {
public:
    virtual ~Module() = default;

    /// Compute output specification given input
    /// @param input Input tensor specification
    /// @return Output tensor specification
    /// @throws ValidationError if input is incompatible with this module
    virtual TensorSpec forward(const TensorSpec& input) const = 0;

    /// Get module type name (e.g., "Conv2d", "Linear")
    virtual std::string name() const = 0;

    /// Get string representation for debugging
    virtual std::string repr() const = 0;

    /// Get parameters in this module (non-recursive)
    /// Override in modules with learnable parameters
    virtual std::vector<Parameter> parameters() const { return {}; }

    /// Get named parameters with hierarchical names
    /// @param prefix Prefix to prepend to parameter names
    /// @return Map of full parameter names to Parameter objects
    virtual std::map<std::string, Parameter> named_parameters(
        const std::string& prefix = ""
    ) const {
        std::map<std::string, Parameter> result;

        // Add this module's parameters
        auto params = parameters();
        for (const auto& p : params) {
            std::string full_name = prefix.empty() ? p.name : prefix + "." + p.name;
            result[full_name] = p;
        }

        // Recursively collect from children
        auto child_modules = children();
        for (size_t i = 0; i < child_modules.size(); ++i) {
            std::string child_prefix = prefix.empty()
                ? std::to_string(i)
                : prefix + "." + std::to_string(i);
            auto child_params = child_modules[i]->named_parameters(child_prefix);
            result.insert(child_params.begin(), child_params.end());
        }

        return result;
    }

    /// Get child modules
    /// Override in container modules (Sequential, etc.)
    virtual std::vector<std::shared_ptr<Module>> children() const { return {}; }

    /// Check if module has learnable parameters
    bool has_parameters() const { return !parameters().empty(); }

    /// Set training mode (affects dropout, batchnorm behavior)
    /// @note Thread Safety: Uses atomic store for thread-safe modification.
    virtual void train(bool mode = true) {
        training_.store(mode, std::memory_order_release);
    }

    /// Set evaluation mode
    void eval() { train(false); }

    /// Check if in training mode
    /// @note Thread Safety: Uses atomic load for thread-safe access.
    bool is_training() const {
        return training_.load(std::memory_order_acquire);
    }

protected:
    /// Training mode flag (mutable but thread-safe via atomic)
    mutable std::atomic<bool> training_{false};

    /// Validate input has expected number of dimensions
    /// @throws ValidationError if dimensions don't match
    void validate_input_dims(const TensorSpec& input, size_t expected) const {
        if (input.shape.size() != expected) {
            throw ValidationError(
                name() + ": expected " + std::to_string(expected) +
                "D input, got " + std::to_string(input.shape.size()) + "D"
            );
        }
    }

    /// Validate input has expected channels (assumes NCHW format)
    /// @throws ValidationError if channels don't match
    void validate_input_channels(const TensorSpec& input, int64_t expected) const {
        if (input.shape.size() < 2) {
            throw ValidationError(name() + ": input must have at least 2 dimensions");
        }
        if (input.shape[1] != expected) {
            throw ValidationError(
                name() + ": expected " + std::to_string(expected) +
                " input channels, got " + std::to_string(input.shape[1])
            );
        }
    }

    /// Validate input has positive dimensions
    /// @throws ValidationError if any dimension is non-positive
    void validate_positive_dims(const TensorSpec& input) const {
        for (size_t i = 0; i < input.shape.size(); ++i) {
            if (input.shape[i] <= 0) {
                throw ValidationError(
                    name() + ": dimension " + std::to_string(i) +
                    " must be positive, got " + std::to_string(input.shape[i])
                );
            }
        }
    }
};

}  // namespace pyflame_vision::nn

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cstdint>

#include "pyflame_vision/core/image_tensor.hpp"
#include "pyflame_vision/core/exceptions.hpp"
#include "pyflame_vision/core/security.hpp"

namespace pyflame_vision::transforms {

/// Base class for all image transforms
/// Transforms are lazy - they build computation graph nodes rather than
/// executing immediately. Computation happens when .eval() is called.
class Transform {
public:
    virtual ~Transform() = default;

    /// Apply transform to input tensor (lazy - creates graph nodes)
    /// @param input Input image tensor in NCHW format
    /// @return Output tensor (lazy, shares graph with input)
    virtual std::vector<int64_t> get_output_shape(
        const std::vector<int64_t>& input_shape
    ) const = 0;

    /// Get human-readable name of the transform
    virtual std::string name() const = 0;

    /// Get string representation (for debugging/display)
    virtual std::string repr() const = 0;

    /// Check if transform is deterministic
    /// Non-deterministic transforms (RandomCrop, etc.) may produce different
    /// results each time they're applied
    virtual bool is_deterministic() const { return true; }

protected:
    /// Validate that input shape is a valid image
    void validate_input(const std::vector<int64_t>& shape) const {
        core::ImageTensor::validate_shape(shape);
    }
};

/// Size specification for transforms
/// Can be initialized with a single int (square) or two ints (height, width)
/// @note Thread Safety: Size is immutable after construction, making it thread-safe.
struct Size {
    int64_t height;
    int64_t width;

    /// Construct square size
    /// @throws ValidationError if size is not positive
    /// @throws ResourceError if size exceeds maximum dimension limit
    explicit Size(int64_t size) : height(size), width(size) {
        validate();
    }

    /// Construct rectangular size
    /// @throws ValidationError if either dimension is not positive
    /// @throws ResourceError if any dimension exceeds maximum limit
    Size(int64_t h, int64_t w) : height(h), width(w) {
        validate();
    }

    /// Check if this is a valid size (always true after construction)
    bool is_valid() const { return height > 0 && width > 0; }

    /// String representation
    std::string to_string() const {
        if (height == width) {
            return std::to_string(height);
        }
        return "(" + std::to_string(height) + ", " + std::to_string(width) + ")";
    }

private:
    /// Validate size parameters
    void validate() const {
        if (height <= 0) {
            throw ValidationError(
                "Size height must be positive, got " + std::to_string(height)
            );
        }
        if (width <= 0) {
            throw ValidationError(
                "Size width must be positive, got " + std::to_string(width)
            );
        }
        if (height > core::SecurityLimits::MAX_DIMENSION) {
            throw ResourceError(
                "Size height (" + std::to_string(height) +
                ") exceeds maximum allowed dimension (" +
                std::to_string(core::SecurityLimits::MAX_DIMENSION) + ")"
            );
        }
        if (width > core::SecurityLimits::MAX_DIMENSION) {
            throw ResourceError(
                "Size width (" + std::to_string(width) +
                ") exceeds maximum allowed dimension (" +
                std::to_string(core::SecurityLimits::MAX_DIMENSION) + ")"
            );
        }
    }
};

/// Base class for transforms that take a size parameter
/// @note Thread Safety: SizeTransform is immutable after construction (size_ is const),
///       making it thread-safe for concurrent read access.
class SizeTransform : public Transform {
public:
    /// Get the target size
    const Size& size() const { return size_; }

protected:
    /// Construct with validated size
    /// @throws ValidationError if size is invalid (handled by Size constructor)
    /// @throws ResourceError if size exceeds limits (handled by Size constructor)
    explicit SizeTransform(Size size) : size_(std::move(size)) {
        // Size constructor already validates, but double-check for safety
        if (!size_.is_valid()) {
            throw ValidationError("Invalid size: " + size_.to_string());
        }
    }

    Size size_;
};

}  // namespace pyflame_vision::transforms

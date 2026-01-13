#pragma once

#include "pyflame_vision/transforms/transform_base.hpp"
#include <vector>
#include <memory>

namespace pyflame_vision::transforms {

/// Compose transform
/// Composes multiple transforms into a sequential pipeline.
/// Equivalent to torchvision.transforms.Compose
///
/// Example usage:
///   auto pipeline = Compose({
///       std::make_shared<Resize>(Size(256)),
///       std::make_shared<CenterCrop>(Size(224)),
///       std::make_shared<Normalize>(mean, std)
///   });
///
/// @note Thread Safety: This class itself is thread-safe for concurrent reads
///       (get_output_shape, accessors, iteration). However, thread safety of
///       the composed pipeline depends on the individual transforms it contains.
///       If any contained transform is not thread-safe (e.g., RandomCrop),
///       then calling get_output_shape() concurrently on this Compose instance
///       is also not thread-safe. For fully thread-safe pipelines, ensure all
///       contained transforms are thread-safe or use external synchronization.
class Compose : public Transform {
public:
    /// Create compose from list of transforms
    /// @param transforms Vector of transforms to compose
    explicit Compose(std::vector<std::shared_ptr<Transform>> transforms);

    /// Get the output shape after applying all transforms
    std::vector<int64_t> get_output_shape(
        const std::vector<int64_t>& input_shape
    ) const override;

    std::string name() const override { return "Compose"; }
    std::string repr() const override;

    /// Check if any transform is non-deterministic
    bool is_deterministic() const override;

    /// Get number of transforms
    size_t size() const { return transforms_.size(); }

    /// Check if empty
    bool empty() const { return transforms_.empty(); }

    /// Get transform at index
    std::shared_ptr<Transform> get(size_t index) const;

    /// Get transform at index (operator[])
    std::shared_ptr<Transform> operator[](size_t index) const {
        return get(index);
    }

    /// Get all transforms
    const std::vector<std::shared_ptr<Transform>>& transforms() const {
        return transforms_;
    }

    /// Iterator support
    auto begin() const { return transforms_.begin(); }
    auto end() const { return transforms_.end(); }

private:
    std::vector<std::shared_ptr<Transform>> transforms_;
};

}  // namespace pyflame_vision::transforms

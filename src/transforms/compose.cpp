#include "pyflame_vision/transforms/compose.hpp"
#include <sstream>

namespace pyflame_vision::transforms {

Compose::Compose(std::vector<std::shared_ptr<Transform>> transforms)
    : transforms_(std::move(transforms))
{
    // Validate no null transforms
    for (size_t i = 0; i < transforms_.size(); ++i) {
        if (!transforms_[i]) {
            throw std::runtime_error(
                "Compose: transform at index " + std::to_string(i) + " is null"
            );
        }
    }
}

std::vector<int64_t> Compose::get_output_shape(
    const std::vector<int64_t>& input_shape
) const {
    if (transforms_.empty()) {
        // Empty compose is identity
        return input_shape;
    }

    // Apply transforms sequentially to compute final shape
    std::vector<int64_t> current_shape = input_shape;
    for (const auto& transform : transforms_) {
        current_shape = transform->get_output_shape(current_shape);
    }
    return current_shape;
}

std::string Compose::repr() const {
    std::ostringstream ss;
    ss << "Compose([\n";
    for (size_t i = 0; i < transforms_.size(); ++i) {
        ss << "    " << transforms_[i]->repr();
        if (i < transforms_.size() - 1) {
            ss << ",";
        }
        ss << "\n";
    }
    ss << "])";
    return ss.str();
}

bool Compose::is_deterministic() const {
    for (const auto& transform : transforms_) {
        if (!transform->is_deterministic()) {
            return false;
        }
    }
    return true;
}

std::shared_ptr<Transform> Compose::get(size_t index) const {
    if (index >= transforms_.size()) {
        throw std::out_of_range(
            "Compose index " + std::to_string(index) +
            " out of range [0, " + std::to_string(transforms_.size()) + ")"
        );
    }
    return transforms_[index];
}

}  // namespace pyflame_vision::transforms

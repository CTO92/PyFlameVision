#include "pyflame_vision/transforms/crop.hpp"
#include "pyflame_vision/core/security.hpp"
#include "pyflame_vision/core/exceptions.hpp"
#include <sstream>
#include <algorithm>

namespace pyflame_vision::transforms {

// ============================================================================
// CenterCrop
// ============================================================================

CenterCrop::CenterCrop(Size size) : SizeTransform(size) {}

std::vector<int64_t> CenterCrop::get_output_shape(
    const std::vector<int64_t>& input_shape
) const {
    validate_input(input_shape);

    int64_t in_height = core::ImageTensor::height(input_shape);
    int64_t in_width = core::ImageTensor::width(input_shape);

    validate_crop_size(in_height, in_width);

    return core::ImageTensor::crop_output_shape(
        input_shape,
        size_.height,
        size_.width
    );
}

std::tuple<int64_t, int64_t, int64_t, int64_t> CenterCrop::compute_bounds(
    int64_t input_height,
    int64_t input_width
) const {
    validate_crop_size(input_height, input_width);

    int64_t top = (input_height - size_.height) / 2;
    int64_t left = (input_width - size_.width) / 2;

    return {top, left, size_.height, size_.width};
}

void CenterCrop::validate_crop_size(int64_t input_height, int64_t input_width) const {
    if (size_.height > input_height || size_.width > input_width) {
        std::ostringstream ss;
        ss << "Crop size (" << size_.height << ", " << size_.width << ") "
           << "larger than image size (" << input_height << ", " << input_width << ")";
        throw BoundsError(ss.str());
    }
}

std::string CenterCrop::repr() const {
    std::ostringstream ss;
    ss << "CenterCrop(size=" << size_.to_string() << ")";
    return ss.str();
}

// ============================================================================
// RandomCrop
// ============================================================================

RandomCrop::RandomCrop(Size size, int padding, bool pad_if_needed, float fill)
    : SizeTransform(size)
    , padding_(padding)
    , pad_if_needed_(pad_if_needed)
    , fill_(fill)
    , rng_(core::generate_secure_seed())  // Use secure seed generation
{
    // Validate padding using security limits
    core::validate_padding(padding);
}

std::vector<int64_t> RandomCrop::get_output_shape(
    const std::vector<int64_t>& input_shape
) const {
    validate_input(input_shape);

    int64_t in_height = core::ImageTensor::height(input_shape);
    int64_t in_width = core::ImageTensor::width(input_shape);

    auto [padded_h, padded_w] = compute_padded_size(in_height, in_width);

    // Check if crop fits
    if (size_.height > padded_h || size_.width > padded_w) {
        std::ostringstream ss;
        ss << "Crop size (" << size_.height << ", " << size_.width << ") "
           << "larger than padded image size (" << padded_h << ", " << padded_w << ")";
        throw BoundsError(ss.str());
    }

    return core::ImageTensor::crop_output_shape(
        input_shape,
        size_.height,
        size_.width
    );
}

void RandomCrop::set_seed(uint64_t seed) {
    rng_.seed(seed);
    seeded_ = true;
}

std::tuple<int64_t, int64_t> RandomCrop::get_random_params(
    int64_t input_height,
    int64_t input_width
) const {
    auto [padded_h, padded_w] = compute_padded_size(input_height, input_width);

    // Range for random crop position
    int64_t max_top = padded_h - size_.height;
    int64_t max_left = padded_w - size_.width;

    if (max_top < 0 || max_left < 0) {
        throw BoundsError("Crop size larger than padded image");
    }

    // Generate random position
    std::uniform_int_distribution<int64_t> dist_top(0, max_top);
    std::uniform_int_distribution<int64_t> dist_left(0, max_left);

    int64_t top = dist_top(rng_);
    int64_t left = dist_left(rng_);

    return {top, left};
}

std::tuple<int64_t, int64_t> RandomCrop::compute_padded_size(
    int64_t input_height,
    int64_t input_width
) const {
    // Use safe arithmetic to prevent integer overflow
    int64_t double_padding = core::safe_multiply(
        static_cast<int64_t>(2),
        static_cast<int64_t>(padding_),
        "padding calculation"
    );
    int64_t padded_h = core::safe_add(input_height, double_padding, "padded height");
    int64_t padded_w = core::safe_add(input_width, double_padding, "padded width");

    if (pad_if_needed_) {
        padded_h = std::max(padded_h, size_.height);
        padded_w = std::max(padded_w, size_.width);
    }

    return {padded_h, padded_w};
}

std::string RandomCrop::repr() const {
    std::ostringstream ss;
    ss << "RandomCrop(size=" << size_.to_string();
    if (padding_ > 0) {
        ss << ", padding=" << padding_;
    }
    if (pad_if_needed_) {
        ss << ", pad_if_needed=True";
    }
    if (fill_ != 0.0f) {
        ss << ", fill=" << fill_;
    }
    ss << ")";
    return ss.str();
}

}  // namespace pyflame_vision::transforms

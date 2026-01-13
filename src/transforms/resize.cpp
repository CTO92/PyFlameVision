#include "pyflame_vision/transforms/resize.hpp"
#include <sstream>

namespace pyflame_vision::transforms {

Resize::Resize(Size size, core::InterpolationMode interpolation, bool antialias)
    : SizeTransform(size)
    , interpolation_(interpolation)
    , antialias_(antialias)
{
}

std::vector<int64_t> Resize::get_output_shape(
    const std::vector<int64_t>& input_shape
) const {
    validate_input(input_shape);

    return core::ImageTensor::resize_output_shape(
        input_shape,
        size_.height,
        size_.width
    );
}

std::string Resize::repr() const {
    std::ostringstream ss;
    ss << "Resize(size=" << size_.to_string();
    ss << ", interpolation=" << core::interpolation_name(interpolation_);
    if (antialias_) {
        ss << ", antialias=True";
    }
    ss << ")";
    return ss.str();
}

}  // namespace pyflame_vision::transforms

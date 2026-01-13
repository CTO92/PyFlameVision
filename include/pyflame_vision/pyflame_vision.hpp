#pragma once

/// @file pyflame_vision.hpp
/// @brief Main include header for PyFlameVision
///
/// PyFlameVision is a Cerebras-native computer vision library that provides
/// PyFlame-compatible implementations of image processing transforms.
///
/// Example usage:
/// @code
/// #include <pyflame_vision/pyflame_vision.hpp>
///
/// using namespace pyflame_vision::transforms;
///
/// // Create a transform pipeline
/// auto pipeline = Compose({
///     std::make_shared<Resize>(Size(256)),
///     std::make_shared<CenterCrop>(Size(224)),
///     std::make_shared<Normalize>(
///         std::vector<float>{0.485f, 0.456f, 0.406f},
///         std::vector<float>{0.229f, 0.224f, 0.225f}
///     )
/// });
///
/// // Get output shape
/// std::vector<int64_t> input_shape = {1, 3, 512, 512};
/// auto output_shape = pipeline.get_output_shape(input_shape);
/// // output_shape = {1, 3, 224, 224}
/// @endcode

// Version info
#include "pyflame_vision/version.hpp"

// Core utilities
#include "pyflame_vision/core/image_tensor.hpp"
#include "pyflame_vision/core/interpolation.hpp"

// Transform base classes
#include "pyflame_vision/transforms/transform_base.hpp"

// Transforms
#include "pyflame_vision/transforms/resize.hpp"
#include "pyflame_vision/transforms/crop.hpp"
#include "pyflame_vision/transforms/normalize.hpp"
#include "pyflame_vision/transforms/compose.hpp"

// Functional API
#include "pyflame_vision/transforms/functional.hpp"

// Phase 4: Specialized Operations
#include "pyflame_vision/ops/ops.hpp"

namespace pyflame_vision {

/// Convenience namespace alias for transforms
namespace T = transforms;

/// Convenience namespace alias for functional API
namespace F = transforms::functional;

}  // namespace pyflame_vision

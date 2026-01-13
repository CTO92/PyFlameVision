#pragma once

/// @file models.hpp
/// @brief Model Architectures API
///
/// Main header for the PyFlameVision model architectures.
/// Includes ResNet, EfficientNet, and related model families.

#include "pyflame_vision/models/resnet.hpp"
#include "pyflame_vision/models/efficientnet.hpp"

namespace pyflame_vision::models {

// Re-export commonly used types at models namespace level
// All types are already in pyflame_vision::models namespace

}  // namespace pyflame_vision::models

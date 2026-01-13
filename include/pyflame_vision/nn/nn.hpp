#pragma once

/// @file nn.hpp
/// @brief Neural Network Module API
///
/// Main header for the PyFlameVision neural network module.
/// Includes all layer types and containers.

#include "pyflame_vision/nn/module.hpp"
#include "pyflame_vision/nn/container.hpp"
#include "pyflame_vision/nn/activation.hpp"
#include "pyflame_vision/nn/conv.hpp"
#include "pyflame_vision/nn/batchnorm.hpp"
#include "pyflame_vision/nn/pooling.hpp"
#include "pyflame_vision/nn/linear.hpp"

namespace pyflame_vision::nn {

// Re-export commonly used types at nn namespace level
// All types are already in pyflame_vision::nn namespace

}  // namespace pyflame_vision::nn

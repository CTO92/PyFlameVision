#pragma once

/// @file ops.hpp
/// @brief Aggregate header for PyFlameVision specialized operations
///
/// Includes all specialized operations for detection and spatial transformers:
/// - GridSample: Coordinate-based spatial sampling
/// - ROIAlign: Region of interest extraction for detection models
/// - NMS: Non-maximum suppression for detection filtering

#include "pyflame_vision/ops/grid_sample.hpp"
#include "pyflame_vision/ops/roi_align.hpp"
#include "pyflame_vision/ops/nms.hpp"

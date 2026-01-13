#pragma once

/// @file grid_sample.hpp
/// @brief Grid-based spatial sampling operation
///
/// Provides GridSample operation equivalent to torch.nn.functional.grid_sample.
/// Used in spatial transformer networks and other coordinate-based sampling.

#include "pyflame_vision/transforms/transform_base.hpp"
#include "pyflame_vision/core/interpolation.hpp"
#include <sstream>

namespace pyflame_vision::ops {

/// Grid sampling operation
///
/// Samples from input tensor at locations specified by a grid.
/// Grid coordinates are normalized to [-1, 1] where:
/// - (-1, -1) is top-left corner
/// - (1, 1) is bottom-right corner
///
/// Equivalent to torch.nn.functional.grid_sample
///
/// @note Thread Safety: Thread-safe for concurrent calls (immutable after construction).
class GridSample {
public:
    /// Create grid sample operation
    /// @param mode Interpolation mode (bilinear or nearest)
    /// @param padding_mode How to handle out-of-bounds coordinates
    /// @param align_corners If true, -1 and 1 map to corner pixel centers
    explicit GridSample(
        core::InterpolationMode mode = core::InterpolationMode::BILINEAR,
        core::PaddingMode padding_mode = core::PaddingMode::ZEROS,
        bool align_corners = false
    )
        : mode_(mode)
        , padding_mode_(padding_mode)
        , align_corners_(align_corners)
    {
        validate_params();
    }

    /// Compute output shape given input and grid shapes
    /// @param input_shape Input tensor shape [N, C, H_in, W_in]
    /// @param grid_shape Grid tensor shape [N, H_out, W_out, 2]
    /// @return Output shape [N, C, H_out, W_out]
    /// @throws ValidationError if shapes are incompatible
    std::vector<int64_t> get_output_shape(
        const std::vector<int64_t>& input_shape,
        const std::vector<int64_t>& grid_shape
    ) const {
        validate_input_shape(input_shape);
        validate_grid_shape(grid_shape, input_shape[0]);

        // Output shape: [N, C, H_out, W_out]
        return {
            input_shape[0],   // N (batch)
            input_shape[1],   // C (channels)
            grid_shape[1],    // H_out
            grid_shape[2]     // W_out
        };
    }

    std::string name() const { return "GridSample"; }

    std::string repr() const {
        std::ostringstream ss;
        ss << "GridSample(mode=" << core::interpolation_name(mode_)
           << ", padding_mode=" << core::padding_mode_name(padding_mode_)
           << ", align_corners=" << (align_corners_ ? "True" : "False") << ")";
        return ss.str();
    }

    /// Get interpolation mode
    core::InterpolationMode mode() const { return mode_; }

    /// Get padding mode
    core::PaddingMode padding_mode() const { return padding_mode_; }

    /// Get align_corners flag
    bool align_corners() const { return align_corners_; }

    /// Get halo size for distributed execution
    int halo_size() const {
        return core::interpolation_halo_size(mode_);
    }

private:
    core::InterpolationMode mode_;
    core::PaddingMode padding_mode_;
    bool align_corners_;

    void validate_params() const {
        // Grid sample only supports bilinear and nearest
        if (mode_ != core::InterpolationMode::BILINEAR &&
            mode_ != core::InterpolationMode::NEAREST) {
            throw transforms::ValidationError(
                "GridSample only supports bilinear and nearest interpolation"
            );
        }
    }

    void validate_input_shape(const std::vector<int64_t>& shape) const {
        if (shape.size() != 4) {
            throw transforms::ValidationError(
                "GridSample input must be 4D (NCHW), got " +
                std::to_string(shape.size()) + "D"
            );
        }
        for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] <= 0) {
                throw transforms::ValidationError(
                    "GridSample input dimensions must be positive"
                );
            }
        }
        // Check resource limits
        core::validate_dimension(shape[2], "input height");
        core::validate_dimension(shape[3], "input width");
    }

    void validate_grid_shape(
        const std::vector<int64_t>& grid_shape,
        int64_t expected_batch
    ) const {
        if (grid_shape.size() != 4) {
            throw transforms::ValidationError(
                "GridSample grid must be 4D [N, H_out, W_out, 2], got " +
                std::to_string(grid_shape.size()) + "D"
            );
        }
        if (grid_shape[0] != expected_batch) {
            throw transforms::ValidationError(
                "GridSample batch size mismatch: input has " +
                std::to_string(expected_batch) + ", grid has " +
                std::to_string(grid_shape[0])
            );
        }
        if (grid_shape[3] != 2) {
            throw transforms::ValidationError(
                "GridSample grid last dimension must be 2 (x, y), got " +
                std::to_string(grid_shape[3])
            );
        }
        // Check output size limits
        if (grid_shape[1] > core::SecurityLimits::MAX_GRID_SAMPLE_SIZE ||
            grid_shape[2] > core::SecurityLimits::MAX_GRID_SAMPLE_SIZE) {
            throw transforms::ResourceError(
                "GridSample output size exceeds maximum (" +
                std::to_string(core::SecurityLimits::MAX_GRID_SAMPLE_SIZE) + ")"
            );
        }
    }
};

}  // namespace pyflame_vision::ops

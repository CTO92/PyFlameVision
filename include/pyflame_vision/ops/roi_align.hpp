#pragma once

/// @file roi_align.hpp
/// @brief Region of Interest Align operation
///
/// Provides ROIAlign operation equivalent to torchvision.ops.roi_align.
/// Used in detection models like Faster R-CNN and Mask R-CNN.

#include "pyflame_vision/transforms/transform_base.hpp"
#include "pyflame_vision/core/interpolation.hpp"
#include <sstream>
#include <algorithm>

namespace pyflame_vision::ops {

/// ROI specification for ROI Align
/// Format: [batch_index, x1, y1, x2, y2] in input image coordinates
///
/// @note For C++ users: Use create() factory for validated construction,
///       or call validate() after direct construction.
struct ROI {
    int64_t batch_index;
    float x1, y1, x2, y2;

    /// Create a validated ROI (recommended for C++ users)
    /// @throws ValidationError if coordinates are invalid
    static ROI create(
        int64_t batch_index,
        float x1, float y1, float x2, float y2
    ) {
        ROI roi{batch_index, x1, y1, x2, y2};
        roi.validate();
        return roi;
    }

    /// Validate ROI coordinates
    /// @throws ValidationError if coordinates are invalid
    void validate() const {
        if (batch_index < 0) {
            throw transforms::ValidationError(
                "ROI batch_index must be non-negative, got " +
                std::to_string(batch_index)
            );
        }
        core::validate_grid_coordinate(x1, "ROI x1");
        core::validate_grid_coordinate(y1, "ROI y1");
        core::validate_grid_coordinate(x2, "ROI x2");
        core::validate_grid_coordinate(y2, "ROI y2");

        if (x2 < x1 || y2 < y1) {
            throw transforms::ValidationError(
                "ROI coordinates must satisfy x1 <= x2 and y1 <= y2"
            );
        }
    }

    /// Get width
    float width() const { return x2 - x1; }

    /// Get height
    float height() const { return y2 - y1; }

    /// Get area
    float area() const {
        return std::max(0.0f, width()) * std::max(0.0f, height());
    }
};

/// ROI Align operation
///
/// Extracts fixed-size feature maps from regions of interest using
/// bilinear interpolation. Properly handles fractional coordinates
/// unlike ROI Pool.
///
/// Equivalent to torchvision.ops.roi_align
///
/// @note Thread Safety: Thread-safe for concurrent calls (immutable after construction).
class ROIAlign {
public:
    /// Create ROI Align operation
    /// @param output_height Output height
    /// @param output_width Output width
    /// @param spatial_scale Scale factor from input to feature map coordinates
    /// @param sampling_ratio Number of sampling points per bin (0 = adaptive)
    /// @param aligned If true, uses -0.5 offset for pixel centers (recommended)
    ROIAlign(
        int64_t output_height,
        int64_t output_width,
        float spatial_scale,
        int sampling_ratio = 0,
        bool aligned = true
    )
        : output_height_(output_height)
        , output_width_(output_width)
        , spatial_scale_(spatial_scale)
        , sampling_ratio_(sampling_ratio)
        , aligned_(aligned)
    {
        validate_params();
    }

    /// Create ROI Align with square output
    ROIAlign(
        int64_t output_size,
        float spatial_scale,
        int sampling_ratio = 0,
        bool aligned = true
    )
        : ROIAlign(output_size, output_size, spatial_scale, sampling_ratio, aligned)
    {}

    /// Compute output shape given input and number of ROIs
    /// @param input_shape Input feature map shape [N, C, H, W]
    /// @param num_rois Number of ROIs
    /// @return Output shape [num_rois, C, output_height, output_width]
    /// @throws ValidationError if shapes are incompatible
    /// @throws ResourceError if total output elements exceed limits
    std::vector<int64_t> get_output_shape(
        const std::vector<int64_t>& input_shape,
        int64_t num_rois
    ) const {
        validate_input_shape(input_shape);
        core::validate_roi_count(num_rois);

        // Validate total output elements to prevent memory exhaustion
        core::validate_total_elements(
            num_rois,
            input_shape[1],  // channels
            output_height_,
            output_width_
        );

        return {
            num_rois,
            input_shape[1],   // C (channels preserved)
            output_height_,
            output_width_
        };
    }

    std::string name() const { return "ROIAlign"; }

    std::string repr() const {
        std::ostringstream ss;
        ss << "ROIAlign(output_size=(" << output_height_ << ", " << output_width_
           << "), spatial_scale=" << spatial_scale_
           << ", sampling_ratio=" << sampling_ratio_
           << ", aligned=" << (aligned_ ? "True" : "False") << ")";
        return ss.str();
    }

    /// Get output height
    int64_t output_height() const { return output_height_; }

    /// Get output width
    int64_t output_width() const { return output_width_; }

    /// Get spatial scale
    float spatial_scale() const { return spatial_scale_; }

    /// Get sampling ratio
    int sampling_ratio() const { return sampling_ratio_; }

    /// Get aligned flag
    bool aligned() const { return aligned_; }

    /// Get halo size for distributed execution
    /// ROI Align needs halo based on potential ROI extent
    int halo_size() const {
        // Bilinear interpolation needs 1-pixel halo
        return 1;
    }

private:
    int64_t output_height_;
    int64_t output_width_;
    float spatial_scale_;
    int sampling_ratio_;
    bool aligned_;

    void validate_params() const {
        core::validate_roi_align_params(
            output_height_, output_width_, spatial_scale_, sampling_ratio_
        );
    }

    void validate_input_shape(const std::vector<int64_t>& shape) const {
        if (shape.size() != 4) {
            throw transforms::ValidationError(
                "ROIAlign input must be 4D (NCHW), got " +
                std::to_string(shape.size()) + "D"
            );
        }
        for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] <= 0) {
                throw transforms::ValidationError(
                    "ROIAlign input dimensions must be positive"
                );
            }
        }
        core::validate_dimension(shape[2], "input height");
        core::validate_dimension(shape[3], "input width");
    }
};

}  // namespace pyflame_vision::ops

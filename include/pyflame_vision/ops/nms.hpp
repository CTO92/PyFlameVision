#pragma once

/// @file nms.hpp
/// @brief Non-Maximum Suppression operation
///
/// Provides NMS operation equivalent to torchvision.ops.nms.
/// Used to filter overlapping detection boxes.

#include "pyflame_vision/transforms/transform_base.hpp"
#include "pyflame_vision/core/security.hpp"
#include <sstream>
#include <algorithm>
#include <cmath>

namespace pyflame_vision::ops {

/// Detection box structure
/// Format: [x1, y1, x2, y2, score, class_id]
///
/// @note For C++ users: Use create() factory for validated construction,
///       or call validate() after direct construction.
struct DetectionBox {
    float x1, y1, x2, y2;
    float score;
    int64_t class_id;

    /// Create a validated DetectionBox (recommended for C++ users)
    /// @throws ValidationError if coordinates or score are invalid
    static DetectionBox create(
        float x1, float y1, float x2, float y2,
        float score, int64_t class_id = 0
    ) {
        DetectionBox box{x1, y1, x2, y2, score, class_id};
        box.validate();
        return box;
    }

    /// Compute area
    float area() const {
        return std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    }

    /// Compute IoU with another box
    float iou(const DetectionBox& other) const {
        float inter_x1 = std::max(x1, other.x1);
        float inter_y1 = std::max(y1, other.y1);
        float inter_x2 = std::min(x2, other.x2);
        float inter_y2 = std::min(y2, other.y2);

        float inter_area = std::max(0.0f, inter_x2 - inter_x1) *
                          std::max(0.0f, inter_y2 - inter_y1);

        float union_area = area() + other.area() - inter_area;

        if (union_area <= 0.0f) return 0.0f;
        return inter_area / union_area;
    }

    /// Validate box coordinates and score
    /// @throws ValidationError if coordinates are NaN/Inf or exceed limits
    /// @throws ValidationError if score is NaN/Inf
    void validate() const {
        core::validate_grid_coordinate(x1, "box x1");
        core::validate_grid_coordinate(y1, "box y1");
        core::validate_grid_coordinate(x2, "box x2");
        core::validate_grid_coordinate(y2, "box y2");

        if (!std::isfinite(score)) {
            throw transforms::ValidationError("Box score must be finite");
        }
    }
};

/// Non-Maximum Suppression operation
///
/// Filters detection boxes by removing overlapping boxes based on
/// Intersection over Union (IoU) threshold.
///
/// Equivalent to torchvision.ops.nms
///
/// @note Thread Safety: Thread-safe for concurrent calls (immutable after construction).
class NMS {
public:
    /// Create NMS operation
    /// @param iou_threshold IoU threshold for suppression [0, 1]
    explicit NMS(float iou_threshold)
        : iou_threshold_(iou_threshold)
    {
        validate_params();
    }

    /// Compute output (indices of kept boxes)
    /// @param num_boxes Number of input boxes
    /// @return Maximum possible number of kept boxes
    int64_t max_output_size(int64_t num_boxes) const {
        core::validate_nms_box_count(num_boxes);
        return num_boxes;  // Worst case: all boxes kept
    }

    std::string name() const { return "NMS"; }

    std::string repr() const {
        std::ostringstream ss;
        ss << "NMS(iou_threshold=" << iou_threshold_ << ")";
        return ss.str();
    }

    /// Get IoU threshold
    float iou_threshold() const { return iou_threshold_; }

private:
    float iou_threshold_;

    void validate_params() const {
        core::validate_nms_params(iou_threshold_, 0.0f);
    }
};

/// Batched NMS with class-aware suppression
///
/// Performs NMS per class within each batch item.
/// This is useful when you want to keep the best detection per class.
///
/// @note Thread Safety: Thread-safe for concurrent calls (immutable after construction).
class BatchedNMS {
public:
    /// Create batched NMS operation
    /// @param iou_threshold IoU threshold for suppression [0, 1]
    explicit BatchedNMS(float iou_threshold)
        : iou_threshold_(iou_threshold)
    {
        validate_params();
    }

    /// Compute output (indices of kept boxes)
    /// @param num_boxes Number of input boxes
    /// @return Maximum possible number of kept boxes
    int64_t max_output_size(int64_t num_boxes) const {
        core::validate_nms_box_count(num_boxes);
        return num_boxes;  // Worst case: all boxes kept
    }

    std::string name() const { return "BatchedNMS"; }

    std::string repr() const {
        std::ostringstream ss;
        ss << "BatchedNMS(iou_threshold=" << iou_threshold_ << ")";
        return ss.str();
    }

    /// Get IoU threshold
    float iou_threshold() const { return iou_threshold_; }

private:
    float iou_threshold_;

    void validate_params() const {
        core::validate_nms_params(iou_threshold_, 0.0f);
    }
};

/// Soft-NMS operation
///
/// Instead of hard suppression, reduces scores of overlapping boxes
/// using a Gaussian or linear penalty.
///
/// @note Thread Safety: Thread-safe for concurrent calls (immutable after construction).
class SoftNMS {
public:
    /// Soft-NMS method
    enum class Method {
        LINEAR,    // Linear score decay
        GAUSSIAN   // Gaussian score decay
    };

    /// Create soft-NMS operation
    /// @param sigma Gaussian sigma for score decay (only used with GAUSSIAN method)
    /// @param iou_threshold IoU threshold for LINEAR method
    /// @param score_threshold Minimum score to keep a box
    /// @param method Soft-NMS method (LINEAR or GAUSSIAN)
    explicit SoftNMS(
        float sigma = 0.5f,
        float iou_threshold = 0.3f,
        float score_threshold = 0.001f,
        Method method = Method::GAUSSIAN
    )
        : sigma_(sigma)
        , iou_threshold_(iou_threshold)
        , score_threshold_(score_threshold)
        , method_(method)
    {
        validate_params();
    }

    /// Compute output (indices of kept boxes)
    /// @param num_boxes Number of input boxes
    /// @return Maximum possible number of kept boxes
    int64_t max_output_size(int64_t num_boxes) const {
        core::validate_nms_box_count(num_boxes);
        return num_boxes;
    }

    std::string name() const { return "SoftNMS"; }

    std::string repr() const {
        std::ostringstream ss;
        ss << "SoftNMS(sigma=" << sigma_
           << ", iou_threshold=" << iou_threshold_
           << ", score_threshold=" << score_threshold_
           << ", method=" << (method_ == Method::GAUSSIAN ? "gaussian" : "linear")
           << ")";
        return ss.str();
    }

    /// Get sigma
    float sigma() const { return sigma_; }

    /// Get IoU threshold
    float iou_threshold() const { return iou_threshold_; }

    /// Get score threshold
    float score_threshold() const { return score_threshold_; }

    /// Get method
    Method method() const { return method_; }

private:
    float sigma_;
    float iou_threshold_;
    float score_threshold_;
    Method method_;

    void validate_params() const {
        if (!std::isfinite(sigma_) || sigma_ <= 0.0f) {
            throw transforms::ValidationError(
                "SoftNMS sigma must be positive finite, got " + std::to_string(sigma_)
            );
        }
        core::validate_nms_params(iou_threshold_, score_threshold_);
    }
};

}  // namespace pyflame_vision::ops

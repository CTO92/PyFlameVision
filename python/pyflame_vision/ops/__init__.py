"""
PyFlameVision Operations

**PRE-RELEASE ALPHA SOFTWARE** - APIs may change without notice.

Specialized operations for detection and spatial transformers.
API is compatible with torchvision.ops.

Example:
    >>> import pyflame_vision.ops as ops
    >>> # ROI Align for detection models
    >>> roi_align = ops.ROIAlign(output_size=7, spatial_scale=0.25)
    >>> output_shape = roi_align.get_output_shape([1, 256, 50, 50], num_rois=100)
    >>>
    >>> # NMS for filtering detections
    >>> nms = ops.NMS(iou_threshold=0.5)
    >>> max_kept = nms.max_output_size(1000)
    >>>
    >>> # Grid sample for spatial transformers
    >>> gs = ops.GridSample(mode='bilinear', padding_mode='zeros')
    >>> output_shape = gs.get_output_shape([1, 3, 64, 64], [1, 32, 32, 2])
"""

try:
    from .._pyflame_vision_cpp.ops import (
        # Enums
        InterpolationMode,
        PaddingMode,
        SoftNMSMethod,
        # Grid sample
        GridSample,
        # ROI operations
        ROI,
        ROIAlign,
        # Detection box and NMS
        DetectionBox,
        NMS,
        BatchedNMS,
        SoftNMS,
        # Functional API
        roi_align,
        nms,
        batched_nms,
        grid_sample,
    )
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False
    # Provide Python fallback implementations
    from .ops import (
        InterpolationMode,
        PaddingMode,
        SoftNMSMethod,
        GridSample,
        ROI,
        ROIAlign,
        DetectionBox,
        NMS,
        BatchedNMS,
        SoftNMS,
        roi_align,
        nms,
        batched_nms,
        grid_sample,
    )

__all__ = [
    # Enums
    "InterpolationMode",
    "PaddingMode",
    "SoftNMSMethod",
    # Grid sample
    "GridSample",
    # ROI operations
    "ROI",
    "ROIAlign",
    # Detection box and NMS
    "DetectionBox",
    "NMS",
    "BatchedNMS",
    "SoftNMS",
    # Functional API
    "roi_align",
    "nms",
    "batched_nms",
    "grid_sample",
]

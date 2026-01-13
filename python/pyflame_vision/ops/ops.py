"""
Pure Python fallback implementations for PyFlameVision ops.

These are used when the C++ bindings are not available.
"""

from enum import Enum, auto
from typing import List, Tuple, Optional
import math


# ============================================================================
# Security Limits (must match C++ SecurityLimits)
# ============================================================================

MAX_DIMENSION = 65536  # 64K pixels
MAX_TOTAL_ELEMENTS = 1 << 32  # ~4 billion
MAX_BATCH_SIZE = 1024
MAX_CHANNELS = 256
MAX_ROIS = 10000
MAX_ROI_OUTPUT_SIZE = 256
MAX_GRID_SAMPLE_SIZE = 4096
MAX_NMS_BOXES = 100000
MAX_GRID_COORDINATE = 1e6


def _validate_dimension(value: int, name: str = "dimension") -> None:
    """Validate dimension is within allowed limits."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    if value > MAX_DIMENSION:
        raise ValueError(f"{name} ({value}) exceeds maximum allowed ({MAX_DIMENSION})")


def _validate_total_elements(batch: int, channels: int, height: int, width: int) -> None:
    """Validate total image elements are within limits."""
    if batch > MAX_BATCH_SIZE:
        raise ValueError(f"Batch size ({batch}) exceeds maximum ({MAX_BATCH_SIZE})")
    if channels > MAX_CHANNELS:
        raise ValueError(f"Channel count ({channels}) exceeds maximum ({MAX_CHANNELS})")
    total = batch * channels * height * width
    if total > MAX_TOTAL_ELEMENTS:
        raise ValueError(f"Total elements ({total}) exceeds maximum ({MAX_TOTAL_ELEMENTS})")


def _validate_coordinate(coord: float, name: str) -> None:
    """Validate coordinate is finite and within limits."""
    if not math.isfinite(coord):
        raise ValueError(f"{name} must be finite, got NaN or Inf")
    if abs(coord) > MAX_GRID_COORDINATE:
        raise ValueError(f"{name} ({coord}) exceeds maximum allowed ({MAX_GRID_COORDINATE})")


def _validate_roi_count(count: int) -> None:
    """Validate number of ROIs is within limits."""
    if count < 0:
        raise ValueError(f"ROI count must be non-negative, got {count}")
    if count > MAX_ROIS:
        raise ValueError(f"ROI count ({count}) exceeds maximum allowed ({MAX_ROIS})")


def _validate_nms_box_count(count: int) -> None:
    """Validate number of boxes for NMS is within limits."""
    if count < 0:
        raise ValueError(f"Box count must be non-negative, got {count}")
    if count > MAX_NMS_BOXES:
        raise ValueError(f"Box count ({count}) exceeds maximum allowed ({MAX_NMS_BOXES})")


class InterpolationMode(Enum):
    """Interpolation modes for spatial operations."""
    NEAREST = 0
    BILINEAR = 1
    BICUBIC = 2
    AREA = 3


class PaddingMode(Enum):
    """Padding modes for out-of-bounds sampling."""
    ZEROS = 0
    BORDER = 1
    REFLECTION = 2


class SoftNMSMethod(Enum):
    """Soft-NMS methods."""
    LINEAR = auto()
    GAUSSIAN = auto()


class GridSample:
    """Grid-based spatial sampling operation.

    Samples from input tensor at locations specified by a grid.
    Grid coordinates are normalized to [-1, 1].
    Equivalent to torch.nn.functional.grid_sample
    """

    def __init__(
        self,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False
    ):
        if mode not in ("bilinear", "nearest"):
            raise ValueError("GridSample only supports bilinear and nearest interpolation")

        self._mode = mode
        self._padding_mode = padding_mode
        self._align_corners = align_corners

    def get_output_shape(
        self,
        input_shape: List[int],
        grid_shape: List[int]
    ) -> List[int]:
        """Compute output shape given input and grid shapes."""
        if len(input_shape) != 4:
            raise ValueError(f"GridSample input must be 4D (NCHW), got {len(input_shape)}D")
        if len(grid_shape) != 4:
            raise ValueError(f"GridSample grid must be 4D [N, H_out, W_out, 2], got {len(grid_shape)}D")
        if input_shape[0] != grid_shape[0]:
            raise ValueError(f"Batch size mismatch: input has {input_shape[0]}, grid has {grid_shape[0]}")
        if grid_shape[3] != 2:
            raise ValueError(f"Grid last dimension must be 2 (x, y), got {grid_shape[3]}")

        # Security: Validate dimensions
        _validate_dimension(input_shape[2], "input height")
        _validate_dimension(input_shape[3], "input width")

        # Security: Validate output size limits
        if grid_shape[1] > MAX_GRID_SAMPLE_SIZE or grid_shape[2] > MAX_GRID_SAMPLE_SIZE:
            raise ValueError(
                f"GridSample output size exceeds maximum ({MAX_GRID_SAMPLE_SIZE})"
            )

        return [input_shape[0], input_shape[1], grid_shape[1], grid_shape[2]]

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def padding_mode(self) -> str:
        return self._padding_mode

    @property
    def align_corners(self) -> bool:
        return self._align_corners

    def halo_size(self) -> int:
        return 1 if self._mode == "bilinear" else 0

    def __repr__(self) -> str:
        return f"GridSample(mode={self._mode}, padding_mode={self._padding_mode}, align_corners={self._align_corners})"


class ROI:
    """Region of Interest specification.

    Format: [batch_index, x1, y1, x2, y2]
    """

    def __init__(
        self,
        batch_index: int,
        x1: float,
        y1: float,
        x2: float,
        y2: float
    ):
        if batch_index < 0:
            raise ValueError(f"ROI batch_index must be non-negative, got {batch_index}")

        # Security: Validate coordinates
        _validate_coordinate(x1, "ROI x1")
        _validate_coordinate(y1, "ROI y1")
        _validate_coordinate(x2, "ROI x2")
        _validate_coordinate(y2, "ROI y2")

        if x2 < x1 or y2 < y1:
            raise ValueError("ROI coordinates must satisfy x1 <= x2 and y1 <= y2")

        self.batch_index = batch_index
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def width(self) -> float:
        return self.x2 - self.x1

    def height(self) -> float:
        return self.y2 - self.y1

    def area(self) -> float:
        return max(0.0, self.width()) * max(0.0, self.height())


class ROIAlign:
    """Region of Interest Align operation.

    Extracts fixed-size feature maps from regions of interest.
    Equivalent to torchvision.ops.roi_align
    """

    def __init__(
        self,
        output_size,
        spatial_scale: float,
        sampling_ratio: int = 0,
        aligned: bool = True
    ):
        if isinstance(output_size, int):
            self._output_height = output_size
            self._output_width = output_size
        else:
            self._output_height, self._output_width = output_size

        if self._output_height <= 0 or self._output_width <= 0:
            raise ValueError(f"ROI Align output size must be positive")

        # Security: Validate output size limits
        if self._output_height > MAX_ROI_OUTPUT_SIZE or self._output_width > MAX_ROI_OUTPUT_SIZE:
            raise ValueError(
                f"ROI Align output size exceeds maximum ({MAX_ROI_OUTPUT_SIZE})"
            )

        if not math.isfinite(spatial_scale) or spatial_scale <= 0:
            raise ValueError(f"ROI Align spatial_scale must be positive finite, got {spatial_scale}")
        if sampling_ratio < 0:
            raise ValueError(f"ROI Align sampling_ratio must be non-negative, got {sampling_ratio}")

        self._spatial_scale = spatial_scale
        self._sampling_ratio = sampling_ratio
        self._aligned = aligned

    def get_output_shape(
        self,
        input_shape: List[int],
        num_rois: int
    ) -> List[int]:
        """Compute output shape given input shape and number of ROIs."""
        if len(input_shape) != 4:
            raise ValueError(f"ROIAlign input must be 4D (NCHW), got {len(input_shape)}D")

        # Security: Validate ROI count
        _validate_roi_count(num_rois)

        # Security: Validate total output elements
        _validate_total_elements(
            num_rois,
            input_shape[1],
            self._output_height,
            self._output_width
        )

        return [num_rois, input_shape[1], self._output_height, self._output_width]

    @property
    def output_height(self) -> int:
        return self._output_height

    @property
    def output_width(self) -> int:
        return self._output_width

    @property
    def spatial_scale(self) -> float:
        return self._spatial_scale

    @property
    def sampling_ratio(self) -> int:
        return self._sampling_ratio

    @property
    def aligned(self) -> bool:
        return self._aligned

    def halo_size(self) -> int:
        return 1

    def __repr__(self) -> str:
        return (f"ROIAlign(output_size=({self._output_height}, {self._output_width}), "
                f"spatial_scale={self._spatial_scale}, sampling_ratio={self._sampling_ratio}, "
                f"aligned={self._aligned})")


class DetectionBox:
    """Detection box for NMS.

    Format: [x1, y1, x2, y2, score, class_id]
    """

    def __init__(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        score: float,
        class_id: int = 0
    ):
        # Security: Validate coordinates
        _validate_coordinate(x1, "box x1")
        _validate_coordinate(y1, "box y1")
        _validate_coordinate(x2, "box x2")
        _validate_coordinate(y2, "box y2")

        if not math.isfinite(score):
            raise ValueError("Box score must be finite")

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.score = score
        self.class_id = class_id

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def iou(self, other: "DetectionBox") -> float:
        """Compute IoU with another box."""
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)

        inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        union_area = self.area() + other.area() - inter_area

        if union_area <= 0.0:
            return 0.0
        return inter_area / union_area


class NMS:
    """Non-Maximum Suppression operation.

    Filters detection boxes by removing overlapping boxes.
    Equivalent to torchvision.ops.nms
    """

    def __init__(self, iou_threshold: float):
        if not math.isfinite(iou_threshold):
            raise ValueError("NMS iou_threshold must be finite")
        if iou_threshold < 0.0 or iou_threshold > 1.0:
            raise ValueError(f"NMS iou_threshold must be in [0, 1], got {iou_threshold}")

        self._iou_threshold = iou_threshold

    def max_output_size(self, num_boxes: int) -> int:
        """Get maximum possible number of kept boxes."""
        # Security: Validate box count
        _validate_nms_box_count(num_boxes)
        return num_boxes

    @property
    def iou_threshold(self) -> float:
        return self._iou_threshold

    def __repr__(self) -> str:
        return f"NMS(iou_threshold={self._iou_threshold})"


class BatchedNMS:
    """Batched class-aware Non-Maximum Suppression.

    Performs NMS per class within each batch item.
    """

    def __init__(self, iou_threshold: float):
        if not math.isfinite(iou_threshold):
            raise ValueError("NMS iou_threshold must be finite")
        if iou_threshold < 0.0 or iou_threshold > 1.0:
            raise ValueError(f"NMS iou_threshold must be in [0, 1], got {iou_threshold}")

        self._iou_threshold = iou_threshold

    def max_output_size(self, num_boxes: int) -> int:
        """Get maximum possible number of kept boxes."""
        # Security: Validate box count
        _validate_nms_box_count(num_boxes)
        return num_boxes

    @property
    def iou_threshold(self) -> float:
        return self._iou_threshold

    def __repr__(self) -> str:
        return f"BatchedNMS(iou_threshold={self._iou_threshold})"


class SoftNMS:
    """Soft Non-Maximum Suppression.

    Reduces scores of overlapping boxes instead of hard suppression.
    """

    def __init__(
        self,
        sigma: float = 0.5,
        iou_threshold: float = 0.3,
        score_threshold: float = 0.001,
        method: str = "gaussian"
    ):
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"SoftNMS sigma must be positive finite, got {sigma}")
        if not math.isfinite(iou_threshold):
            raise ValueError("SoftNMS iou_threshold must be finite")
        if iou_threshold < 0.0 or iou_threshold > 1.0:
            raise ValueError(f"SoftNMS iou_threshold must be in [0, 1], got {iou_threshold}")
        if not math.isfinite(score_threshold):
            raise ValueError("SoftNMS score_threshold must be finite")
        # Security: Validate score_threshold is non-negative
        if score_threshold < 0.0:
            raise ValueError(f"SoftNMS score_threshold must be non-negative, got {score_threshold}")

        self._sigma = sigma
        self._iou_threshold = iou_threshold
        self._score_threshold = score_threshold
        self._method = SoftNMSMethod.GAUSSIAN if method == "gaussian" else SoftNMSMethod.LINEAR

    def max_output_size(self, num_boxes: int) -> int:
        """Get maximum possible number of kept boxes."""
        # Security: Validate box count
        _validate_nms_box_count(num_boxes)
        return num_boxes

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def iou_threshold(self) -> float:
        return self._iou_threshold

    @property
    def score_threshold(self) -> float:
        return self._score_threshold

    def __repr__(self) -> str:
        method_str = "gaussian" if self._method == SoftNMSMethod.GAUSSIAN else "linear"
        return (f"SoftNMS(sigma={self._sigma}, iou_threshold={self._iou_threshold}, "
                f"score_threshold={self._score_threshold}, method={method_str})")


# Functional API
def roi_align(
    input_shape: List[int],
    num_rois: int,
    output_size,
    spatial_scale: float,
    sampling_ratio: int = 0,
    aligned: bool = True
) -> List[int]:
    """Compute ROI Align output shape."""
    op = ROIAlign(output_size, spatial_scale, sampling_ratio, aligned)
    return op.get_output_shape(input_shape, num_rois)


def nms(num_boxes: int, iou_threshold: float) -> int:
    """Get maximum output size for NMS."""
    op = NMS(iou_threshold)
    return op.max_output_size(num_boxes)


def batched_nms(num_boxes: int, iou_threshold: float) -> int:
    """Get maximum output size for batched NMS."""
    op = BatchedNMS(iou_threshold)
    return op.max_output_size(num_boxes)


def grid_sample(
    input_shape: List[int],
    grid_shape: List[int],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False
) -> List[int]:
    """Compute grid sample output shape."""
    op = GridSample(mode, padding_mode, align_corners)
    return op.get_output_shape(input_shape, grid_shape)

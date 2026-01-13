"""
Unit tests for Phase 4 specialized operations.

Tests for GridSample, ROIAlign, and NMS operations.
"""

import pytest
import math


class TestImports:
    """Test that ops module imports successfully."""

    def test_module_import(self):
        from pyflame_vision import ops
        assert hasattr(ops, 'GridSample')
        assert hasattr(ops, 'ROIAlign')
        assert hasattr(ops, 'NMS')

    def test_all_exports(self):
        from pyflame_vision.ops import (
            GridSample, ROIAlign, NMS, BatchedNMS, SoftNMS,
            ROI, DetectionBox,
            roi_align, nms, batched_nms, grid_sample
        )


class TestGridSample:
    """Tests for GridSample operation."""

    def test_default_construction(self):
        from pyflame_vision.ops import GridSample
        gs = GridSample()
        assert gs.mode == "bilinear"
        assert gs.padding_mode == "zeros"
        assert gs.align_corners == False

    def test_construction_with_params(self):
        from pyflame_vision.ops import GridSample
        gs = GridSample(mode="nearest", padding_mode="border", align_corners=True)
        assert gs.mode == "nearest"
        assert gs.padding_mode == "border"
        assert gs.align_corners == True

    def test_invalid_mode_raises(self):
        from pyflame_vision.ops import GridSample
        with pytest.raises(Exception):
            GridSample(mode="bicubic")

    def test_output_shape(self):
        from pyflame_vision.ops import GridSample
        gs = GridSample()
        input_shape = [2, 3, 64, 64]
        grid_shape = [2, 32, 32, 2]
        output = gs.get_output_shape(input_shape, grid_shape)
        assert output == [2, 3, 32, 32]

    def test_batch_mismatch_raises(self):
        from pyflame_vision.ops import GridSample
        gs = GridSample()
        with pytest.raises(Exception):
            gs.get_output_shape([2, 3, 64, 64], [4, 32, 32, 2])

    def test_invalid_grid_last_dim_raises(self):
        from pyflame_vision.ops import GridSample
        gs = GridSample()
        with pytest.raises(Exception):
            gs.get_output_shape([2, 3, 64, 64], [2, 32, 32, 3])

    def test_halo_size(self):
        from pyflame_vision.ops import GridSample
        bilinear = GridSample(mode="bilinear")
        nearest = GridSample(mode="nearest")
        assert bilinear.halo_size() == 1
        assert nearest.halo_size() == 0

    def test_repr(self):
        from pyflame_vision.ops import GridSample
        gs = GridSample(mode="bilinear", padding_mode="reflection", align_corners=True)
        repr_str = repr(gs)
        assert "bilinear" in repr_str
        assert "reflection" in repr_str


class TestROI:
    """Tests for ROI struct."""

    def test_construction(self):
        from pyflame_vision.ops import ROI
        roi = ROI(batch_index=0, x1=10.0, y1=20.0, x2=30.0, y2=50.0)
        assert roi.batch_index == 0
        assert roi.x1 == pytest.approx(10.0)
        assert roi.y1 == pytest.approx(20.0)
        assert roi.x2 == pytest.approx(30.0)
        assert roi.y2 == pytest.approx(50.0)

    def test_width_height(self):
        from pyflame_vision.ops import ROI
        roi = ROI(0, 10.0, 20.0, 30.0, 50.0)
        assert roi.width() == pytest.approx(20.0)
        assert roi.height() == pytest.approx(30.0)

    def test_area(self):
        from pyflame_vision.ops import ROI
        roi = ROI(0, 10.0, 20.0, 30.0, 50.0)
        assert roi.area() == pytest.approx(600.0)

    def test_invalid_batch_index_raises(self):
        from pyflame_vision.ops import ROI
        with pytest.raises(Exception):
            ROI(-1, 0, 0, 10, 10)

    def test_invalid_coordinates_raises(self):
        from pyflame_vision.ops import ROI
        with pytest.raises(Exception):
            ROI(0, 30.0, 50.0, 10.0, 20.0)  # x2 < x1


class TestROIAlign:
    """Tests for ROIAlign operation."""

    def test_construction(self):
        from pyflame_vision.ops import ROIAlign
        ra = ROIAlign(output_size=7, spatial_scale=0.25)
        assert ra.output_height == 7
        assert ra.output_width == 7
        assert ra.spatial_scale == pytest.approx(0.25)
        assert ra.sampling_ratio == 0
        assert ra.aligned == True

    def test_tuple_output_size(self):
        from pyflame_vision.ops import ROIAlign
        ra = ROIAlign(output_size=(7, 14), spatial_scale=0.25)
        assert ra.output_height == 7
        assert ra.output_width == 14

    def test_invalid_spatial_scale_raises(self):
        from pyflame_vision.ops import ROIAlign
        with pytest.raises(Exception):
            ROIAlign(7, 0.0)
        with pytest.raises(Exception):
            ROIAlign(7, -0.5)
        with pytest.raises(Exception):
            ROIAlign(7, float('inf'))

    def test_output_shape(self):
        from pyflame_vision.ops import ROIAlign
        ra = ROIAlign(7, 0.25)
        output = ra.get_output_shape([1, 256, 50, 50], num_rois=100)
        assert output == [100, 256, 7, 7]

    def test_halo_size(self):
        from pyflame_vision.ops import ROIAlign
        ra = ROIAlign(7, 0.25)
        assert ra.halo_size() == 1

    def test_repr(self):
        from pyflame_vision.ops import ROIAlign
        ra = ROIAlign(7, 0.25, sampling_ratio=2)
        repr_str = repr(ra)
        assert "7" in repr_str
        assert "0.25" in repr_str


class TestDetectionBox:
    """Tests for DetectionBox struct."""

    def test_construction(self):
        from pyflame_vision.ops import DetectionBox
        box = DetectionBox(10.0, 20.0, 30.0, 50.0, 0.9, 1)
        assert box.x1 == pytest.approx(10.0)
        assert box.y1 == pytest.approx(20.0)
        assert box.x2 == pytest.approx(30.0)
        assert box.y2 == pytest.approx(50.0)
        assert box.score == pytest.approx(0.9)
        assert box.class_id == 1

    def test_area(self):
        from pyflame_vision.ops import DetectionBox
        box = DetectionBox(10.0, 20.0, 30.0, 50.0, 0.9, 1)
        assert box.area() == pytest.approx(600.0)

    def test_iou(self):
        from pyflame_vision.ops import DetectionBox
        box1 = DetectionBox(0.0, 0.0, 10.0, 10.0, 0.9, 1)
        box2 = DetectionBox(5.0, 5.0, 15.0, 15.0, 0.8, 1)
        # Intersection: [5,5] to [10,10] = 25
        # Union: 100 + 100 - 25 = 175
        expected_iou = 25.0 / 175.0
        assert box1.iou(box2) == pytest.approx(expected_iou, rel=1e-5)

    def test_no_overlap_iou(self):
        from pyflame_vision.ops import DetectionBox
        box1 = DetectionBox(0.0, 0.0, 10.0, 10.0, 0.9, 1)
        box2 = DetectionBox(20.0, 20.0, 30.0, 30.0, 0.8, 1)
        assert box1.iou(box2) == pytest.approx(0.0)

    def test_full_overlap_iou(self):
        from pyflame_vision.ops import DetectionBox
        box1 = DetectionBox(0.0, 0.0, 10.0, 10.0, 0.9, 1)
        box2 = DetectionBox(0.0, 0.0, 10.0, 10.0, 0.8, 1)
        assert box1.iou(box2) == pytest.approx(1.0)

    def test_nan_score_raises(self):
        from pyflame_vision.ops import DetectionBox
        with pytest.raises(Exception):
            DetectionBox(0.0, 0.0, 10.0, 10.0, float('nan'), 1)


class TestNMS:
    """Tests for NMS operation."""

    def test_construction(self):
        from pyflame_vision.ops import NMS
        nms = NMS(0.5)
        assert nms.iou_threshold == pytest.approx(0.5)

    def test_invalid_threshold_raises(self):
        from pyflame_vision.ops import NMS
        with pytest.raises(Exception):
            NMS(-0.1)
        with pytest.raises(Exception):
            NMS(1.5)
        with pytest.raises(Exception):
            NMS(float('nan'))

    def test_boundary_thresholds(self):
        from pyflame_vision.ops import NMS
        nms0 = NMS(0.0)
        nms1 = NMS(1.0)
        assert nms0.iou_threshold == pytest.approx(0.0)
        assert nms1.iou_threshold == pytest.approx(1.0)

    def test_max_output_size(self):
        from pyflame_vision.ops import NMS
        nms = NMS(0.5)
        assert nms.max_output_size(100) == 100
        assert nms.max_output_size(0) == 0

    def test_repr(self):
        from pyflame_vision.ops import NMS
        nms = NMS(0.45)
        repr_str = repr(nms)
        assert "0.45" in repr_str


class TestBatchedNMS:
    """Tests for BatchedNMS operation."""

    def test_construction(self):
        from pyflame_vision.ops import BatchedNMS
        bnms = BatchedNMS(0.5)
        assert bnms.iou_threshold == pytest.approx(0.5)

    def test_invalid_threshold_raises(self):
        from pyflame_vision.ops import BatchedNMS
        with pytest.raises(Exception):
            BatchedNMS(-0.1)

    def test_max_output_size(self):
        from pyflame_vision.ops import BatchedNMS
        bnms = BatchedNMS(0.5)
        assert bnms.max_output_size(100) == 100


class TestSoftNMS:
    """Tests for SoftNMS operation."""

    def test_default_construction(self):
        from pyflame_vision.ops import SoftNMS
        snms = SoftNMS()
        assert snms.sigma == pytest.approx(0.5)
        assert snms.iou_threshold == pytest.approx(0.3)
        assert snms.score_threshold == pytest.approx(0.001)

    def test_custom_params(self):
        from pyflame_vision.ops import SoftNMS
        snms = SoftNMS(sigma=0.7, iou_threshold=0.4, method="linear")
        assert snms.sigma == pytest.approx(0.7)
        assert snms.iou_threshold == pytest.approx(0.4)

    def test_invalid_sigma_raises(self):
        from pyflame_vision.ops import SoftNMS
        with pytest.raises(Exception):
            SoftNMS(sigma=0.0)
        with pytest.raises(Exception):
            SoftNMS(sigma=-0.5)

    def test_max_output_size(self):
        from pyflame_vision.ops import SoftNMS
        snms = SoftNMS()
        assert snms.max_output_size(100) == 100


class TestFunctionalAPI:
    """Tests for functional API."""

    def test_roi_align_function(self):
        from pyflame_vision.ops import roi_align
        result = roi_align(
            input_shape=[1, 256, 50, 50],
            num_rois=100,
            output_size=7,
            spatial_scale=0.25
        )
        assert result == [100, 256, 7, 7]

    def test_nms_function(self):
        from pyflame_vision.ops import nms
        result = nms(num_boxes=100, iou_threshold=0.5)
        assert result == 100

    def test_batched_nms_function(self):
        from pyflame_vision.ops import batched_nms
        result = batched_nms(num_boxes=100, iou_threshold=0.5)
        assert result == 100

    def test_grid_sample_function(self):
        from pyflame_vision.ops import grid_sample
        result = grid_sample(
            input_shape=[2, 3, 64, 64],
            grid_shape=[2, 32, 32, 2],
            mode="bilinear",
            padding_mode="zeros"
        )
        assert result == [2, 3, 32, 32]


class TestSecurityLimits:
    """Tests for security validation."""

    def test_roi_count_limit(self):
        """Test that ROI count is limited."""
        from pyflame_vision.ops import ROIAlign
        ra = ROIAlign(7, 0.25)
        with pytest.raises(Exception):
            ra.get_output_shape([1, 256, 50, 50], num_rois=20000)  # Exceeds MAX_ROIS

    def test_nms_box_count_limit(self):
        """Test that NMS box count is limited."""
        from pyflame_vision.ops import NMS
        nms = NMS(0.5)
        with pytest.raises(Exception):
            nms.max_output_size(200000)  # Exceeds MAX_NMS_BOXES

    def test_grid_sample_size_limit(self):
        """Test that grid sample output size is limited."""
        from pyflame_vision.ops import GridSample
        gs = GridSample()
        with pytest.raises(Exception):
            gs.get_output_shape([1, 3, 64, 64], [1, 5000, 5000, 2])  # Exceeds MAX_GRID_SAMPLE_SIZE

    def test_roi_coordinate_limit(self):
        """Test that ROI coordinates are limited."""
        from pyflame_vision.ops import ROI
        with pytest.raises(Exception):
            ROI(0, 0, 0, 2e6, 10)  # Exceeds MAX_GRID_COORDINATE

    def test_detection_box_coordinate_limit(self):
        """Test that DetectionBox coordinates are limited."""
        from pyflame_vision.ops import DetectionBox
        with pytest.raises(Exception):
            DetectionBox(0, 0, 2e6, 10, 0.9)  # Exceeds MAX_GRID_COORDINATE

    def test_roi_nan_coordinate_raises(self):
        """Test that NaN coordinates are rejected."""
        from pyflame_vision.ops import ROI
        with pytest.raises(Exception):
            ROI(0, float('nan'), 0, 10, 10)

    def test_detection_box_nan_coordinate_raises(self):
        """Test that NaN coordinates are rejected in DetectionBox."""
        from pyflame_vision.ops import DetectionBox
        with pytest.raises(Exception):
            DetectionBox(float('nan'), 0, 10, 10, 0.9)

    def test_soft_nms_negative_score_threshold_raises(self):
        """Test that negative score_threshold is rejected."""
        from pyflame_vision.ops import SoftNMS
        with pytest.raises(Exception):
            SoftNMS(score_threshold=-0.001)

    def test_roi_align_output_size_limit(self):
        """Test that ROI Align output size is limited."""
        from pyflame_vision.ops import ROIAlign
        with pytest.raises(Exception):
            ROIAlign(300, 0.25)  # Exceeds MAX_ROI_OUTPUT_SIZE


class TestIntegration:
    """Integration tests combining multiple operations."""

    def test_grid_sample_with_resize(self):
        """Test GridSample integrates with existing transforms."""
        from pyflame_vision.ops import GridSample
        from pyflame_vision.transforms import Resize

        # Resize then grid sample
        resize = Resize((64, 64))
        gs = GridSample()

        input_shape = [1, 3, 128, 128]
        resized = resize.get_output_shape(input_shape)
        assert resized == [1, 3, 64, 64]

        grid_shape = [1, 32, 32, 2]
        output = gs.get_output_shape(resized, grid_shape)
        assert output == [1, 3, 32, 32]

    def test_roi_align_typical_fpn(self):
        """Test ROIAlign with typical FPN settings."""
        from pyflame_vision.ops import ROIAlign

        # FPN levels with different scales
        scales = [1/4, 1/8, 1/16, 1/32]

        for scale in scales:
            ra = ROIAlign(7, scale)
            input_shape = [1, 256, 200, 200]
            output = ra.get_output_shape(input_shape, num_rois=512)
            assert output == [512, 256, 7, 7]

    def test_nms_cascade(self):
        """Test NMS with different thresholds."""
        from pyflame_vision.ops import NMS

        thresholds = [0.7, 0.6, 0.5]
        for thresh in thresholds:
            nms = NMS(thresh)
            max_boxes = nms.max_output_size(1000)
            assert max_boxes <= 1000

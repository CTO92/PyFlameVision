"""
Unit tests for PyFlameVision functional API.
"""

import pytest
from pyflame_vision.transforms.functional import (
    get_output_shape_resize,
    get_output_shape_center_crop,
    get_output_shape_random_crop,
    get_output_shape_normalize,
    compute_center_crop_bounds,
    validate_image_shape,
    validate_normalize_params,
)


class TestGetOutputShapeResize:
    """Tests for get_output_shape_resize function."""

    def test_int_size(self):
        shape = get_output_shape_resize([1, 3, 480, 640], 224)
        assert shape == [1, 3, 224, 224]

    def test_tuple_size(self):
        shape = get_output_shape_resize([1, 3, 480, 640], (256, 512))
        assert shape == [1, 3, 256, 512]

    def test_preserves_batch_and_channels(self):
        shape = get_output_shape_resize([8, 1, 480, 640], 224)
        assert shape[0] == 8
        assert shape[1] == 1

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            get_output_shape_resize([3, 224, 224], 224)


class TestGetOutputShapeCenterCrop:
    """Tests for get_output_shape_center_crop function."""

    def test_int_size(self):
        shape = get_output_shape_center_crop([1, 3, 256, 256], 224)
        assert shape == [1, 3, 224, 224]

    def test_tuple_size(self):
        shape = get_output_shape_center_crop([1, 3, 256, 256], (200, 240))
        assert shape == [1, 3, 200, 240]

    def test_crop_larger_than_input(self):
        with pytest.raises(ValueError):
            get_output_shape_center_crop([1, 3, 256, 256], 512)

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            get_output_shape_center_crop([3, 256, 256], 224)


class TestGetOutputShapeRandomCrop:
    """Tests for get_output_shape_random_crop function."""

    def test_int_size(self):
        shape = get_output_shape_random_crop([1, 3, 256, 256], 224)
        assert shape == [1, 3, 224, 224]

    def test_tuple_size(self):
        shape = get_output_shape_random_crop([1, 3, 256, 256], (200, 240))
        assert shape == [1, 3, 200, 240]

    def test_with_padding(self):
        shape = get_output_shape_random_crop([1, 3, 256, 256], 224, padding=4)
        assert shape == [1, 3, 224, 224]

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            get_output_shape_random_crop([3, 256, 256], 224)


class TestGetOutputShapeNormalize:
    """Tests for get_output_shape_normalize function."""

    def test_preserves_shape(self):
        input_shape = [1, 3, 224, 224]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        shape = get_output_shape_normalize(input_shape, mean, std)
        assert shape == input_shape

    def test_channel_mismatch_mean(self):
        with pytest.raises(ValueError):
            get_output_shape_normalize([1, 3, 224, 224], [0.5], [0.5])

    def test_channel_mismatch_std(self):
        with pytest.raises(ValueError):
            get_output_shape_normalize([1, 3, 224, 224], [0.5, 0.5, 0.5], [0.5])

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            get_output_shape_normalize([3, 224, 224], [0.5], [0.5])


class TestComputeCenterCropBounds:
    """Tests for compute_center_crop_bounds function."""

    def test_square_crop(self):
        top, left, height, width = compute_center_crop_bounds(256, 256, 224, 224)
        assert top == 16  # (256 - 224) / 2
        assert left == 16
        assert height == 224
        assert width == 224

    def test_rectangular_crop(self):
        top, left, height, width = compute_center_crop_bounds(480, 640, 256, 512)
        assert top == (480 - 256) // 2
        assert left == (640 - 512) // 2
        assert height == 256
        assert width == 512

    def test_crop_larger_than_input(self):
        with pytest.raises(ValueError):
            compute_center_crop_bounds(256, 256, 512, 512)


class TestValidateImageShape:
    """Tests for validate_image_shape function."""

    def test_valid_shape(self):
        validate_image_shape([1, 3, 224, 224])  # Should not raise

    def test_wrong_dims(self):
        with pytest.raises(ValueError):
            validate_image_shape([3, 224, 224])

    def test_zero_dim(self):
        with pytest.raises(ValueError):
            validate_image_shape([1, 0, 224, 224])

    def test_negative_dim(self):
        with pytest.raises(ValueError):
            validate_image_shape([1, 3, -1, 224])


class TestValidateNormalizeParams:
    """Tests for validate_normalize_params function."""

    def test_valid_params(self):
        validate_normalize_params(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )  # Should not raise

    def test_mismatched_length(self):
        with pytest.raises(ValueError):
            validate_normalize_params([0.5, 0.5], [0.5, 0.5, 0.5])

    def test_empty_mean(self):
        with pytest.raises(ValueError):
            validate_normalize_params([], [])

    def test_zero_std(self):
        with pytest.raises(ValueError):
            validate_normalize_params([0.5, 0.5, 0.5], [0.5, 0.0, 0.5])

    def test_negative_std(self):
        with pytest.raises(ValueError):
            validate_normalize_params([0.5, 0.5, 0.5], [0.5, -0.5, 0.5])

    def test_channel_mismatch(self):
        with pytest.raises(ValueError):
            validate_normalize_params([0.5], [0.5], num_channels=3)

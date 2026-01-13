"""
Unit tests for PyFlameVision Python transforms.

Tests the pure Python fallback implementations.
"""

import pytest
from pyflame_vision.transforms import (
    Resize,
    CenterCrop,
    RandomCrop,
    Normalize,
    Compose,
    Size,
)


class TestSize:
    """Tests for Size class."""

    def test_square_size(self):
        size = Size(224)
        assert size.height == 224
        assert size.width == 224

    def test_rectangular_size(self):
        size = Size(480, 640)
        assert size.height == 480
        assert size.width == 640

    def test_is_valid(self):
        assert Size(224).is_valid()
        assert Size(100, 200).is_valid()

    def test_repr_square(self):
        assert repr(Size(224)) == "224"

    def test_repr_rectangular(self):
        assert repr(Size(480, 640)) == "(480, 640)"


class TestResize:
    """Tests for Resize transform."""

    def test_square_resize(self):
        resize = Resize(224)
        shape = resize.get_output_shape([1, 3, 480, 640])
        assert shape == [1, 3, 224, 224]

    def test_tuple_size(self):
        resize = Resize((256, 512))
        shape = resize.get_output_shape([1, 3, 480, 640])
        assert shape == [1, 3, 256, 512]

    def test_preserves_batch_and_channels(self):
        resize = Resize(224)
        shape = resize.get_output_shape([4, 1, 480, 640])
        assert shape[0] == 4  # batch
        assert shape[1] == 1  # channels

    def test_interpolation_default(self):
        resize = Resize(224)
        assert resize.interpolation() == "bilinear"

    def test_interpolation_nearest(self):
        resize = Resize(224, interpolation="nearest")
        assert resize.interpolation() == "nearest"

    def test_antialias_default(self):
        resize = Resize(224)
        assert resize.antialias() is True

    def test_name(self):
        resize = Resize(224)
        assert resize.name() == "Resize"

    def test_is_deterministic(self):
        resize = Resize(224)
        assert resize.is_deterministic() is True

    def test_invalid_input_shape(self):
        resize = Resize(224)
        with pytest.raises(ValueError):
            resize.get_output_shape([3, 224, 224])  # Missing batch dim

    def test_repr(self):
        resize = Resize(224)
        assert "Resize" in repr(resize)
        assert "224" in repr(resize)


class TestCenterCrop:
    """Tests for CenterCrop transform."""

    def test_square_crop(self):
        crop = CenterCrop(224)
        shape = crop.get_output_shape([1, 3, 256, 256])
        assert shape == [1, 3, 224, 224]

    def test_tuple_size(self):
        crop = CenterCrop((200, 250))
        shape = crop.get_output_shape([1, 3, 256, 256])
        assert shape == [1, 3, 200, 250]

    def test_compute_bounds(self):
        crop = CenterCrop(224)
        top, left, height, width = crop.compute_bounds(256, 256)
        assert top == 16  # (256 - 224) / 2
        assert left == 16
        assert height == 224
        assert width == 224

    def test_crop_larger_than_input(self):
        crop = CenterCrop(512)
        with pytest.raises(ValueError):
            crop.get_output_shape([1, 3, 256, 256])

    def test_name(self):
        crop = CenterCrop(224)
        assert crop.name() == "CenterCrop"

    def test_is_deterministic(self):
        crop = CenterCrop(224)
        assert crop.is_deterministic() is True


class TestRandomCrop:
    """Tests for RandomCrop transform."""

    def test_output_shape(self):
        crop = RandomCrop(224)
        shape = crop.get_output_shape([1, 3, 256, 256])
        assert shape == [1, 3, 224, 224]

    def test_with_padding(self):
        crop = RandomCrop(224, padding=4)
        assert crop.padding() == 4

    def test_pad_if_needed(self):
        crop = RandomCrop(224, pad_if_needed=True)
        assert crop.pad_if_needed() is True

    def test_set_seed(self):
        crop = RandomCrop(224)
        crop.set_seed(42)
        # Seed should be set without error

    def test_name(self):
        crop = RandomCrop(224)
        assert crop.name() == "RandomCrop"

    def test_is_not_deterministic(self):
        crop = RandomCrop(224)
        assert crop.is_deterministic() is False


class TestNormalize:
    """Tests for Normalize transform."""

    @pytest.fixture
    def imagenet_params(self):
        return {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }

    def test_output_shape_preserved(self, imagenet_params):
        norm = Normalize(**imagenet_params)
        input_shape = [1, 3, 224, 224]
        assert norm.get_output_shape(input_shape) == input_shape

    def test_mean_accessor(self, imagenet_params):
        norm = Normalize(**imagenet_params)
        assert norm.mean() == imagenet_params["mean"]

    def test_std_accessor(self, imagenet_params):
        norm = Normalize(**imagenet_params)
        assert norm.std() == imagenet_params["std"]

    def test_inv_std(self, imagenet_params):
        norm = Normalize(**imagenet_params)
        inv_std = norm.inv_std()
        for i, s in enumerate(imagenet_params["std"]):
            assert abs(inv_std[i] - 1.0 / s) < 1e-6

    def test_inplace_default(self, imagenet_params):
        norm = Normalize(**imagenet_params)
        assert norm.inplace() is False

    def test_inplace_true(self, imagenet_params):
        norm = Normalize(**imagenet_params, inplace=True)
        assert norm.inplace() is True

    def test_mismatched_mean_std_length(self):
        with pytest.raises(ValueError):
            Normalize(mean=[0.5, 0.5], std=[0.5, 0.5, 0.5])

    def test_empty_mean(self):
        with pytest.raises(ValueError):
            Normalize(mean=[], std=[])

    def test_zero_std(self):
        with pytest.raises(ValueError):
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.0, 0.5])

    def test_negative_std(self):
        with pytest.raises(ValueError):
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, -0.5, 0.5])

    def test_channel_mismatch(self):
        norm = Normalize(mean=[0.5], std=[0.5])  # Single channel
        with pytest.raises(ValueError):
            norm.get_output_shape([1, 3, 224, 224])  # 3 channels

    def test_name(self, imagenet_params):
        norm = Normalize(**imagenet_params)
        assert norm.name() == "Normalize"

    def test_is_deterministic(self, imagenet_params):
        norm = Normalize(**imagenet_params)
        assert norm.is_deterministic() is True


class TestCompose:
    """Tests for Compose transform."""

    @pytest.fixture
    def imagenet_mean(self):
        return [0.485, 0.456, 0.406]

    @pytest.fixture
    def imagenet_std(self):
        return [0.229, 0.224, 0.225]

    def test_single_transform(self):
        pipeline = Compose([Resize(224)])
        shape = pipeline.get_output_shape([1, 3, 480, 640])
        assert shape == [1, 3, 224, 224]

    def test_multiple_transforms(self):
        pipeline = Compose([
            Resize(256),
            CenterCrop(224),
        ])
        shape = pipeline.get_output_shape([1, 3, 480, 640])
        assert shape == [1, 3, 224, 224]

    def test_imagenet_pipeline(self, imagenet_mean, imagenet_std):
        pipeline = Compose([
            Resize(256),
            CenterCrop(224),
            Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
        shape = pipeline.get_output_shape([1, 3, 480, 640])
        assert shape == [1, 3, 224, 224]

    def test_empty_pipeline(self):
        pipeline = Compose([])
        assert pipeline.empty() is True
        shape = pipeline.get_output_shape([1, 3, 480, 640])
        assert shape == [1, 3, 480, 640]  # Passthrough

    def test_length(self):
        pipeline = Compose([Resize(256), CenterCrop(224)])
        assert len(pipeline) == 2

    def test_index_access(self):
        pipeline = Compose([Resize(256), CenterCrop(224)])
        assert pipeline[0].name() == "Resize"
        assert pipeline[1].name() == "CenterCrop"

    def test_transforms_accessor(self):
        pipeline = Compose([Resize(256), CenterCrop(224)])
        transforms = pipeline.transforms()
        assert len(transforms) == 2

    def test_deterministic_pipeline(self):
        pipeline = Compose([Resize(256), CenterCrop(224)])
        assert pipeline.is_deterministic() is True

    def test_non_deterministic_pipeline(self):
        pipeline = Compose([Resize(256), RandomCrop(224)])
        assert pipeline.is_deterministic() is False

    def test_name(self):
        pipeline = Compose([])
        assert pipeline.name() == "Compose"

    def test_none_transform_raises(self):
        with pytest.raises(ValueError):
            Compose([None])

    def test_repr(self):
        pipeline = Compose([Resize(256), CenterCrop(224)])
        r = repr(pipeline)
        assert "Compose" in r
        assert "Resize" in r
        assert "CenterCrop" in r

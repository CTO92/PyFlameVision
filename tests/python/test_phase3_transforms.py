"""
Unit tests for PyFlameVision Phase 3 transforms.

Tests for random data augmentation transforms:
- RandomHorizontalFlip
- RandomVerticalFlip
- RandomRotation
- ColorJitter
- GaussianBlur
"""

import pytest
from pyflame_vision.transforms import (
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    ColorJitter,
    GaussianBlur,
    Compose,
)


class TestRandomHorizontalFlip:
    """Tests for RandomHorizontalFlip transform."""

    def test_output_shape_preserved(self):
        flip = RandomHorizontalFlip()
        shape = flip.get_output_shape([1, 3, 224, 224])
        assert shape == [1, 3, 224, 224]

    def test_default_probability(self):
        flip = RandomHorizontalFlip()
        assert flip.probability() == pytest.approx(0.5)

    def test_custom_probability(self):
        flip = RandomHorizontalFlip(p=0.7)
        assert flip.probability() == pytest.approx(0.7)

    def test_zero_probability_never_flips(self):
        flip = RandomHorizontalFlip(p=0.0)
        flip.set_seed(42)
        for _ in range(100):
            flip.get_output_shape([1, 3, 224, 224])
            assert flip.was_flipped() is False

    def test_one_probability_always_flips(self):
        flip = RandomHorizontalFlip(p=1.0)
        flip.set_seed(42)
        for _ in range(100):
            flip.get_output_shape([1, 3, 224, 224])
            assert flip.was_flipped() is True

    def test_invalid_probability_negative(self):
        with pytest.raises(ValueError):
            RandomHorizontalFlip(p=-0.1)

    def test_invalid_probability_greater_than_one(self):
        with pytest.raises(ValueError):
            RandomHorizontalFlip(p=1.5)

    def test_deterministic_with_seed(self):
        flip1 = RandomHorizontalFlip(p=0.5)
        flip2 = RandomHorizontalFlip(p=0.5)
        flip1.set_seed(12345)
        flip2.set_seed(12345)

        for _ in range(20):
            flip1.get_output_shape([1, 3, 224, 224])
            flip2.get_output_shape([1, 3, 224, 224])
            assert flip1.was_flipped() == flip2.was_flipped()

    def test_name(self):
        flip = RandomHorizontalFlip()
        assert flip.name() == "RandomHorizontalFlip"

    def test_is_not_deterministic(self):
        flip = RandomHorizontalFlip()
        assert flip.is_deterministic() is False

    def test_repr(self):
        flip = RandomHorizontalFlip(p=0.3)
        r = repr(flip)
        assert "RandomHorizontalFlip" in r
        assert "0.3" in r


class TestRandomVerticalFlip:
    """Tests for RandomVerticalFlip transform."""

    def test_output_shape_preserved(self):
        flip = RandomVerticalFlip()
        shape = flip.get_output_shape([1, 3, 224, 224])
        assert shape == [1, 3, 224, 224]

    def test_default_probability(self):
        flip = RandomVerticalFlip()
        assert flip.probability() == pytest.approx(0.5)

    def test_custom_probability(self):
        flip = RandomVerticalFlip(p=0.8)
        assert flip.probability() == pytest.approx(0.8)

    def test_zero_probability_never_flips(self):
        flip = RandomVerticalFlip(p=0.0)
        flip.set_seed(42)
        for _ in range(100):
            flip.get_output_shape([1, 3, 224, 224])
            assert flip.was_flipped() is False

    def test_one_probability_always_flips(self):
        flip = RandomVerticalFlip(p=1.0)
        flip.set_seed(42)
        for _ in range(100):
            flip.get_output_shape([1, 3, 224, 224])
            assert flip.was_flipped() is True

    def test_invalid_probability_negative(self):
        with pytest.raises(ValueError):
            RandomVerticalFlip(p=-0.5)

    def test_invalid_probability_greater_than_one(self):
        with pytest.raises(ValueError):
            RandomVerticalFlip(p=2.0)

    def test_deterministic_with_seed(self):
        flip1 = RandomVerticalFlip(p=0.5)
        flip2 = RandomVerticalFlip(p=0.5)
        flip1.set_seed(99999)
        flip2.set_seed(99999)

        for _ in range(20):
            flip1.get_output_shape([1, 3, 224, 224])
            flip2.get_output_shape([1, 3, 224, 224])
            assert flip1.was_flipped() == flip2.was_flipped()

    def test_name(self):
        flip = RandomVerticalFlip()
        assert flip.name() == "RandomVerticalFlip"

    def test_is_not_deterministic(self):
        flip = RandomVerticalFlip()
        assert flip.is_deterministic() is False


class TestRandomRotation:
    """Tests for RandomRotation transform."""

    def test_output_shape_preserved_no_expand(self):
        rot = RandomRotation(30.0)
        shape = rot.get_output_shape([1, 3, 224, 224])
        assert shape == [1, 3, 224, 224]

    def test_symmetric_degrees(self):
        rot = RandomRotation(45.0)
        degrees = rot.degrees()
        assert degrees[0] == pytest.approx(-45.0)
        assert degrees[1] == pytest.approx(45.0)

    def test_asymmetric_degrees(self):
        rot = RandomRotation(degrees=(-30.0, 60.0))
        degrees = rot.degrees()
        assert degrees[0] == pytest.approx(-30.0)
        assert degrees[1] == pytest.approx(60.0)

    def test_angle_in_range(self):
        rot = RandomRotation(45.0)
        rot.set_seed(42)
        for _ in range(100):
            rot.get_output_shape([1, 3, 224, 224])
            angle = rot.last_angle()
            assert -45.0 <= angle <= 45.0

    def test_expand_changes_size(self):
        rot = RandomRotation(degrees=(45.0, 45.0), expand=True)
        rot.set_seed(42)
        input_shape = [1, 3, 224, 224]
        output = rot.get_output_shape(input_shape)
        # 45-degree rotation should increase bounding box
        assert output[2] > input_shape[2]
        assert output[3] > input_shape[3]

    def test_interpolation_default(self):
        rot = RandomRotation(30.0)
        assert rot.interpolation() == "bilinear"

    def test_interpolation_nearest(self):
        rot = RandomRotation(30.0, interpolation="nearest")
        assert rot.interpolation() == "nearest"

    def test_expand_default(self):
        rot = RandomRotation(30.0)
        assert rot.expand() is False

    def test_expand_true(self):
        rot = RandomRotation(30.0, expand=True)
        assert rot.expand() is True

    def test_invalid_degrees_min_greater_than_max(self):
        with pytest.raises(ValueError):
            RandomRotation(degrees=(60.0, 30.0))

    def test_invalid_degrees_exceeds_max(self):
        with pytest.raises(ValueError):
            RandomRotation(400.0)

    def test_fill_values(self):
        rot = RandomRotation(30.0, fill=[0.5, 0.5, 0.5])
        assert len(rot.fill()) == 3
        assert rot.fill()[0] == pytest.approx(0.5)

    def test_deterministic_with_seed(self):
        rot1 = RandomRotation(45.0)
        rot2 = RandomRotation(45.0)
        rot1.set_seed(12345)
        rot2.set_seed(12345)

        for _ in range(20):
            rot1.get_output_shape([1, 3, 224, 224])
            rot2.get_output_shape([1, 3, 224, 224])
            assert rot1.last_angle() == pytest.approx(rot2.last_angle())

    def test_name(self):
        rot = RandomRotation(30.0)
        assert rot.name() == "RandomRotation"

    def test_is_not_deterministic(self):
        rot = RandomRotation(30.0)
        assert rot.is_deterministic() is False

    def test_repr(self):
        rot = RandomRotation(30.0)
        r = repr(rot)
        assert "RandomRotation" in r


class TestColorJitter:
    """Tests for ColorJitter transform."""

    def test_output_shape_preserved(self):
        jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        shape = jitter.get_output_shape([1, 3, 224, 224])
        assert shape == [1, 3, 224, 224]

    def test_default_values(self):
        jitter = ColorJitter()
        brightness = jitter.brightness()
        contrast = jitter.contrast()
        saturation = jitter.saturation()
        hue = jitter.hue()

        # Default should be no-op (identity ranges)
        assert brightness[0] == pytest.approx(1.0)
        assert brightness[1] == pytest.approx(1.0)
        assert contrast[0] == pytest.approx(1.0)
        assert contrast[1] == pytest.approx(1.0)
        assert saturation[0] == pytest.approx(1.0)
        assert saturation[1] == pytest.approx(1.0)
        assert hue[0] == pytest.approx(0.0)
        assert hue[1] == pytest.approx(0.0)

    def test_single_value_constructor(self):
        jitter = ColorJitter(brightness=0.2, contrast=0.3, saturation=0.4, hue=0.1)

        # brightness=0.2 -> [0.8, 1.2]
        brightness = jitter.brightness()
        assert brightness[0] == pytest.approx(0.8)
        assert brightness[1] == pytest.approx(1.2)

        # contrast=0.3 -> [0.7, 1.3]
        contrast = jitter.contrast()
        assert contrast[0] == pytest.approx(0.7)
        assert contrast[1] == pytest.approx(1.3)

        # saturation=0.4 -> [0.6, 1.4]
        saturation = jitter.saturation()
        assert saturation[0] == pytest.approx(0.6)
        assert saturation[1] == pytest.approx(1.4)

        # hue=0.1 -> [-0.1, 0.1]
        hue = jitter.hue()
        assert hue[0] == pytest.approx(-0.1)
        assert hue[1] == pytest.approx(0.1)

    def test_factors_in_range(self):
        jitter = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
        jitter.set_seed(42)

        for _ in range(100):
            jitter.get_output_shape([1, 3, 224, 224])

            assert 0.5 <= jitter.last_brightness_factor() <= 1.5
            assert 0.5 <= jitter.last_contrast_factor() <= 1.5
            assert 0.5 <= jitter.last_saturation_factor() <= 1.5
            assert -0.2 <= jitter.last_hue_factor() <= 0.2

    def test_random_order(self):
        jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        jitter.set_seed(42)

        seen_orders = set()
        for _ in range(100):
            jitter.get_output_shape([1, 3, 224, 224])
            seen_orders.add(tuple(jitter.last_order()))

        # Should have seen multiple permutations
        assert len(seen_orders) > 1

    def test_invalid_hue_exceeds_max(self):
        with pytest.raises((ValueError, RuntimeError)):
            ColorJitter(hue=0.6)

    def test_deterministic_with_seed(self):
        jitter1 = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        jitter2 = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        jitter1.set_seed(54321)
        jitter2.set_seed(54321)

        for _ in range(20):
            jitter1.get_output_shape([1, 3, 224, 224])
            jitter2.get_output_shape([1, 3, 224, 224])

            assert jitter1.last_brightness_factor() == pytest.approx(jitter2.last_brightness_factor())
            assert jitter1.last_contrast_factor() == pytest.approx(jitter2.last_contrast_factor())
            assert jitter1.last_saturation_factor() == pytest.approx(jitter2.last_saturation_factor())
            assert jitter1.last_hue_factor() == pytest.approx(jitter2.last_hue_factor())
            assert jitter1.last_order() == jitter2.last_order()

    def test_name(self):
        jitter = ColorJitter()
        assert jitter.name() == "ColorJitter"

    def test_is_not_deterministic(self):
        jitter = ColorJitter()
        assert jitter.is_deterministic() is False

    def test_repr(self):
        jitter = ColorJitter(brightness=0.2)
        r = repr(jitter)
        assert "ColorJitter" in r


class TestGaussianBlur:
    """Tests for GaussianBlur transform."""

    def test_output_shape_preserved(self):
        blur = GaussianBlur(kernel_size=5)
        shape = blur.get_output_shape([1, 3, 224, 224])
        assert shape == [1, 3, 224, 224]

    def test_fixed_kernel_size(self):
        blur = GaussianBlur(kernel_size=7)
        ks = blur.kernel_size()
        assert ks[0] == 7
        assert ks[1] == 7

    def test_kernel_size_range(self):
        blur = GaussianBlur(kernel_size=(3, 7))
        ks = blur.kernel_size()
        assert ks[0] == 3
        assert ks[1] == 7

    def test_kernel_size_in_range(self):
        blur = GaussianBlur(kernel_size=(3, 9))
        blur.set_seed(42)

        seen_sizes = set()
        for _ in range(100):
            blur.get_output_shape([1, 3, 224, 224])
            k = blur.last_kernel_size()
            assert 3 <= k <= 9
            assert k % 2 == 1  # Must be odd
            seen_sizes.add(k)

        # Should have seen multiple sizes
        assert len(seen_sizes) > 1

    def test_sigma_range(self):
        blur = GaussianBlur(kernel_size=5, sigma=(0.5, 2.5))
        sigma = blur.sigma()
        assert sigma[0] == pytest.approx(0.5)
        assert sigma[1] == pytest.approx(2.5)

    def test_sigma_in_range(self):
        blur = GaussianBlur(kernel_size=5, sigma=(0.5, 2.0))
        blur.set_seed(42)

        for _ in range(100):
            blur.get_output_shape([1, 3, 224, 224])
            s = blur.last_sigma()
            assert 0.5 <= s <= 2.0

    def test_invalid_kernel_size_even(self):
        with pytest.raises(ValueError):
            GaussianBlur(kernel_size=4)

    def test_invalid_kernel_size_zero(self):
        with pytest.raises(ValueError):
            GaussianBlur(kernel_size=0)

    def test_invalid_kernel_size_negative(self):
        with pytest.raises(ValueError):
            GaussianBlur(kernel_size=-1)

    def test_invalid_kernel_size_exceeds_max(self):
        with pytest.raises((ValueError, RuntimeError)):
            GaussianBlur(kernel_size=33)

    def test_invalid_kernel_range_min_greater_than_max(self):
        with pytest.raises(ValueError):
            GaussianBlur(kernel_size=(7, 3))

    def test_kernel_weights(self):
        blur = GaussianBlur(kernel_size=5, sigma=(1.0, 1.0))
        blur.get_output_shape([1, 3, 224, 224])

        weights = blur.get_kernel_weights()

        # Should have kernel_size elements
        assert len(weights) == 5

        # Weights should sum to 1
        assert sum(weights) == pytest.approx(1.0, abs=0.001)

        # All weights should be positive
        for w in weights:
            assert w > 0

        # Should be symmetric
        assert weights[0] == pytest.approx(weights[4])
        assert weights[1] == pytest.approx(weights[3])

        # Center should have highest weight
        assert weights[2] > weights[1]
        assert weights[1] > weights[0]

    def test_halo_size(self):
        blur = GaussianBlur(kernel_size=5)
        blur.get_output_shape([1, 3, 224, 224])
        assert blur.halo_size() == 2  # 5 // 2

    def test_deterministic_with_seed(self):
        blur1 = GaussianBlur(kernel_size=(3, 9))
        blur2 = GaussianBlur(kernel_size=(3, 9))
        blur1.set_seed(11111)
        blur2.set_seed(11111)

        for _ in range(20):
            blur1.get_output_shape([1, 3, 224, 224])
            blur2.get_output_shape([1, 3, 224, 224])

            assert blur1.last_kernel_size() == blur2.last_kernel_size()
            assert blur1.last_sigma() == pytest.approx(blur2.last_sigma())

    def test_name(self):
        blur = GaussianBlur(kernel_size=5)
        assert blur.name() == "GaussianBlur"

    def test_is_not_deterministic(self):
        blur = GaussianBlur(kernel_size=5)
        assert blur.is_deterministic() is False

    def test_repr(self):
        blur = GaussianBlur(kernel_size=5)
        r = repr(blur)
        assert "GaussianBlur" in r
        assert "5" in r


class TestPhase3Compose:
    """Integration tests for Phase 3 transforms with Compose."""

    def test_augmentation_pipeline(self):
        """Test typical training augmentation pipeline."""
        pipeline = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomRotation(15.0),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            GaussianBlur(kernel_size=(3, 7)),
        ])

        input_shape = [1, 3, 256, 256]
        output_shape = pipeline.get_output_shape(input_shape)

        # All transforms preserve shape
        assert output_shape == input_shape

    def test_reproducible_augmentation(self):
        """Test that seeding produces reproducible results."""
        flip = RandomHorizontalFlip(p=0.5)
        rot = RandomRotation(30.0)
        jitter = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)

        # First pass
        flip.set_seed(42)
        rot.set_seed(42)
        jitter.set_seed(42)

        flip.get_output_shape([1, 3, 256, 256])
        rot.get_output_shape([1, 3, 256, 256])
        jitter.get_output_shape([1, 3, 256, 256])

        flipped1 = flip.was_flipped()
        angle1 = rot.last_angle()
        brightness1 = jitter.last_brightness_factor()

        # Second pass with same seeds
        flip.set_seed(42)
        rot.set_seed(42)
        jitter.set_seed(42)

        flip.get_output_shape([1, 3, 256, 256])
        rot.get_output_shape([1, 3, 256, 256])
        jitter.get_output_shape([1, 3, 256, 256])

        assert flipped1 == flip.was_flipped()
        assert angle1 == pytest.approx(rot.last_angle())
        assert brightness1 == pytest.approx(jitter.last_brightness_factor())

    def test_non_deterministic_pipeline(self):
        """Test that pipeline with random transforms is not deterministic."""
        pipeline = Compose([
            RandomHorizontalFlip(),
            RandomRotation(15.0),
        ])
        assert pipeline.is_deterministic() is False

/**
 * Unit tests for PyFlameVision Phase 3 transforms
 *
 * Tests for random data augmentation transforms:
 * - RandomHorizontalFlip
 * - RandomVerticalFlip
 * - RandomRotation
 * - ColorJitter
 * - GaussianBlur
 */

#include <gtest/gtest.h>
#include <pyflame_vision/transforms/flip.hpp>
#include <pyflame_vision/transforms/rotation.hpp>
#include <pyflame_vision/transforms/color_jitter.hpp>
#include <pyflame_vision/transforms/blur.hpp>
#include <pyflame_vision/core/security.hpp>
#include <cmath>
#include <set>

using namespace pyflame_vision::transforms;
using namespace pyflame_vision::core;

// ============================================================================
// RandomHorizontalFlip tests
// ============================================================================

class RandomHorizontalFlipTest : public ::testing::Test {
protected:
    std::vector<int64_t> input_shape = {1, 3, 224, 224};
};

TEST_F(RandomHorizontalFlipTest, OutputShapePreserved) {
    RandomHorizontalFlip flip;
    auto output = flip.get_output_shape(input_shape);
    EXPECT_EQ(output, input_shape);
}

TEST_F(RandomHorizontalFlipTest, DefaultProbability) {
    RandomHorizontalFlip flip;
    EXPECT_FLOAT_EQ(flip.probability(), 0.5f);
}

TEST_F(RandomHorizontalFlipTest, CustomProbability) {
    RandomHorizontalFlip flip(0.7f);
    EXPECT_FLOAT_EQ(flip.probability(), 0.7f);
}

TEST_F(RandomHorizontalFlipTest, ZeroProbability) {
    RandomHorizontalFlip flip(0.0f);
    flip.set_seed(42);

    // With p=0, should never flip
    for (int i = 0; i < 100; ++i) {
        flip.get_output_shape(input_shape);
        EXPECT_FALSE(flip.was_flipped());
    }
}

TEST_F(RandomHorizontalFlipTest, OneProbability) {
    RandomHorizontalFlip flip(1.0f);
    flip.set_seed(42);

    // With p=1, should always flip
    for (int i = 0; i < 100; ++i) {
        flip.get_output_shape(input_shape);
        EXPECT_TRUE(flip.was_flipped());
    }
}

TEST_F(RandomHorizontalFlipTest, InvalidProbabilityNegative) {
    EXPECT_THROW(RandomHorizontalFlip(-0.1f), ValidationError);
}

TEST_F(RandomHorizontalFlipTest, InvalidProbabilityGreaterThanOne) {
    EXPECT_THROW(RandomHorizontalFlip(1.5f), ValidationError);
}

TEST_F(RandomHorizontalFlipTest, DeterministicWithSeed) {
    RandomHorizontalFlip flip1(0.5f);
    RandomHorizontalFlip flip2(0.5f);

    flip1.set_seed(12345);
    flip2.set_seed(12345);

    // Same seed should produce same sequence
    for (int i = 0; i < 20; ++i) {
        flip1.get_output_shape(input_shape);
        flip2.get_output_shape(input_shape);
        EXPECT_EQ(flip1.was_flipped(), flip2.was_flipped());
    }
}

TEST_F(RandomHorizontalFlipTest, Name) {
    RandomHorizontalFlip flip;
    EXPECT_EQ(flip.name(), "RandomHorizontalFlip");
}

TEST_F(RandomHorizontalFlipTest, IsNotDeterministic) {
    RandomHorizontalFlip flip;
    EXPECT_FALSE(flip.is_deterministic());
}

TEST_F(RandomHorizontalFlipTest, Repr) {
    RandomHorizontalFlip flip(0.3f);
    std::string repr = flip.repr();
    EXPECT_TRUE(repr.find("RandomHorizontalFlip") != std::string::npos);
    EXPECT_TRUE(repr.find("0.3") != std::string::npos);
}

TEST_F(RandomHorizontalFlipTest, InvalidInputShape) {
    RandomHorizontalFlip flip;
    std::vector<int64_t> invalid_shape = {3, 224, 224};  // Missing batch dim
    EXPECT_THROW(flip.get_output_shape(invalid_shape), std::runtime_error);
}

// ============================================================================
// RandomVerticalFlip tests
// ============================================================================

class RandomVerticalFlipTest : public ::testing::Test {
protected:
    std::vector<int64_t> input_shape = {1, 3, 224, 224};
};

TEST_F(RandomVerticalFlipTest, OutputShapePreserved) {
    RandomVerticalFlip flip;
    auto output = flip.get_output_shape(input_shape);
    EXPECT_EQ(output, input_shape);
}

TEST_F(RandomVerticalFlipTest, DefaultProbability) {
    RandomVerticalFlip flip;
    EXPECT_FLOAT_EQ(flip.probability(), 0.5f);
}

TEST_F(RandomVerticalFlipTest, CustomProbability) {
    RandomVerticalFlip flip(0.8f);
    EXPECT_FLOAT_EQ(flip.probability(), 0.8f);
}

TEST_F(RandomVerticalFlipTest, ZeroProbability) {
    RandomVerticalFlip flip(0.0f);
    flip.set_seed(42);

    for (int i = 0; i < 100; ++i) {
        flip.get_output_shape(input_shape);
        EXPECT_FALSE(flip.was_flipped());
    }
}

TEST_F(RandomVerticalFlipTest, OneProbability) {
    RandomVerticalFlip flip(1.0f);
    flip.set_seed(42);

    for (int i = 0; i < 100; ++i) {
        flip.get_output_shape(input_shape);
        EXPECT_TRUE(flip.was_flipped());
    }
}

TEST_F(RandomVerticalFlipTest, InvalidProbabilityNegative) {
    EXPECT_THROW(RandomVerticalFlip(-0.5f), ValidationError);
}

TEST_F(RandomVerticalFlipTest, InvalidProbabilityGreaterThanOne) {
    EXPECT_THROW(RandomVerticalFlip(2.0f), ValidationError);
}

TEST_F(RandomVerticalFlipTest, DeterministicWithSeed) {
    RandomVerticalFlip flip1(0.5f);
    RandomVerticalFlip flip2(0.5f);

    flip1.set_seed(99999);
    flip2.set_seed(99999);

    for (int i = 0; i < 20; ++i) {
        flip1.get_output_shape(input_shape);
        flip2.get_output_shape(input_shape);
        EXPECT_EQ(flip1.was_flipped(), flip2.was_flipped());
    }
}

TEST_F(RandomVerticalFlipTest, Name) {
    RandomVerticalFlip flip;
    EXPECT_EQ(flip.name(), "RandomVerticalFlip");
}

TEST_F(RandomVerticalFlipTest, IsNotDeterministic) {
    RandomVerticalFlip flip;
    EXPECT_FALSE(flip.is_deterministic());
}

// ============================================================================
// RandomRotation tests
// ============================================================================

class RandomRotationTest : public ::testing::Test {
protected:
    std::vector<int64_t> input_shape = {1, 3, 224, 224};
};

TEST_F(RandomRotationTest, OutputShapePreservedNoExpand) {
    RandomRotation rot(30.0f);
    auto output = rot.get_output_shape(input_shape);
    EXPECT_EQ(output, input_shape);
}

TEST_F(RandomRotationTest, SymmetricDegrees) {
    RandomRotation rot(45.0f);
    auto [min_deg, max_deg] = rot.degrees();
    EXPECT_FLOAT_EQ(min_deg, -45.0f);
    EXPECT_FLOAT_EQ(max_deg, 45.0f);
}

TEST_F(RandomRotationTest, AsymmetricDegrees) {
    RandomRotation rot(-30.0f, 60.0f);
    auto [min_deg, max_deg] = rot.degrees();
    EXPECT_FLOAT_EQ(min_deg, -30.0f);
    EXPECT_FLOAT_EQ(max_deg, 60.0f);
}

TEST_F(RandomRotationTest, AngleInRange) {
    RandomRotation rot(45.0f);
    rot.set_seed(42);

    for (int i = 0; i < 100; ++i) {
        rot.get_output_shape(input_shape);
        float angle = rot.last_angle();
        EXPECT_GE(angle, -45.0f);
        EXPECT_LE(angle, 45.0f);
    }
}

TEST_F(RandomRotationTest, ExpandedOutputSize) {
    // 45 degree rotation of square should increase size
    RandomRotation rot(45.0f, 45.0f, InterpolationMode::BILINEAR, true);
    rot.set_seed(42);

    auto output = rot.get_output_shape(input_shape);

    // For 45 degree rotation, the bounding box increases
    // New size ≈ width * sqrt(2) for a square
    EXPECT_GT(output[2], input_shape[2]);
    EXPECT_GT(output[3], input_shape[3]);
}

TEST_F(RandomRotationTest, ZeroRotationNoExpand) {
    RandomRotation rot(0.0f, 0.0f, InterpolationMode::BILINEAR, true);
    auto output = rot.get_output_shape(input_shape);

    // Zero rotation should keep size (or very close)
    EXPECT_EQ(output[2], input_shape[2]);
    EXPECT_EQ(output[3], input_shape[3]);
}

TEST_F(RandomRotationTest, InterpolationDefault) {
    RandomRotation rot(30.0f);
    EXPECT_EQ(rot.interpolation(), InterpolationMode::BILINEAR);
}

TEST_F(RandomRotationTest, InterpolationNearest) {
    RandomRotation rot(30.0f, InterpolationMode::NEAREST);
    EXPECT_EQ(rot.interpolation(), InterpolationMode::NEAREST);
}

TEST_F(RandomRotationTest, ExpandDefault) {
    RandomRotation rot(30.0f);
    EXPECT_FALSE(rot.expand());
}

TEST_F(RandomRotationTest, ExpandTrue) {
    RandomRotation rot(30.0f, InterpolationMode::BILINEAR, true);
    EXPECT_TRUE(rot.expand());
}

TEST_F(RandomRotationTest, InvalidDegreesMinGreaterThanMax) {
    EXPECT_THROW(RandomRotation(60.0f, 30.0f), ValidationError);
}

TEST_F(RandomRotationTest, InvalidDegreesExceedsMax) {
    EXPECT_THROW(RandomRotation(400.0f), ValidationError);
}

TEST_F(RandomRotationTest, FillValues) {
    std::vector<float> fill = {0.5f, 0.5f, 0.5f};
    RandomRotation rot(30.0f, InterpolationMode::BILINEAR, false, std::nullopt, fill);

    EXPECT_EQ(rot.fill().size(), 3);
    EXPECT_FLOAT_EQ(rot.fill()[0], 0.5f);
}

TEST_F(RandomRotationTest, DeterministicWithSeed) {
    RandomRotation rot1(45.0f);
    RandomRotation rot2(45.0f);

    rot1.set_seed(12345);
    rot2.set_seed(12345);

    for (int i = 0; i < 20; ++i) {
        rot1.get_output_shape(input_shape);
        rot2.get_output_shape(input_shape);
        EXPECT_FLOAT_EQ(rot1.last_angle(), rot2.last_angle());
    }
}

TEST_F(RandomRotationTest, Name) {
    RandomRotation rot(30.0f);
    EXPECT_EQ(rot.name(), "RandomRotation");
}

TEST_F(RandomRotationTest, IsNotDeterministic) {
    RandomRotation rot(30.0f);
    EXPECT_FALSE(rot.is_deterministic());
}

TEST_F(RandomRotationTest, HaloSize) {
    RandomRotation rot(30.0f, InterpolationMode::BILINEAR);
    EXPECT_GE(rot.halo_size(), 1);
}

TEST_F(RandomRotationTest, Repr) {
    RandomRotation rot(30.0f);
    std::string repr = rot.repr();
    EXPECT_TRUE(repr.find("RandomRotation") != std::string::npos);
    EXPECT_TRUE(repr.find("-30") != std::string::npos || repr.find("30") != std::string::npos);
}

// ============================================================================
// ColorJitter tests
// ============================================================================

class ColorJitterTest : public ::testing::Test {
protected:
    std::vector<int64_t> input_shape = {1, 3, 224, 224};
};

TEST_F(ColorJitterTest, OutputShapePreserved) {
    ColorJitter jitter(0.2f, 0.2f, 0.2f, 0.1f);
    auto output = jitter.get_output_shape(input_shape);
    EXPECT_EQ(output, input_shape);
}

TEST_F(ColorJitterTest, DefaultValues) {
    ColorJitter jitter;

    // Default should be no-op (all ranges = identity)
    auto brightness = jitter.brightness();
    auto contrast = jitter.contrast();
    auto saturation = jitter.saturation();
    auto hue = jitter.hue();

    EXPECT_FLOAT_EQ(brightness.first, 1.0f);
    EXPECT_FLOAT_EQ(brightness.second, 1.0f);
    EXPECT_FLOAT_EQ(contrast.first, 1.0f);
    EXPECT_FLOAT_EQ(contrast.second, 1.0f);
    EXPECT_FLOAT_EQ(saturation.first, 1.0f);
    EXPECT_FLOAT_EQ(saturation.second, 1.0f);
    EXPECT_FLOAT_EQ(hue.first, 0.0f);
    EXPECT_FLOAT_EQ(hue.second, 0.0f);
}

TEST_F(ColorJitterTest, SingleValueConstructor) {
    ColorJitter jitter(0.2f, 0.3f, 0.4f, 0.1f);

    // brightness=0.2 -> [max(0, 1-0.2), 1+0.2] = [0.8, 1.2]
    auto brightness = jitter.brightness();
    EXPECT_FLOAT_EQ(brightness.first, 0.8f);
    EXPECT_FLOAT_EQ(brightness.second, 1.2f);

    // contrast=0.3 -> [0.7, 1.3]
    auto contrast = jitter.contrast();
    EXPECT_FLOAT_EQ(contrast.first, 0.7f);
    EXPECT_FLOAT_EQ(contrast.second, 1.3f);

    // saturation=0.4 -> [0.6, 1.4]
    auto saturation = jitter.saturation();
    EXPECT_FLOAT_EQ(saturation.first, 0.6f);
    EXPECT_FLOAT_EQ(saturation.second, 1.4f);

    // hue=0.1 -> [-0.1, 0.1]
    auto hue = jitter.hue();
    EXPECT_FLOAT_EQ(hue.first, -0.1f);
    EXPECT_FLOAT_EQ(hue.second, 0.1f);
}

TEST_F(ColorJitterTest, ExplicitRangeConstructor) {
    ColorJitter jitter(
        {0.5f, 1.5f},   // brightness
        {0.6f, 1.4f},   // contrast
        {0.7f, 1.3f},   // saturation
        {-0.2f, 0.2f}   // hue
    );

    EXPECT_FLOAT_EQ(jitter.brightness().first, 0.5f);
    EXPECT_FLOAT_EQ(jitter.brightness().second, 1.5f);
    EXPECT_FLOAT_EQ(jitter.contrast().first, 0.6f);
    EXPECT_FLOAT_EQ(jitter.contrast().second, 1.4f);
}

TEST_F(ColorJitterTest, FactorsInRange) {
    ColorJitter jitter(0.5f, 0.5f, 0.5f, 0.2f);
    jitter.set_seed(42);

    for (int i = 0; i < 100; ++i) {
        jitter.get_output_shape(input_shape);

        // Check brightness in [0.5, 1.5]
        EXPECT_GE(jitter.last_brightness_factor(), 0.5f);
        EXPECT_LE(jitter.last_brightness_factor(), 1.5f);

        // Check contrast in [0.5, 1.5]
        EXPECT_GE(jitter.last_contrast_factor(), 0.5f);
        EXPECT_LE(jitter.last_contrast_factor(), 1.5f);

        // Check saturation in [0.5, 1.5]
        EXPECT_GE(jitter.last_saturation_factor(), 0.5f);
        EXPECT_LE(jitter.last_saturation_factor(), 1.5f);

        // Check hue in [-0.2, 0.2]
        EXPECT_GE(jitter.last_hue_factor(), -0.2f);
        EXPECT_LE(jitter.last_hue_factor(), 0.2f);
    }
}

TEST_F(ColorJitterTest, RandomOrder) {
    ColorJitter jitter(0.2f, 0.2f, 0.2f, 0.1f);
    jitter.set_seed(42);

    std::set<std::array<int, 4>> seen_orders;

    // Should see multiple different orders
    for (int i = 0; i < 100; ++i) {
        jitter.get_output_shape(input_shape);
        seen_orders.insert(jitter.last_order());
    }

    // Should have seen multiple permutations
    EXPECT_GT(seen_orders.size(), 1);
}

TEST_F(ColorJitterTest, InvalidBrightnessNegative) {
    EXPECT_THROW(
        ColorJitter({-0.5f, 1.0f}, {1.0f, 1.0f}, {1.0f, 1.0f}, {0.0f, 0.0f}),
        ValidationError
    );
}

TEST_F(ColorJitterTest, InvalidBrightnessExceedsMax) {
    EXPECT_THROW(
        ColorJitter({1.0f, 3.0f}, {1.0f, 1.0f}, {1.0f, 1.0f}, {0.0f, 0.0f}),
        ResourceError
    );
}

TEST_F(ColorJitterTest, InvalidHueExceedsMax) {
    EXPECT_THROW(
        ColorJitter({1.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 1.0f}, {-0.6f, 0.6f}),
        ResourceError
    );
}

TEST_F(ColorJitterTest, InvalidRangeMinGreaterThanMax) {
    EXPECT_THROW(
        ColorJitter({1.5f, 0.5f}, {1.0f, 1.0f}, {1.0f, 1.0f}, {0.0f, 0.0f}),
        ValidationError
    );
}

TEST_F(ColorJitterTest, DeterministicWithSeed) {
    ColorJitter jitter1(0.3f, 0.3f, 0.3f, 0.1f);
    ColorJitter jitter2(0.3f, 0.3f, 0.3f, 0.1f);

    jitter1.set_seed(54321);
    jitter2.set_seed(54321);

    for (int i = 0; i < 20; ++i) {
        jitter1.get_output_shape(input_shape);
        jitter2.get_output_shape(input_shape);

        EXPECT_FLOAT_EQ(jitter1.last_brightness_factor(), jitter2.last_brightness_factor());
        EXPECT_FLOAT_EQ(jitter1.last_contrast_factor(), jitter2.last_contrast_factor());
        EXPECT_FLOAT_EQ(jitter1.last_saturation_factor(), jitter2.last_saturation_factor());
        EXPECT_FLOAT_EQ(jitter1.last_hue_factor(), jitter2.last_hue_factor());
        EXPECT_EQ(jitter1.last_order(), jitter2.last_order());
    }
}

TEST_F(ColorJitterTest, Name) {
    ColorJitter jitter;
    EXPECT_EQ(jitter.name(), "ColorJitter");
}

TEST_F(ColorJitterTest, IsNotDeterministic) {
    ColorJitter jitter;
    EXPECT_FALSE(jitter.is_deterministic());
}

TEST_F(ColorJitterTest, Repr) {
    ColorJitter jitter(0.2f, 0.0f, 0.0f, 0.0f);
    std::string repr = jitter.repr();
    EXPECT_TRUE(repr.find("ColorJitter") != std::string::npos);
    EXPECT_TRUE(repr.find("brightness") != std::string::npos);
}

// ============================================================================
// GaussianBlur tests
// ============================================================================

class GaussianBlurTest : public ::testing::Test {
protected:
    std::vector<int64_t> input_shape = {1, 3, 224, 224};
};

TEST_F(GaussianBlurTest, OutputShapePreserved) {
    GaussianBlur blur(5);
    auto output = blur.get_output_shape(input_shape);
    EXPECT_EQ(output, input_shape);
}

TEST_F(GaussianBlurTest, FixedKernelSize) {
    GaussianBlur blur(7);
    auto [min_k, max_k] = blur.kernel_size();
    EXPECT_EQ(min_k, 7);
    EXPECT_EQ(max_k, 7);
}

TEST_F(GaussianBlurTest, KernelSizeRange) {
    GaussianBlur blur(3, 7);
    auto [min_k, max_k] = blur.kernel_size();
    EXPECT_EQ(min_k, 3);
    EXPECT_EQ(max_k, 7);
}

TEST_F(GaussianBlurTest, KernelSizeInRange) {
    GaussianBlur blur(3, 9);
    blur.set_seed(42);

    std::set<int> seen_sizes;
    for (int i = 0; i < 100; ++i) {
        blur.get_output_shape(input_shape);
        int k = blur.last_kernel_size();

        // Must be in range
        EXPECT_GE(k, 3);
        EXPECT_LE(k, 9);

        // Must be odd
        EXPECT_EQ(k % 2, 1);

        seen_sizes.insert(k);
    }

    // Should have seen multiple sizes: 3, 5, 7, 9
    EXPECT_GT(seen_sizes.size(), 1);
}

TEST_F(GaussianBlurTest, SigmaRange) {
    GaussianBlur blur(5, {0.5f, 2.5f});
    auto sigma = blur.sigma();
    EXPECT_FLOAT_EQ(sigma.first, 0.5f);
    EXPECT_FLOAT_EQ(sigma.second, 2.5f);
}

TEST_F(GaussianBlurTest, SigmaInRange) {
    GaussianBlur blur(5, {0.5f, 2.0f});
    blur.set_seed(42);

    for (int i = 0; i < 100; ++i) {
        blur.get_output_shape(input_shape);
        float s = blur.last_sigma();
        EXPECT_GE(s, 0.5f);
        EXPECT_LE(s, 2.0f);
    }
}

TEST_F(GaussianBlurTest, InvalidKernelSizeEven) {
    EXPECT_THROW(GaussianBlur(4), ValidationError);
}

TEST_F(GaussianBlurTest, InvalidKernelSizeZero) {
    EXPECT_THROW(GaussianBlur(0), ValidationError);
}

TEST_F(GaussianBlurTest, InvalidKernelSizeNegative) {
    EXPECT_THROW(GaussianBlur(-1), ValidationError);
}

TEST_F(GaussianBlurTest, InvalidKernelSizeExceedsMax) {
    EXPECT_THROW(GaussianBlur(33), ResourceError);
}

TEST_F(GaussianBlurTest, InvalidKernelRangeMinGreaterThanMax) {
    EXPECT_THROW(GaussianBlur(7, 3), ValidationError);
}

TEST_F(GaussianBlurTest, InvalidSigmaZero) {
    EXPECT_THROW(GaussianBlur(5, {0.0f, 1.0f}), ValidationError);
}

TEST_F(GaussianBlurTest, InvalidSigmaNegative) {
    EXPECT_THROW(GaussianBlur(5, {-1.0f, 1.0f}), ValidationError);
}

TEST_F(GaussianBlurTest, InvalidSigmaExceedsMax) {
    EXPECT_THROW(GaussianBlur(5, {1.0f, 15.0f}), ResourceError);
}

TEST_F(GaussianBlurTest, InvalidSigmaRangeMinGreaterThanMax) {
    EXPECT_THROW(GaussianBlur(5, {2.0f, 1.0f}), ValidationError);
}

TEST_F(GaussianBlurTest, KernelWeights) {
    GaussianBlur blur(5, {1.0f, 1.0f});  // Fixed sigma for predictable kernel
    blur.get_output_shape(input_shape);

    auto weights = blur.get_kernel_weights();

    // Should have kernel_size elements
    EXPECT_EQ(weights.size(), 5);

    // Weights should sum to 1 (normalized)
    float sum = 0.0f;
    for (float w : weights) {
        sum += w;
        EXPECT_GT(w, 0.0f);  // All weights should be positive
    }
    EXPECT_NEAR(sum, 1.0f, 0.001f);

    // Should be symmetric (Gaussian property)
    EXPECT_FLOAT_EQ(weights[0], weights[4]);
    EXPECT_FLOAT_EQ(weights[1], weights[3]);

    // Center should have highest weight
    EXPECT_GT(weights[2], weights[1]);
    EXPECT_GT(weights[1], weights[0]);
}

TEST_F(GaussianBlurTest, KernelWeightsNotCalledBeforeGetOutputShape) {
    GaussianBlur blur(5);
    // Should throw if get_output_shape wasn't called
    EXPECT_THROW(blur.get_kernel_weights(), ConfigurationError);
}

TEST_F(GaussianBlurTest, HaloSize) {
    GaussianBlur blur(5);
    blur.get_output_shape(input_shape);

    EXPECT_EQ(blur.halo_size(), 2);  // kernel_size / 2 = 5 / 2 = 2
}

TEST_F(GaussianBlurTest, DeterministicWithSeed) {
    GaussianBlur blur1(3, 9);
    GaussianBlur blur2(3, 9);

    blur1.set_seed(11111);
    blur2.set_seed(11111);

    for (int i = 0; i < 20; ++i) {
        blur1.get_output_shape(input_shape);
        blur2.get_output_shape(input_shape);

        EXPECT_EQ(blur1.last_kernel_size(), blur2.last_kernel_size());
        EXPECT_FLOAT_EQ(blur1.last_sigma(), blur2.last_sigma());
    }
}

TEST_F(GaussianBlurTest, Name) {
    GaussianBlur blur(5);
    EXPECT_EQ(blur.name(), "GaussianBlur");
}

TEST_F(GaussianBlurTest, IsNotDeterministic) {
    GaussianBlur blur(5);
    EXPECT_FALSE(blur.is_deterministic());
}

TEST_F(GaussianBlurTest, Repr) {
    GaussianBlur blur(5);
    std::string repr = blur.repr();
    EXPECT_TRUE(repr.find("GaussianBlur") != std::string::npos);
    EXPECT_TRUE(repr.find("kernel_size=5") != std::string::npos);
}

TEST_F(GaussianBlurTest, ReprWithRange) {
    GaussianBlur blur(3, 7);
    std::string repr = blur.repr();
    EXPECT_TRUE(repr.find("(3, 7)") != std::string::npos);
}

// ============================================================================
// Security validation tests
// ============================================================================

class SecurityValidationTest : public ::testing::Test {};

TEST_F(SecurityValidationTest, ValidateRotationAngle) {
    EXPECT_NO_THROW(validate_rotation_angle(0.0f));
    EXPECT_NO_THROW(validate_rotation_angle(360.0f));
    EXPECT_NO_THROW(validate_rotation_angle(-360.0f));
    EXPECT_THROW(validate_rotation_angle(361.0f), ValidationError);
    EXPECT_THROW(validate_rotation_angle(-361.0f), ValidationError);
}

TEST_F(SecurityValidationTest, ValidateBlurKernelSize) {
    EXPECT_NO_THROW(validate_blur_kernel_size(1));
    EXPECT_NO_THROW(validate_blur_kernel_size(3));
    EXPECT_NO_THROW(validate_blur_kernel_size(31));
    EXPECT_THROW(validate_blur_kernel_size(0), ValidationError);
    EXPECT_THROW(validate_blur_kernel_size(-1), ValidationError);
    EXPECT_THROW(validate_blur_kernel_size(2), ValidationError);  // Even
    EXPECT_THROW(validate_blur_kernel_size(33), ResourceError);   // Exceeds max
}

TEST_F(SecurityValidationTest, ValidateBlurSigma) {
    EXPECT_NO_THROW(validate_blur_sigma(0.1f));
    EXPECT_NO_THROW(validate_blur_sigma(10.0f));
    EXPECT_THROW(validate_blur_sigma(0.0f), ValidationError);
    EXPECT_THROW(validate_blur_sigma(-1.0f), ValidationError);
    EXPECT_THROW(validate_blur_sigma(11.0f), ResourceError);
}

TEST_F(SecurityValidationTest, ValidateColorFactor) {
    EXPECT_NO_THROW(validate_color_factor(0.0f, "test"));
    EXPECT_NO_THROW(validate_color_factor(2.0f, "test"));
    EXPECT_THROW(validate_color_factor(-0.1f, "test"), ValidationError);
    EXPECT_THROW(validate_color_factor(2.1f, "test"), ResourceError);
}

TEST_F(SecurityValidationTest, ValidateHueFactor) {
    EXPECT_NO_THROW(validate_hue_factor(0.0f));
    EXPECT_NO_THROW(validate_hue_factor(0.5f));
    EXPECT_NO_THROW(validate_hue_factor(-0.5f));
    EXPECT_THROW(validate_hue_factor(0.6f), ResourceError);
    EXPECT_THROW(validate_hue_factor(-0.6f), ResourceError);
}

TEST_F(SecurityValidationTest, GenerateSecureSeed) {
    // Two secure seeds should be different (with very high probability)
    uint64_t seed1 = generate_secure_seed();
    uint64_t seed2 = generate_secure_seed();
    EXPECT_NE(seed1, seed2);
}

// ============================================================================
// Integration tests with Compose
// ============================================================================

class Phase3ComposeTest : public ::testing::Test {
protected:
    std::vector<int64_t> input_shape = {1, 3, 256, 256};
};

TEST_F(Phase3ComposeTest, DataAugmentationPipeline) {
    // Typical training augmentation pipeline
    auto pipeline = std::vector<std::shared_ptr<Transform>>{
        std::make_shared<RandomHorizontalFlip>(0.5f),
        std::make_shared<RandomRotation>(15.0f),
        std::make_shared<ColorJitter>(0.2f, 0.2f, 0.2f, 0.1f),
        std::make_shared<GaussianBlur>(3, 7)
    };

    // All transforms preserve shape
    auto shape = input_shape;
    for (const auto& transform : pipeline) {
        shape = transform->get_output_shape(shape);
    }

    EXPECT_EQ(shape, input_shape);
}

TEST_F(Phase3ComposeTest, ReproducibleAugmentation) {
    auto flip = std::make_shared<RandomHorizontalFlip>(0.5f);
    auto rot = std::make_shared<RandomRotation>(30.0f);
    auto jitter = std::make_shared<ColorJitter>(0.3f, 0.3f, 0.3f, 0.1f);

    // Set seeds for reproducibility
    flip->set_seed(42);
    rot->set_seed(42);
    jitter->set_seed(42);

    // First pass
    flip->get_output_shape(input_shape);
    rot->get_output_shape(input_shape);
    jitter->get_output_shape(input_shape);

    bool flipped1 = flip->was_flipped();
    float angle1 = rot->last_angle();
    float brightness1 = jitter->last_brightness_factor();

    // Reset seeds
    flip->set_seed(42);
    rot->set_seed(42);
    jitter->set_seed(42);

    // Second pass should produce same results
    flip->get_output_shape(input_shape);
    rot->get_output_shape(input_shape);
    jitter->get_output_shape(input_shape);

    EXPECT_EQ(flipped1, flip->was_flipped());
    EXPECT_FLOAT_EQ(angle1, rot->last_angle());
    EXPECT_FLOAT_EQ(brightness1, jitter->last_brightness_factor());
}

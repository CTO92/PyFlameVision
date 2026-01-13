/**
 * Unit tests for PyFlameVision transforms
 */

#include <gtest/gtest.h>
#include <pyflame_vision/transforms/resize.hpp>
#include <pyflame_vision/transforms/crop.hpp>
#include <pyflame_vision/transforms/normalize.hpp>
#include <pyflame_vision/transforms/compose.hpp>
#include <pyflame_vision/transforms/transform_base.hpp>

using namespace pyflame_vision::transforms;

// Size tests

class SizeTest : public ::testing::Test {};

TEST_F(SizeTest, SquareSize) {
    Size size(224);
    EXPECT_EQ(size.height(), 224);
    EXPECT_EQ(size.width(), 224);
    EXPECT_TRUE(size.is_square());
}

TEST_F(SizeTest, RectangularSize) {
    Size size(480, 640);
    EXPECT_EQ(size.height(), 480);
    EXPECT_EQ(size.width(), 640);
    EXPECT_FALSE(size.is_square());
}

TEST_F(SizeTest, IsValid) {
    EXPECT_TRUE(Size(224).is_valid());
    EXPECT_TRUE(Size(100, 200).is_valid());
    EXPECT_FALSE(Size(0).is_valid());
    EXPECT_FALSE(Size(-1).is_valid());
    EXPECT_FALSE(Size(100, 0).is_valid());
}

// Resize tests

class ResizeTest : public ::testing::Test {
protected:
    std::vector<int64_t> input_shape = {1, 3, 480, 640};
};

TEST_F(ResizeTest, SquareResize) {
    Resize resize(224);
    auto output = resize.get_output_shape(input_shape);

    EXPECT_EQ(output[0], 1);
    EXPECT_EQ(output[1], 3);
    EXPECT_EQ(output[2], 224);
    EXPECT_EQ(output[3], 224);
}

TEST_F(ResizeTest, RectangularResize) {
    Resize resize(Size(256, 512));
    auto output = resize.get_output_shape(input_shape);

    EXPECT_EQ(output[2], 256);
    EXPECT_EQ(output[3], 512);
}

TEST_F(ResizeTest, InterpolationDefault) {
    Resize resize(224);
    EXPECT_EQ(resize.interpolation(), InterpolationMode::BILINEAR);
}

TEST_F(ResizeTest, InterpolationNearest) {
    Resize resize(224, InterpolationMode::NEAREST);
    EXPECT_EQ(resize.interpolation(), InterpolationMode::NEAREST);
}

TEST_F(ResizeTest, AntialiasDefault) {
    Resize resize(224);
    EXPECT_TRUE(resize.antialias());
}

TEST_F(ResizeTest, Name) {
    Resize resize(224);
    EXPECT_EQ(resize.name(), "Resize");
}

TEST_F(ResizeTest, IsDeterministic) {
    Resize resize(224);
    EXPECT_TRUE(resize.is_deterministic());
}

TEST_F(ResizeTest, InvalidInputShape) {
    Resize resize(224);
    std::vector<int64_t> invalid_shape = {3, 224, 224};  // Missing batch dim
    EXPECT_THROW(resize.get_output_shape(invalid_shape), std::runtime_error);
}

// CenterCrop tests

class CenterCropTest : public ::testing::Test {
protected:
    std::vector<int64_t> input_shape = {1, 3, 256, 256};
};

TEST_F(CenterCropTest, SquareCrop) {
    CenterCrop crop(224);
    auto output = crop.get_output_shape(input_shape);

    EXPECT_EQ(output[0], 1);
    EXPECT_EQ(output[1], 3);
    EXPECT_EQ(output[2], 224);
    EXPECT_EQ(output[3], 224);
}

TEST_F(CenterCropTest, RectangularCrop) {
    CenterCrop crop(Size(200, 250));
    auto output = crop.get_output_shape(input_shape);

    EXPECT_EQ(output[2], 200);
    EXPECT_EQ(output[3], 250);
}

TEST_F(CenterCropTest, ComputeBounds) {
    CenterCrop crop(224);
    auto [top, left, height, width] = crop.compute_bounds(256, 256);

    EXPECT_EQ(top, 16);      // (256 - 224) / 2
    EXPECT_EQ(left, 16);
    EXPECT_EQ(height, 224);
    EXPECT_EQ(width, 224);
}

TEST_F(CenterCropTest, CropLargerThanInput) {
    CenterCrop crop(512);  // Larger than input
    EXPECT_THROW(crop.get_output_shape(input_shape), std::runtime_error);
}

TEST_F(CenterCropTest, Name) {
    CenterCrop crop(224);
    EXPECT_EQ(crop.name(), "CenterCrop");
}

TEST_F(CenterCropTest, IsDeterministic) {
    CenterCrop crop(224);
    EXPECT_TRUE(crop.is_deterministic());
}

// RandomCrop tests

class RandomCropTest : public ::testing::Test {
protected:
    std::vector<int64_t> input_shape = {1, 3, 256, 256};
};

TEST_F(RandomCropTest, OutputShape) {
    RandomCrop crop(224);
    auto output = crop.get_output_shape(input_shape);

    EXPECT_EQ(output[2], 224);
    EXPECT_EQ(output[3], 224);
}

TEST_F(RandomCropTest, WithPadding) {
    RandomCrop crop(224, 4);
    EXPECT_EQ(crop.padding(), 4);
    EXPECT_FALSE(crop.pad_if_needed());
}

TEST_F(RandomCropTest, PadIfNeeded) {
    RandomCrop crop(224, 0, true);
    EXPECT_TRUE(crop.pad_if_needed());
}

TEST_F(RandomCropTest, SetSeed) {
    RandomCrop crop(224);
    crop.set_seed(42);
    // Seed should be reproducible
}

TEST_F(RandomCropTest, Name) {
    RandomCrop crop(224);
    EXPECT_EQ(crop.name(), "RandomCrop");
}

TEST_F(RandomCropTest, IsNotDeterministic) {
    RandomCrop crop(224);
    EXPECT_FALSE(crop.is_deterministic());
}

// Normalize tests

class NormalizeTest : public ::testing::Test {
protected:
    std::vector<float> imagenet_mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> imagenet_std = {0.229f, 0.224f, 0.225f};
    std::vector<int64_t> input_shape = {1, 3, 224, 224};
};

TEST_F(NormalizeTest, OutputShapePreserved) {
    Normalize norm(imagenet_mean, imagenet_std);
    auto output = norm.get_output_shape(input_shape);

    EXPECT_EQ(output, input_shape);
}

TEST_F(NormalizeTest, MeanStdAccessors) {
    Normalize norm(imagenet_mean, imagenet_std);

    EXPECT_EQ(norm.mean(), imagenet_mean);
    EXPECT_EQ(norm.std(), imagenet_std);
}

TEST_F(NormalizeTest, InvStd) {
    Normalize norm(imagenet_mean, imagenet_std);
    auto inv_std = norm.inv_std();

    EXPECT_FLOAT_EQ(inv_std[0], 1.0f / 0.229f);
    EXPECT_FLOAT_EQ(inv_std[1], 1.0f / 0.224f);
    EXPECT_FLOAT_EQ(inv_std[2], 1.0f / 0.225f);
}

TEST_F(NormalizeTest, InplaceDefault) {
    Normalize norm(imagenet_mean, imagenet_std);
    EXPECT_FALSE(norm.inplace());
}

TEST_F(NormalizeTest, InplaceTrue) {
    Normalize norm(imagenet_mean, imagenet_std, true);
    EXPECT_TRUE(norm.inplace());
}

TEST_F(NormalizeTest, MismatchedMeanStdLength) {
    std::vector<float> mean = {0.5f, 0.5f};
    std::vector<float> std = {0.5f, 0.5f, 0.5f};
    EXPECT_THROW(Normalize(mean, std), std::invalid_argument);
}

TEST_F(NormalizeTest, EmptyMean) {
    EXPECT_THROW(Normalize({}, {}), std::invalid_argument);
}

TEST_F(NormalizeTest, ZeroStd) {
    EXPECT_THROW(
        Normalize({0.5f, 0.5f, 0.5f}, {0.5f, 0.0f, 0.5f}),
        std::invalid_argument
    );
}

TEST_F(NormalizeTest, NegativeStd) {
    EXPECT_THROW(
        Normalize({0.5f, 0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}),
        std::invalid_argument
    );
}

TEST_F(NormalizeTest, ChannelMismatch) {
    std::vector<float> single_channel_mean = {0.5f};
    std::vector<float> single_channel_std = {0.5f};
    Normalize norm(single_channel_mean, single_channel_std);

    // 3-channel input but 1-channel normalization
    EXPECT_THROW(norm.get_output_shape(input_shape), std::runtime_error);
}

TEST_F(NormalizeTest, Name) {
    Normalize norm(imagenet_mean, imagenet_std);
    EXPECT_EQ(norm.name(), "Normalize");
}

TEST_F(NormalizeTest, IsDeterministic) {
    Normalize norm(imagenet_mean, imagenet_std);
    EXPECT_TRUE(norm.is_deterministic());
}

// Compose tests

class ComposeTest : public ::testing::Test {
protected:
    std::vector<int64_t> input_shape = {1, 3, 480, 640};
};

TEST_F(ComposeTest, SingleTransform) {
    Compose pipeline({
        std::make_shared<Resize>(224)
    });

    auto output = pipeline.get_output_shape(input_shape);
    EXPECT_EQ(output[2], 224);
    EXPECT_EQ(output[3], 224);
}

TEST_F(ComposeTest, MultipleTransforms) {
    Compose pipeline({
        std::make_shared<Resize>(256),
        std::make_shared<CenterCrop>(224)
    });

    auto output = pipeline.get_output_shape(input_shape);
    EXPECT_EQ(output[2], 224);
    EXPECT_EQ(output[3], 224);
}

TEST_F(ComposeTest, ImageNetPipeline) {
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};

    Compose pipeline({
        std::make_shared<Resize>(256),
        std::make_shared<CenterCrop>(224),
        std::make_shared<Normalize>(mean, std)
    });

    auto output = pipeline.get_output_shape(input_shape);
    EXPECT_EQ(output[0], 1);
    EXPECT_EQ(output[1], 3);
    EXPECT_EQ(output[2], 224);
    EXPECT_EQ(output[3], 224);
}

TEST_F(ComposeTest, EmptyPipeline) {
    Compose pipeline({});
    EXPECT_TRUE(pipeline.empty());

    auto output = pipeline.get_output_shape(input_shape);
    EXPECT_EQ(output, input_shape);  // Passthrough
}

TEST_F(ComposeTest, Length) {
    Compose pipeline({
        std::make_shared<Resize>(256),
        std::make_shared<CenterCrop>(224)
    });

    EXPECT_EQ(pipeline.size(), 2);
}

TEST_F(ComposeTest, IndexAccess) {
    auto resize = std::make_shared<Resize>(256);
    auto crop = std::make_shared<CenterCrop>(224);

    Compose pipeline({resize, crop});

    EXPECT_EQ(pipeline[0]->name(), "Resize");
    EXPECT_EQ(pipeline[1]->name(), "CenterCrop");
}

TEST_F(ComposeTest, Transforms) {
    Compose pipeline({
        std::make_shared<Resize>(256),
        std::make_shared<CenterCrop>(224)
    });

    auto transforms = pipeline.transforms();
    EXPECT_EQ(transforms.size(), 2);
}

TEST_F(ComposeTest, DeterministicPipeline) {
    Compose pipeline({
        std::make_shared<Resize>(256),
        std::make_shared<CenterCrop>(224)
    });

    EXPECT_TRUE(pipeline.is_deterministic());
}

TEST_F(ComposeTest, NonDeterministicPipeline) {
    Compose pipeline({
        std::make_shared<Resize>(256),
        std::make_shared<RandomCrop>(224)
    });

    EXPECT_FALSE(pipeline.is_deterministic());
}

TEST_F(ComposeTest, Name) {
    Compose pipeline({});
    EXPECT_EQ(pipeline.name(), "Compose");
}

TEST_F(ComposeTest, NullTransformThrows) {
    EXPECT_THROW(
        Compose({nullptr}),
        std::invalid_argument
    );
}

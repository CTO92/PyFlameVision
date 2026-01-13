/**
 * Unit tests for ImageTensor utilities
 */

#include <gtest/gtest.h>
#include <pyflame_vision/core/image_tensor.hpp>

using namespace pyflame_vision::core;

class ImageTensorTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

// Shape validation tests

TEST_F(ImageTensorTest, ValidShape) {
    std::vector<int64_t> shape = {1, 3, 224, 224};
    EXPECT_TRUE(ImageTensor::is_valid_shape(shape));
}

TEST_F(ImageTensorTest, InvalidShapeTooFewDims) {
    std::vector<int64_t> shape = {3, 224, 224};
    EXPECT_FALSE(ImageTensor::is_valid_shape(shape));
}

TEST_F(ImageTensorTest, InvalidShapeTooManyDims) {
    std::vector<int64_t> shape = {1, 1, 3, 224, 224};
    EXPECT_FALSE(ImageTensor::is_valid_shape(shape));
}

TEST_F(ImageTensorTest, InvalidShapeZeroDim) {
    std::vector<int64_t> shape = {1, 0, 224, 224};
    EXPECT_FALSE(ImageTensor::is_valid_shape(shape));
}

TEST_F(ImageTensorTest, InvalidShapeNegativeDim) {
    std::vector<int64_t> shape = {1, 3, -1, 224};
    EXPECT_FALSE(ImageTensor::is_valid_shape(shape));
}

TEST_F(ImageTensorTest, ValidateShapeThrowsOnInvalid) {
    std::vector<int64_t> shape = {1, 3, 224};
    EXPECT_THROW(ImageTensor::validate_shape(shape), std::runtime_error);
}

// Dimension accessor tests

TEST_F(ImageTensorTest, GetDimensions) {
    std::vector<int64_t> shape = {4, 3, 480, 640};
    auto [batch, channels, height, width] = ImageTensor::get_dimensions(shape);

    EXPECT_EQ(batch, 4);
    EXPECT_EQ(channels, 3);
    EXPECT_EQ(height, 480);
    EXPECT_EQ(width, 640);
}

TEST_F(ImageTensorTest, BatchSize) {
    std::vector<int64_t> shape = {8, 3, 224, 224};
    EXPECT_EQ(ImageTensor::batch_size(shape), 8);
}

TEST_F(ImageTensorTest, NumChannels) {
    std::vector<int64_t> shape = {1, 1, 224, 224};  // Grayscale
    EXPECT_EQ(ImageTensor::num_channels(shape), 1);
}

TEST_F(ImageTensorTest, Height) {
    std::vector<int64_t> shape = {1, 3, 480, 640};
    EXPECT_EQ(ImageTensor::height(shape), 480);
}

TEST_F(ImageTensorTest, Width) {
    std::vector<int64_t> shape = {1, 3, 480, 640};
    EXPECT_EQ(ImageTensor::width(shape), 640);
}

// Output shape tests

TEST_F(ImageTensorTest, ResizeOutputShape) {
    std::vector<int64_t> input_shape = {1, 3, 480, 640};
    auto output = ImageTensor::resize_output_shape(input_shape, 224, 224);

    EXPECT_EQ(output[0], 1);    // batch preserved
    EXPECT_EQ(output[1], 3);    // channels preserved
    EXPECT_EQ(output[2], 224);  // new height
    EXPECT_EQ(output[3], 224);  // new width
}

TEST_F(ImageTensorTest, CropOutputShape) {
    std::vector<int64_t> input_shape = {2, 3, 256, 256};
    auto output = ImageTensor::crop_output_shape(input_shape, 224, 224);

    EXPECT_EQ(output[0], 2);
    EXPECT_EQ(output[1], 3);
    EXPECT_EQ(output[2], 224);
    EXPECT_EQ(output[3], 224);
}

// Memory calculations

TEST_F(ImageTensorTest, Numel) {
    std::vector<int64_t> shape = {1, 3, 224, 224};
    EXPECT_EQ(ImageTensor::numel(shape), 1 * 3 * 224 * 224);
}

TEST_F(ImageTensorTest, SizeBytes) {
    std::vector<int64_t> shape = {1, 3, 224, 224};
    // Default element size is 4 bytes (float32)
    EXPECT_EQ(ImageTensor::size_bytes(shape), 1 * 3 * 224 * 224 * 4);
}

TEST_F(ImageTensorTest, SizeBytesWithElementSize) {
    std::vector<int64_t> shape = {1, 3, 224, 224};
    // float16 = 2 bytes
    EXPECT_EQ(ImageTensor::size_bytes(shape, 2), 1 * 3 * 224 * 224 * 2);
}

// Layout tests

TEST_F(ImageTensorTest, OptimalLayoutSmallImage) {
    // Small image should fit on single PE
    auto layout = ImageTensor::optimal_layout(32, 32);
    EXPECT_EQ(layout.type, pyflame::MeshLayout::Type::SINGLE_PE);
    EXPECT_EQ(layout.total_pes(), 1);
}

TEST_F(ImageTensorTest, OptimalLayoutLargeImage) {
    // Large image should be distributed
    auto layout = ImageTensor::optimal_layout(1024, 1024);
    EXPECT_EQ(layout.type, pyflame::MeshLayout::Type::GRID);
    EXPECT_GT(layout.total_pes(), 1);
}

TEST_F(ImageTensorTest, TileShapeSinglePE) {
    pyflame::MeshLayout layout = pyflame::MeshLayout::SinglePE();
    auto [h, w] = ImageTensor::tile_shape(224, 224, layout, 0, 0);

    EXPECT_EQ(h, 224);
    EXPECT_EQ(w, 224);
}

TEST_F(ImageTensorTest, TileShapeGrid) {
    pyflame::MeshLayout layout = pyflame::MeshLayout::Grid(2, 2);
    auto [h, w] = ImageTensor::tile_shape(224, 224, layout, 0, 0);

    EXPECT_EQ(h, 112);  // 224 / 2
    EXPECT_EQ(w, 112);
}

// Halo tests

TEST_F(ImageTensorTest, NeedsHaloKernel1) {
    EXPECT_FALSE(ImageTensor::needs_halo(1));
}

TEST_F(ImageTensorTest, NeedsHaloKernel3) {
    EXPECT_TRUE(ImageTensor::needs_halo(3));
}

TEST_F(ImageTensorTest, HaloSizeKernel3) {
    EXPECT_EQ(ImageTensor::halo_size(3), 1);  // 3 / 2 = 1
}

TEST_F(ImageTensorTest, HaloSizeKernel5) {
    EXPECT_EQ(ImageTensor::halo_size(5), 2);  // 5 / 2 = 2
}

// ColorSpace tests

TEST_F(ImageTensorTest, ColorSpaceNames) {
    EXPECT_EQ(colorspace_name(ColorSpace::RGB), "RGB");
    EXPECT_EQ(colorspace_name(ColorSpace::BGR), "BGR");
    EXPECT_EQ(colorspace_name(ColorSpace::GRAY), "GRAY");
    EXPECT_EQ(colorspace_name(ColorSpace::HSV), "HSV");
    EXPECT_EQ(colorspace_name(ColorSpace::LAB), "LAB");
}

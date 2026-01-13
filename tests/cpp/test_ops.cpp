/**
 * @file test_ops.cpp
 * @brief Unit tests for Phase 4 specialized operations
 *
 * Tests for GridSample, ROIAlign, and NMS operations.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include "pyflame_vision/ops/ops.hpp"

using namespace pyflame_vision::ops;
using namespace pyflame_vision::core;
using namespace pyflame_vision::transforms;

// ============================================================================
// GridSample Tests
// ============================================================================

TEST(GridSampleTest, DefaultConstruction) {
    GridSample gs;
    EXPECT_EQ(gs.mode(), InterpolationMode::BILINEAR);
    EXPECT_EQ(gs.padding_mode(), PaddingMode::ZEROS);
    EXPECT_FALSE(gs.align_corners());
}

TEST(GridSampleTest, ConstructionWithParams) {
    GridSample gs(InterpolationMode::NEAREST, PaddingMode::BORDER, true);
    EXPECT_EQ(gs.mode(), InterpolationMode::NEAREST);
    EXPECT_EQ(gs.padding_mode(), PaddingMode::BORDER);
    EXPECT_TRUE(gs.align_corners());
}

TEST(GridSampleTest, InvalidModeThrows) {
    EXPECT_THROW(
        GridSample(InterpolationMode::BICUBIC),
        ValidationError
    );
    EXPECT_THROW(
        GridSample(InterpolationMode::AREA),
        ValidationError
    );
}

TEST(GridSampleTest, OutputShape) {
    GridSample gs;
    std::vector<int64_t> input_shape = {2, 3, 64, 64};
    std::vector<int64_t> grid_shape = {2, 32, 32, 2};

    auto output = gs.get_output_shape(input_shape, grid_shape);

    EXPECT_EQ(output.size(), 4);
    EXPECT_EQ(output[0], 2);   // batch
    EXPECT_EQ(output[1], 3);   // channels
    EXPECT_EQ(output[2], 32);  // H_out
    EXPECT_EQ(output[3], 32);  // W_out
}

TEST(GridSampleTest, BatchMismatchThrows) {
    GridSample gs;
    std::vector<int64_t> input_shape = {2, 3, 64, 64};
    std::vector<int64_t> grid_shape = {4, 32, 32, 2};  // Wrong batch

    EXPECT_THROW(
        gs.get_output_shape(input_shape, grid_shape),
        ValidationError
    );
}

TEST(GridSampleTest, InvalidGridLastDim) {
    GridSample gs;
    std::vector<int64_t> input_shape = {2, 3, 64, 64};
    std::vector<int64_t> grid_shape = {2, 32, 32, 3};  // Should be 2

    EXPECT_THROW(
        gs.get_output_shape(input_shape, grid_shape),
        ValidationError
    );
}

TEST(GridSampleTest, InvalidInputShape) {
    GridSample gs;
    std::vector<int64_t> input_shape = {2, 3, 64};  // 3D instead of 4D
    std::vector<int64_t> grid_shape = {2, 32, 32, 2};

    EXPECT_THROW(
        gs.get_output_shape(input_shape, grid_shape),
        ValidationError
    );
}

TEST(GridSampleTest, HaloSize) {
    GridSample bilinear(InterpolationMode::BILINEAR);
    GridSample nearest(InterpolationMode::NEAREST);

    EXPECT_EQ(bilinear.halo_size(), 1);
    EXPECT_EQ(nearest.halo_size(), 0);
}

TEST(GridSampleTest, Repr) {
    GridSample gs(InterpolationMode::BILINEAR, PaddingMode::REFLECTION, true);
    std::string repr = gs.repr();

    EXPECT_NE(repr.find("bilinear"), std::string::npos);
    EXPECT_NE(repr.find("reflection"), std::string::npos);
    EXPECT_NE(repr.find("True"), std::string::npos);
}

// ============================================================================
// ROI Tests
// ============================================================================

TEST(ROITest, Construction) {
    ROI roi{0, 10.0f, 20.0f, 30.0f, 50.0f};
    EXPECT_EQ(roi.batch_index, 0);
    EXPECT_FLOAT_EQ(roi.x1, 10.0f);
    EXPECT_FLOAT_EQ(roi.y1, 20.0f);
    EXPECT_FLOAT_EQ(roi.x2, 30.0f);
    EXPECT_FLOAT_EQ(roi.y2, 50.0f);
}

TEST(ROITest, WidthHeight) {
    ROI roi{0, 10.0f, 20.0f, 30.0f, 50.0f};
    EXPECT_FLOAT_EQ(roi.width(), 20.0f);
    EXPECT_FLOAT_EQ(roi.height(), 30.0f);
}

TEST(ROITest, Area) {
    ROI roi{0, 10.0f, 20.0f, 30.0f, 50.0f};
    EXPECT_FLOAT_EQ(roi.area(), 20.0f * 30.0f);
}

TEST(ROITest, InvalidBatchIndex) {
    ROI roi{-1, 0, 0, 10, 10};
    EXPECT_THROW(roi.validate(), ValidationError);
}

TEST(ROITest, InvalidCoordinates) {
    ROI roi{0, 30.0f, 50.0f, 10.0f, 20.0f};  // x2 < x1
    EXPECT_THROW(roi.validate(), ValidationError);
}

TEST(ROITest, NaNCoordinates) {
    ROI roi{0, std::numeric_limits<float>::quiet_NaN(), 0, 10, 10};
    EXPECT_THROW(roi.validate(), ValidationError);
}

// ============================================================================
// ROIAlign Tests
// ============================================================================

TEST(ROIAlignTest, Construction) {
    ROIAlign ra(7, 7, 1.0f / 16.0f);
    EXPECT_EQ(ra.output_height(), 7);
    EXPECT_EQ(ra.output_width(), 7);
    EXPECT_FLOAT_EQ(ra.spatial_scale(), 1.0f / 16.0f);
    EXPECT_EQ(ra.sampling_ratio(), 0);
    EXPECT_TRUE(ra.aligned());
}

TEST(ROIAlignTest, SquareConstruction) {
    ROIAlign ra(7, 0.25f, 2, false);
    EXPECT_EQ(ra.output_height(), 7);
    EXPECT_EQ(ra.output_width(), 7);
    EXPECT_FLOAT_EQ(ra.spatial_scale(), 0.25f);
    EXPECT_EQ(ra.sampling_ratio(), 2);
    EXPECT_FALSE(ra.aligned());
}

TEST(ROIAlignTest, InvalidOutputSizeThrows) {
    EXPECT_THROW(ROIAlign(0, 7, 0.25f), ValidationError);
    EXPECT_THROW(ROIAlign(-1, 7, 0.25f), ValidationError);
    EXPECT_THROW(ROIAlign(7, 0, 0.25f), ValidationError);
}

TEST(ROIAlignTest, InvalidSpatialScaleThrows) {
    EXPECT_THROW(ROIAlign(7, 7, 0.0f), ValidationError);
    EXPECT_THROW(ROIAlign(7, 7, -0.5f), ValidationError);
    EXPECT_THROW(
        ROIAlign(7, 7, std::numeric_limits<float>::infinity()),
        ValidationError
    );
    EXPECT_THROW(
        ROIAlign(7, 7, std::numeric_limits<float>::quiet_NaN()),
        ValidationError
    );
}

TEST(ROIAlignTest, InvalidSamplingRatioThrows) {
    EXPECT_THROW(ROIAlign(7, 7, 0.25f, -1), ValidationError);
}

TEST(ROIAlignTest, OutputShape) {
    ROIAlign ra(7, 7, 0.25f);
    std::vector<int64_t> input_shape = {1, 256, 50, 50};

    auto output = ra.get_output_shape(input_shape, 100);

    EXPECT_EQ(output.size(), 4);
    EXPECT_EQ(output[0], 100);  // num_rois
    EXPECT_EQ(output[1], 256);  // channels
    EXPECT_EQ(output[2], 7);    // output_height
    EXPECT_EQ(output[3], 7);    // output_width
}

TEST(ROIAlignTest, TooManyROIsThrows) {
    ROIAlign ra(7, 7, 0.25f);
    std::vector<int64_t> input_shape = {1, 256, 50, 50};

    EXPECT_THROW(
        ra.get_output_shape(input_shape, 20000),  // Exceeds MAX_ROIS
        ResourceError
    );
}

TEST(ROIAlignTest, InvalidInputShapeThrows) {
    ROIAlign ra(7, 7, 0.25f);
    std::vector<int64_t> input_shape = {1, 256, 50};  // 3D instead of 4D

    EXPECT_THROW(ra.get_output_shape(input_shape, 10), ValidationError);
}

TEST(ROIAlignTest, HaloSize) {
    ROIAlign ra(7, 7, 0.25f);
    EXPECT_EQ(ra.halo_size(), 1);
}

TEST(ROIAlignTest, Repr) {
    ROIAlign ra(7, 14, 0.25f, 2, true);
    std::string repr = ra.repr();

    EXPECT_NE(repr.find("7"), std::string::npos);
    EXPECT_NE(repr.find("14"), std::string::npos);
    EXPECT_NE(repr.find("0.25"), std::string::npos);
}

// ============================================================================
// DetectionBox Tests
// ============================================================================

TEST(DetectionBoxTest, Construction) {
    DetectionBox box{10.0f, 20.0f, 30.0f, 50.0f, 0.9f, 1};
    EXPECT_FLOAT_EQ(box.x1, 10.0f);
    EXPECT_FLOAT_EQ(box.y1, 20.0f);
    EXPECT_FLOAT_EQ(box.x2, 30.0f);
    EXPECT_FLOAT_EQ(box.y2, 50.0f);
    EXPECT_FLOAT_EQ(box.score, 0.9f);
    EXPECT_EQ(box.class_id, 1);
}

TEST(DetectionBoxTest, Area) {
    DetectionBox box{10.0f, 20.0f, 30.0f, 50.0f, 0.9f, 1};
    EXPECT_FLOAT_EQ(box.area(), 20.0f * 30.0f);
}

TEST(DetectionBoxTest, ZeroArea) {
    DetectionBox box{10.0f, 20.0f, 10.0f, 20.0f, 0.9f, 1};  // Point
    EXPECT_FLOAT_EQ(box.area(), 0.0f);
}

TEST(DetectionBoxTest, IoU) {
    DetectionBox box1{0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 1};
    DetectionBox box2{5.0f, 5.0f, 15.0f, 15.0f, 0.8f, 1};

    // Intersection: [5,5] to [10,10] = 25
    // Union: 100 + 100 - 25 = 175
    float expected_iou = 25.0f / 175.0f;
    EXPECT_NEAR(box1.iou(box2), expected_iou, 1e-5f);
}

TEST(DetectionBoxTest, NoOverlap) {
    DetectionBox box1{0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 1};
    DetectionBox box2{20.0f, 20.0f, 30.0f, 30.0f, 0.8f, 1};

    EXPECT_FLOAT_EQ(box1.iou(box2), 0.0f);
}

TEST(DetectionBoxTest, FullOverlap) {
    DetectionBox box1{0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 1};
    DetectionBox box2{0.0f, 0.0f, 10.0f, 10.0f, 0.8f, 1};

    EXPECT_FLOAT_EQ(box1.iou(box2), 1.0f);
}

TEST(DetectionBoxTest, ValidationNaNScore) {
    DetectionBox box{0.0f, 0.0f, 10.0f, 10.0f,
        std::numeric_limits<float>::quiet_NaN(), 1};
    EXPECT_THROW(box.validate(), ValidationError);
}

// ============================================================================
// NMS Tests
// ============================================================================

TEST(NMSTest, Construction) {
    NMS nms(0.5f);
    EXPECT_FLOAT_EQ(nms.iou_threshold(), 0.5f);
}

TEST(NMSTest, InvalidThresholdThrows) {
    EXPECT_THROW(NMS(-0.1f), ValidationError);
    EXPECT_THROW(NMS(1.1f), ValidationError);
    EXPECT_THROW(
        NMS(std::numeric_limits<float>::quiet_NaN()),
        ValidationError
    );
}

TEST(NMSTest, BoundaryThresholds) {
    EXPECT_NO_THROW(NMS(0.0f));
    EXPECT_NO_THROW(NMS(1.0f));
}

TEST(NMSTest, MaxOutputSize) {
    NMS nms(0.5f);
    EXPECT_EQ(nms.max_output_size(100), 100);
    EXPECT_EQ(nms.max_output_size(0), 0);
}

TEST(NMSTest, TooManyBoxesThrows) {
    NMS nms(0.5f);
    EXPECT_THROW(
        nms.max_output_size(200000),  // Exceeds MAX_NMS_BOXES
        ResourceError
    );
}

TEST(NMSTest, NegativeBoxCountThrows) {
    NMS nms(0.5f);
    EXPECT_THROW(nms.max_output_size(-1), ValidationError);
}

TEST(NMSTest, Repr) {
    NMS nms(0.45f);
    std::string repr = nms.repr();
    EXPECT_NE(repr.find("0.45"), std::string::npos);
}

// ============================================================================
// BatchedNMS Tests
// ============================================================================

TEST(BatchedNMSTest, Construction) {
    BatchedNMS bnms(0.5f);
    EXPECT_FLOAT_EQ(bnms.iou_threshold(), 0.5f);
}

TEST(BatchedNMSTest, InvalidThresholdThrows) {
    EXPECT_THROW(BatchedNMS(-0.1f), ValidationError);
    EXPECT_THROW(BatchedNMS(1.5f), ValidationError);
}

TEST(BatchedNMSTest, MaxOutputSize) {
    BatchedNMS bnms(0.5f);
    EXPECT_EQ(bnms.max_output_size(100), 100);
}

// ============================================================================
// SoftNMS Tests
// ============================================================================

TEST(SoftNMSTest, DefaultConstruction) {
    SoftNMS snms;
    EXPECT_FLOAT_EQ(snms.sigma(), 0.5f);
    EXPECT_FLOAT_EQ(snms.iou_threshold(), 0.3f);
    EXPECT_FLOAT_EQ(snms.score_threshold(), 0.001f);
    EXPECT_EQ(snms.method(), SoftNMS::Method::GAUSSIAN);
}

TEST(SoftNMSTest, LinearMethod) {
    SoftNMS snms(0.5f, 0.3f, 0.001f, SoftNMS::Method::LINEAR);
    EXPECT_EQ(snms.method(), SoftNMS::Method::LINEAR);
}

TEST(SoftNMSTest, InvalidSigmaThrows) {
    EXPECT_THROW(SoftNMS(0.0f), ValidationError);
    EXPECT_THROW(SoftNMS(-0.5f), ValidationError);
    EXPECT_THROW(
        SoftNMS(std::numeric_limits<float>::quiet_NaN()),
        ValidationError
    );
}

TEST(SoftNMSTest, MaxOutputSize) {
    SoftNMS snms;
    EXPECT_EQ(snms.max_output_size(100), 100);
}

TEST(SoftNMSTest, Repr) {
    SoftNMS snms(0.5f, 0.3f, 0.001f, SoftNMS::Method::GAUSSIAN);
    std::string repr = snms.repr();
    EXPECT_NE(repr.find("gaussian"), std::string::npos);
}

// ============================================================================
// Interpolation Utilities Tests
// ============================================================================

TEST(InterpolationTest, PaddingModeName) {
    EXPECT_EQ(padding_mode_name(PaddingMode::ZEROS), "zeros");
    EXPECT_EQ(padding_mode_name(PaddingMode::BORDER), "border");
    EXPECT_EQ(padding_mode_name(PaddingMode::REFLECTION), "reflection");
}

TEST(InterpolationTest, PaddingModeFromString) {
    EXPECT_EQ(padding_mode_from_string("zeros"), PaddingMode::ZEROS);
    EXPECT_EQ(padding_mode_from_string("border"), PaddingMode::BORDER);
    EXPECT_EQ(padding_mode_from_string("reflection"), PaddingMode::REFLECTION);
}

TEST(InterpolationTest, InvalidPaddingModeThrows) {
    EXPECT_THROW(padding_mode_from_string("invalid"), ValidationError);
}

TEST(InterpolationTest, BicubicWeight) {
    // At t=0, weight should be 1
    EXPECT_NEAR(bicubic_weight(0.0f), 1.0f, 1e-5f);

    // At t=1 or t=-1, weight should be 0
    EXPECT_NEAR(bicubic_weight(1.0f), 0.0f, 1e-5f);
    EXPECT_NEAR(bicubic_weight(-1.0f), 0.0f, 1e-5f);

    // At t>=2 or t<=-2, weight should be 0
    EXPECT_FLOAT_EQ(bicubic_weight(2.0f), 0.0f);
    EXPECT_FLOAT_EQ(bicubic_weight(-2.0f), 0.0f);
    EXPECT_FLOAT_EQ(bicubic_weight(3.0f), 0.0f);
}

TEST(InterpolationTest, BicubicWeights) {
    float weights[4];
    bicubic_weights(0.5f, weights);

    // All weights should be finite
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(std::isfinite(weights[i]));
    }

    // Sum should be approximately 1 (partition of unity)
    float sum = weights[0] + weights[1] + weights[2] + weights[3];
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

// ============================================================================
// Security Validation Tests
// ============================================================================

TEST(SecurityTest, ValidateROIAlignParams) {
    EXPECT_NO_THROW(validate_roi_align_params(7, 7, 0.25f, 0));
    EXPECT_NO_THROW(validate_roi_align_params(7, 7, 0.25f, 2));
}

TEST(SecurityTest, ValidateGridCoordinate) {
    EXPECT_NO_THROW(validate_grid_coordinate(0.0f, "test"));
    EXPECT_NO_THROW(validate_grid_coordinate(100.0f, "test"));
    EXPECT_NO_THROW(validate_grid_coordinate(-100.0f, "test"));

    EXPECT_THROW(
        validate_grid_coordinate(std::numeric_limits<float>::quiet_NaN(), "test"),
        ValidationError
    );
    EXPECT_THROW(
        validate_grid_coordinate(std::numeric_limits<float>::infinity(), "test"),
        ValidationError
    );
}

TEST(SecurityTest, ValidateNMSParams) {
    EXPECT_NO_THROW(validate_nms_params(0.5f, 0.0f));
    EXPECT_NO_THROW(validate_nms_params(0.0f, 0.0f));
    EXPECT_NO_THROW(validate_nms_params(1.0f, 0.0f));

    EXPECT_THROW(validate_nms_params(-0.1f, 0.0f), ValidationError);
    EXPECT_THROW(validate_nms_params(1.1f, 0.0f), ValidationError);
}

TEST(SecurityTest, ValidateROICount) {
    EXPECT_NO_THROW(validate_roi_count(0));
    EXPECT_NO_THROW(validate_roi_count(100));
    EXPECT_NO_THROW(validate_roi_count(10000));

    EXPECT_THROW(validate_roi_count(-1), ValidationError);
    EXPECT_THROW(validate_roi_count(20000), ResourceError);
}

TEST(SecurityTest, ValidateNMSBoxCount) {
    EXPECT_NO_THROW(validate_nms_box_count(0));
    EXPECT_NO_THROW(validate_nms_box_count(100000));

    EXPECT_THROW(validate_nms_box_count(-1), ValidationError);
    EXPECT_THROW(validate_nms_box_count(200000), ResourceError);
}

TEST(SecurityTest, NegativeScoreThresholdThrows) {
    // Score threshold must be non-negative
    EXPECT_THROW(validate_nms_params(0.5f, -0.1f), ValidationError);
    EXPECT_NO_THROW(validate_nms_params(0.5f, 0.0f));
    EXPECT_NO_THROW(validate_nms_params(0.5f, 0.001f));
}

// ============================================================================
// Factory Function Tests
// ============================================================================

TEST(DetectionBoxTest, CreateFactory) {
    // Valid creation
    EXPECT_NO_THROW(DetectionBox::create(0, 0, 10, 10, 0.9f, 0));

    // Invalid coordinates
    EXPECT_THROW(
        DetectionBox::create(0, 0, 1e10f, 10, 0.9f, 0),  // Exceeds MAX_GRID_COORDINATE
        ResourceError
    );

    // NaN score
    EXPECT_THROW(
        DetectionBox::create(0, 0, 10, 10, std::numeric_limits<float>::quiet_NaN(), 0),
        ValidationError
    );
}

TEST(ROITest, CreateFactory) {
    // Valid creation
    EXPECT_NO_THROW(ROI::create(0, 0, 0, 100, 100));

    // Invalid batch index
    EXPECT_THROW(ROI::create(-1, 0, 0, 10, 10), ValidationError);

    // Invalid coordinates (x2 < x1)
    EXPECT_THROW(ROI::create(0, 100, 0, 10, 10), ValidationError);

    // NaN coordinate
    EXPECT_THROW(
        ROI::create(0, std::numeric_limits<float>::quiet_NaN(), 0, 10, 10),
        ValidationError
    );

    // Coordinate exceeds limit
    EXPECT_THROW(ROI::create(0, 0, 0, 2e6f, 10), ResourceError);
}

TEST(ROIAlignTest, TotalElementsValidation) {
    ROIAlign ra(7, 7, 0.25f);
    std::vector<int64_t> input_shape = {1, 256, 50, 50};

    // Valid: 10000 ROIs * 256 channels * 7 * 7 = 125M elements (under limit)
    EXPECT_NO_THROW(ra.get_output_shape(input_shape, 10000));

    // This test verifies that the validation runs
    // The actual limit depends on MAX_BATCH_SIZE which applies to num_rois
}

TEST(SoftNMSTest, NegativeScoreThresholdThrows) {
    EXPECT_THROW(SoftNMS(0.5f, 0.3f, -0.001f), ValidationError);
    EXPECT_NO_THROW(SoftNMS(0.5f, 0.3f, 0.0f));
}

// ============================================================================
// CSL Template Security Tests
// ============================================================================

#include "pyflame_vision/backend/csl_generator.hpp"

using namespace pyflame_vision::backend;
using namespace pyflame_vision::core::template_security;

TEST(TemplateSecurityTest, ValidNumericValues) {
    EXPECT_NO_THROW(validate_numeric_value("123"));
    EXPECT_NO_THROW(validate_numeric_value("3.14159"));
    EXPECT_NO_THROW(validate_numeric_value("-42"));
    EXPECT_NO_THROW(validate_numeric_value("1.5e-10"));
    EXPECT_NO_THROW(validate_numeric_value("1.0f"));
}

TEST(TemplateSecurityTest, InvalidNumericValues) {
    // Code injection attempts
    EXPECT_THROW(validate_numeric_value("123; rm -rf /"), TemplateError);
    EXPECT_THROW(validate_numeric_value("0; @import"), TemplateError);
    EXPECT_THROW(validate_numeric_value("1 + 1"), TemplateError);
    EXPECT_THROW(validate_numeric_value("$(calc)"), TemplateError);
    EXPECT_THROW(validate_numeric_value(""), TemplateError);
}

TEST(TemplateSecurityTest, ValidIdentifiers) {
    EXPECT_NO_THROW(validate_identifier_value("width"));
    EXPECT_NO_THROW(validate_identifier_value("HEIGHT"));
    EXPECT_NO_THROW(validate_identifier_value("_private"));
    EXPECT_NO_THROW(validate_identifier_value("var123"));
}

TEST(TemplateSecurityTest, InvalidIdentifiers) {
    EXPECT_THROW(validate_identifier_value("123start"), TemplateError);
    EXPECT_THROW(validate_identifier_value("with-dash"), TemplateError);
    EXPECT_THROW(validate_identifier_value("with space"), TemplateError);
    EXPECT_THROW(validate_identifier_value("@import"), TemplateError);
    EXPECT_THROW(validate_identifier_value(""), TemplateError);
}

TEST(TemplateSecurityTest, ValidDtypes) {
    EXPECT_NO_THROW(validate_dtype_value("f32"));
    EXPECT_NO_THROW(validate_dtype_value("f16"));
    EXPECT_NO_THROW(validate_dtype_value("i32"));
    EXPECT_NO_THROW(validate_dtype_value("u8"));
    EXPECT_NO_THROW(validate_dtype_value("bool"));
}

TEST(TemplateSecurityTest, InvalidDtypes) {
    EXPECT_THROW(validate_dtype_value("float"), TemplateError);
    EXPECT_THROW(validate_dtype_value("f64"), TemplateError);
    EXPECT_THROW(validate_dtype_value("int"), TemplateError);
    EXPECT_THROW(validate_dtype_value("f32; malicious"), TemplateError);
}

TEST(TemplateSecurityTest, ValidParamNames) {
    EXPECT_NO_THROW(validate_param_name("width"));
    EXPECT_NO_THROW(validate_param_name("HEIGHT"));
    EXPECT_NO_THROW(validate_param_name("num_channels"));
}

TEST(TemplateSecurityTest, InvalidParamNames) {
    EXPECT_THROW(validate_param_name(""), TemplateError);
    EXPECT_THROW(validate_param_name("with-dash"), TemplateError);
    EXPECT_THROW(validate_param_name("with space"), TemplateError);
}

TEST(CSLGeneratorTest, BasicSubstitution) {
    CSLGenerator gen;
    gen.set_numeric("width", 256);
    gen.set_numeric("height", 256);
    gen.set_dtype("dtype", "f32");
    gen.set_boolean("aligned", true);

    std::string tmpl = "const W: u16 = ${width};\nconst H: u16 = ${height};\nconst T = ${dtype};\nconst ALIGNED: bool = ${aligned};";
    std::string result = gen.generate(tmpl);

    EXPECT_NE(result.find("const W: u16 = 256"), std::string::npos);
    EXPECT_NE(result.find("const H: u16 = 256"), std::string::npos);
    EXPECT_NE(result.find("const T = f32"), std::string::npos);
    EXPECT_NE(result.find("const ALIGNED: bool = true"), std::string::npos);
}

TEST(CSLGeneratorTest, MissingPlaceholderThrows) {
    CSLGenerator gen;
    gen.set_numeric("width", 256);
    // height is NOT set

    std::string tmpl = "const W: u16 = ${width};\nconst H: u16 = ${height};";

    EXPECT_THROW(gen.generate(tmpl), TemplateError);
}

TEST(CSLGeneratorTest, RejectsCodeInjection) {
    CSLGenerator gen;

    // These should fail validation
    EXPECT_THROW(
        gen.set_param("bad", TemplateParam::numeric("123; @import")),
        TemplateError
    );
    EXPECT_THROW(
        gen.set_param("bad", TemplateParam::identifier("sys; rm")),
        TemplateError
    );
    EXPECT_THROW(
        gen.set_param("bad", TemplateParam::dtype("f32; @import")),
        TemplateError
    );
}

TEST(CSLBuilderTest, FluentInterface) {
    std::string result = CSLBuilder()
        .set("width", 128)
        .set("height", 128)
        .set("spatial_scale", 0.25f)
        .set_bool("aligned", false)
        .set_dtype("dtype", "f32")
        .generate("W=${width} H=${height} S=${spatial_scale} A=${aligned} T=${dtype}");

    EXPECT_NE(result.find("W=128"), std::string::npos);
    EXPECT_NE(result.find("H=128"), std::string::npos);
    EXPECT_NE(result.find("A=false"), std::string::npos);
    EXPECT_NE(result.find("T=f32"), std::string::npos);
}

TEST(CSLGeneratorTest, MultiplePlaceholderInstances) {
    CSLGenerator gen;
    gen.set_numeric("size", 64);

    std::string tmpl = "const A: u16 = ${size};\nconst B: u16 = ${size};\nconst C: u16 = ${size};";
    std::string result = gen.generate(tmpl);

    // Count occurrences of "64"
    size_t count = 0;
    size_t pos = 0;
    while ((pos = result.find("64", pos)) != std::string::npos) {
        ++count;
        pos += 2;
    }
    EXPECT_EQ(count, 3);
}

TEST(CSLGeneratorTest, FloatParameterValidation) {
    CSLGenerator gen;

    // Valid floats
    EXPECT_NO_THROW(gen.set_numeric("scale", 0.25f));
    EXPECT_NO_THROW(gen.set_numeric("factor", -1.5f));

    // NaN/Inf should throw
    EXPECT_THROW(
        gen.set_numeric("bad", std::numeric_limits<float>::quiet_NaN()),
        TemplateError
    );
    EXPECT_THROW(
        gen.set_numeric("bad", std::numeric_limits<float>::infinity()),
        TemplateError
    );
}

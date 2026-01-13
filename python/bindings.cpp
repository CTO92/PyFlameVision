#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "pyflame_vision/pyflame_vision.hpp"
#include "pyflame_vision/core/exceptions.hpp"
#include "pyflame_vision/core/security.hpp"
#include "pyflame_vision/nn/nn.hpp"
#include "pyflame_vision/models/models.hpp"

// Phase 3: Advanced Transforms
#include "pyflame_vision/transforms/random_transform.hpp"
#include "pyflame_vision/transforms/flip.hpp"
#include "pyflame_vision/transforms/rotation.hpp"
#include "pyflame_vision/transforms/color_jitter.hpp"
#include "pyflame_vision/transforms/blur.hpp"

// Phase 4: Specialized Operations
#include "pyflame_vision/ops/ops.hpp"

namespace py = pybind11;

using namespace pyflame_vision;
using namespace pyflame_vision::transforms;
using namespace pyflame_vision::core;
using namespace pyflame_vision::nn;
using namespace pyflame_vision::models;

// Helper to validate size values from Python with clear error messages
static void validate_python_size(int64_t value, const char* name) {
    if (value <= 0) {
        throw ValidationError(
            std::string(name) + " must be positive, got " + std::to_string(value)
        );
    }
    if (value > SecurityLimits::MAX_DIMENSION) {
        throw ResourceError(
            std::string(name) + " (" + std::to_string(value) +
            ") exceeds maximum allowed (" +
            std::to_string(SecurityLimits::MAX_DIMENSION) + ")"
        );
    }
}

// Helper to validate floating-point values from Python (NaN/Inf check)
static void validate_python_float(float value, const char* name) {
    if (!std::isfinite(value)) {
        throw ValidationError(
            std::string(name) + " must be finite, got NaN or Inf"
        );
    }
}

// Helper to parse and validate size from Python object
static Size parse_size_with_validation(py::object size_obj) {
    if (py::isinstance<py::int_>(size_obj)) {
        int64_t size = size_obj.cast<int64_t>();
        validate_python_size(size, "size");
        return Size(size);
    } else if (py::isinstance<py::tuple>(size_obj)) {
        auto tup = size_obj.cast<std::tuple<int64_t, int64_t>>();
        int64_t h = std::get<0>(tup);
        int64_t w = std::get<1>(tup);
        validate_python_size(h, "height");
        validate_python_size(w, "width");
        return Size(h, w);
    } else if (py::isinstance<py::list>(size_obj)) {
        auto lst = size_obj.cast<std::vector<int64_t>>();
        if (lst.size() == 1) {
            validate_python_size(lst[0], "size");
            return Size(lst[0]);
        } else if (lst.size() == 2) {
            validate_python_size(lst[0], "height");
            validate_python_size(lst[1], "width");
            return Size(lst[0], lst[1]);
        } else {
            throw ValidationError(
                "Size list must have 1 or 2 elements, got " + std::to_string(lst.size())
            );
        }
    } else {
        throw ValidationError("Size must be int, tuple(height, width), or list");
    }
}

PYBIND11_MODULE(_pyflame_vision_cpp, m) {
    m.doc() = "PyFlameVision: Cerebras-native computer vision library";

    // ========================================================================
    // Exception registration
    // ========================================================================
    // Register custom exceptions so they translate properly to Python
    static py::exception<Exception> exc_base(m, "PyFlameVisionError");
    static py::exception<ValidationError> exc_validation(m, "ValidationError", exc_base.ptr());
    static py::exception<BoundsError> exc_bounds(m, "BoundsError", exc_base.ptr());
    static py::exception<OverflowError> exc_overflow(m, "OverflowError", exc_base.ptr());
    static py::exception<ConfigurationError> exc_config(m, "ConfigurationError", exc_base.ptr());
    static py::exception<ResourceError> exc_resource(m, "ResourceError", exc_base.ptr());
    static py::exception<TemplateError> exc_template(m, "TemplateError", exc_base.ptr());

    // ========================================================================
    // Version info
    // ========================================================================
    m.attr("__version__") = PYFLAME_VISION_VERSION_STRING;
    m.attr("version_major") = PYFLAME_VISION_VERSION_MAJOR;
    m.attr("version_minor") = PYFLAME_VISION_VERSION_MINOR;
    m.attr("version_patch") = PYFLAME_VISION_VERSION_PATCH;

    // ========================================================================
    // Core module
    // ========================================================================
    auto core_module = m.def_submodule("core", "Core utilities");

    // InterpolationMode enum
    py::enum_<InterpolationMode>(core_module, "InterpolationMode")
        .value("NEAREST", InterpolationMode::NEAREST)
        .value("BILINEAR", InterpolationMode::BILINEAR)
        .value("BICUBIC", InterpolationMode::BICUBIC)
        .value("AREA", InterpolationMode::AREA)
        .export_values();

    // ColorSpace enum
    py::enum_<ColorSpace>(core_module, "ColorSpace")
        .value("RGB", ColorSpace::RGB)
        .value("BGR", ColorSpace::BGR)
        .value("GRAY", ColorSpace::GRAY)
        .value("HSV", ColorSpace::HSV)
        .value("LAB", ColorSpace::LAB)
        .export_values();

    // ImageTensor utilities (static methods)
    py::class_<ImageTensor>(core_module, "ImageTensor")
        .def_static("is_valid_shape", &ImageTensor::is_valid_shape,
            "Check if shape represents a valid NCHW image")
        .def_static("validate_shape", &ImageTensor::validate_shape,
            "Validate shape and throw if invalid")
        .def_static("batch_size", &ImageTensor::batch_size,
            "Get batch size from shape")
        .def_static("num_channels", &ImageTensor::num_channels,
            "Get number of channels from shape")
        .def_static("height", &ImageTensor::height,
            "Get height from shape")
        .def_static("width", &ImageTensor::width,
            "Get width from shape")
        .def_static("numel", &ImageTensor::numel,
            "Get total number of elements");

    // ========================================================================
    // Transforms module
    // ========================================================================
    auto transforms_module = m.def_submodule("transforms", "Image transforms");

    // Size class
    py::class_<Size>(transforms_module, "Size")
        .def(py::init<int64_t>(), py::arg("size"),
            "Create square size")
        .def(py::init<int64_t, int64_t>(), py::arg("height"), py::arg("width"),
            "Create rectangular size")
        .def_readonly("height", &Size::height)
        .def_readonly("width", &Size::width)
        .def("is_valid", &Size::is_valid)
        .def("__repr__", &Size::to_string);

    // Base Transform class
    py::class_<Transform, std::shared_ptr<Transform>>(transforms_module, "Transform")
        .def("get_output_shape", &Transform::get_output_shape,
            "Get output shape for given input shape")
        .def("name", &Transform::name,
            "Get transform name")
        .def("is_deterministic", &Transform::is_deterministic,
            "Check if transform is deterministic")
        .def("__repr__", &Transform::repr);

    // Resize
    py::class_<Resize, Transform, std::shared_ptr<Resize>>(transforms_module, "Resize")
        .def(py::init([](py::object size_obj, std::string interpolation, bool antialias) {
            Size size = parse_size_with_validation(size_obj);
            InterpolationMode mode = interpolation_from_string(interpolation);
            return std::make_shared<Resize>(size, mode, antialias);
        }),
        py::arg("size"),
        py::arg("interpolation") = "bilinear",
        py::arg("antialias") = true,
        "Resize image to given size")
        .def("interpolation", &Resize::interpolation)
        .def("antialias", &Resize::antialias)
        .def("halo_size", &Resize::halo_size);

    // CenterCrop
    py::class_<CenterCrop, Transform, std::shared_ptr<CenterCrop>>(transforms_module, "CenterCrop")
        .def(py::init([](py::object size_obj) {
            Size size = parse_size_with_validation(size_obj);
            return std::make_shared<CenterCrop>(size);
        }),
        py::arg("size"),
        "Center crop image to given size")
        .def("compute_bounds", &CenterCrop::compute_bounds,
            py::arg("input_height"), py::arg("input_width"),
            "Compute crop bounds for given input size");

    // RandomCrop
    py::class_<RandomCrop, Transform, std::shared_ptr<RandomCrop>>(transforms_module, "RandomCrop")
        .def(py::init([](py::object size_obj, int padding, bool pad_if_needed, float fill) {
            Size size = parse_size_with_validation(size_obj);
            // Validate padding at the Python boundary for clearer error messages
            if (padding < 0) {
                throw ValidationError(
                    "Padding must be non-negative, got " + std::to_string(padding)
                );
            }
            if (padding > SecurityLimits::MAX_PADDING) {
                throw ResourceError(
                    "Padding (" + std::to_string(padding) +
                    ") exceeds maximum allowed (" +
                    std::to_string(SecurityLimits::MAX_PADDING) + ")"
                );
            }
            return std::make_shared<RandomCrop>(size, padding, pad_if_needed, fill);
        }),
        py::arg("size"),
        py::arg("padding") = 0,
        py::arg("pad_if_needed") = false,
        py::arg("fill") = 0.0f,
        "Random crop image to given size")
        .def("set_seed", &RandomCrop::set_seed, py::arg("seed"),
            "Set random seed for reproducibility")
        .def("padding", &RandomCrop::padding)
        .def("pad_if_needed", &RandomCrop::pad_if_needed);

    // Normalize
    py::class_<Normalize, Transform, std::shared_ptr<Normalize>>(transforms_module, "Normalize")
        .def(py::init<std::vector<float>, std::vector<float>, bool>(),
            py::arg("mean"),
            py::arg("std"),
            py::arg("inplace") = false,
            "Normalize tensor with mean and std")
        .def("mean", &Normalize::mean)
        .def("std", &Normalize::std)
        .def("inv_std", &Normalize::inv_std)
        .def("inplace", &Normalize::inplace);

    // Compose
    py::class_<Compose, Transform, std::shared_ptr<Compose>>(transforms_module, "Compose")
        .def(py::init<std::vector<std::shared_ptr<Transform>>>(),
            py::arg("transforms"),
            "Compose multiple transforms into a pipeline")
        .def("__len__", &Compose::size)
        .def("__getitem__", &Compose::get)
        .def("transforms", &Compose::transforms)
        .def("empty", &Compose::empty);

    // ========================================================================
    // Phase 3: Advanced Transforms (Data Augmentation)
    // ========================================================================

    // RandomHorizontalFlip
    py::class_<RandomHorizontalFlip, Transform, std::shared_ptr<RandomHorizontalFlip>>(
        transforms_module, "RandomHorizontalFlip")
        .def(py::init([](float p) {
            validate_python_float(p, "probability");
            return std::make_shared<RandomHorizontalFlip>(p);
        }), py::arg("p") = 0.5f,
            "Randomly flip image horizontally with given probability")
        .def("set_seed", &RandomHorizontalFlip::set_seed, py::arg("seed"),
            "Set random seed for reproducibility")
        .def_property_readonly("p", &RandomHorizontalFlip::probability)
        .def("was_flipped", &RandomHorizontalFlip::was_flipped,
            "Check if last call resulted in a flip");

    // RandomVerticalFlip
    py::class_<RandomVerticalFlip, Transform, std::shared_ptr<RandomVerticalFlip>>(
        transforms_module, "RandomVerticalFlip")
        .def(py::init([](float p) {
            validate_python_float(p, "probability");
            return std::make_shared<RandomVerticalFlip>(p);
        }), py::arg("p") = 0.5f,
            "Randomly flip image vertically with given probability")
        .def("set_seed", &RandomVerticalFlip::set_seed, py::arg("seed"),
            "Set random seed for reproducibility")
        .def_property_readonly("p", &RandomVerticalFlip::probability)
        .def("was_flipped", &RandomVerticalFlip::was_flipped,
            "Check if last call resulted in a flip");

    // RotationFillMode enum
    py::enum_<RotationFillMode>(transforms_module, "RotationFillMode")
        .value("CONSTANT", RotationFillMode::CONSTANT)
        .value("REFLECT", RotationFillMode::REFLECT)
        .value("REPLICATE", RotationFillMode::REPLICATE)
        .export_values();

    // RandomRotation
    py::class_<RandomRotation, Transform, std::shared_ptr<RandomRotation>>(
        transforms_module, "RandomRotation")
        .def(py::init([](py::object degrees, std::string interpolation, bool expand,
                         py::object center, py::object fill) {
            // Parse degrees
            float deg_min, deg_max;
            if (py::isinstance<py::float_>(degrees) || py::isinstance<py::int_>(degrees)) {
                float d = degrees.cast<float>();
                validate_python_float(d, "degrees");
                deg_min = -std::abs(d);
                deg_max = std::abs(d);
            } else if (py::isinstance<py::tuple>(degrees) || py::isinstance<py::list>(degrees)) {
                auto tup = degrees.cast<std::tuple<float, float>>();
                deg_min = std::get<0>(tup);
                deg_max = std::get<1>(tup);
                validate_python_float(deg_min, "degrees min");
                validate_python_float(deg_max, "degrees max");
            } else {
                throw ValidationError("degrees must be float or tuple(min, max)");
            }

            // Parse interpolation
            InterpolationMode mode = interpolation_from_string(interpolation);

            // Parse center
            std::optional<std::pair<float, float>> ctr = std::nullopt;
            if (!center.is_none()) {
                auto c = center.cast<std::tuple<float, float>>();
                ctr = std::make_pair(std::get<0>(c), std::get<1>(c));
            }

            // Parse fill
            std::vector<float> fill_vals = {0.0f};
            if (!fill.is_none()) {
                if (py::isinstance<py::float_>(fill) || py::isinstance<py::int_>(fill)) {
                    fill_vals = {fill.cast<float>()};
                } else {
                    fill_vals = fill.cast<std::vector<float>>();
                }
            }

            return std::make_shared<RandomRotation>(
                deg_min, deg_max, mode, expand, ctr, fill_vals
            );
        }),
        py::arg("degrees"),
        py::arg("interpolation") = "bilinear",
        py::arg("expand") = false,
        py::arg("center") = py::none(),
        py::arg("fill") = py::none(),
        "Randomly rotate image by angle within degrees range")
        .def("set_seed", &RandomRotation::set_seed, py::arg("seed"),
            "Set random seed for reproducibility")
        .def("degrees", &RandomRotation::degrees,
            "Get angle range (min, max)")
        .def("last_angle", &RandomRotation::last_angle,
            "Get last applied rotation angle")
        .def_property_readonly("interpolation", &RandomRotation::interpolation)
        .def_property_readonly("expand", &RandomRotation::expand)
        .def("halo_size", &RandomRotation::halo_size);

    // ColorJitter
    py::class_<ColorJitter, Transform, std::shared_ptr<ColorJitter>>(
        transforms_module, "ColorJitter")
        .def(py::init([](float brightness, float contrast, float saturation, float hue) {
            validate_python_float(brightness, "brightness");
            validate_python_float(contrast, "contrast");
            validate_python_float(saturation, "saturation");
            validate_python_float(hue, "hue");
            return std::make_shared<ColorJitter>(brightness, contrast, saturation, hue);
        }),
            py::arg("brightness") = 0.0f,
            py::arg("contrast") = 0.0f,
            py::arg("saturation") = 0.0f,
            py::arg("hue") = 0.0f,
            "Randomly adjust brightness, contrast, saturation, and hue")
        .def(py::init([](py::object brightness, py::object contrast,
                         py::object saturation, py::object hue) {
            // Parse brightness
            std::pair<float, float> b_range = {1.0f, 1.0f};
            if (!brightness.is_none()) {
                if (py::isinstance<py::float_>(brightness) || py::isinstance<py::int_>(brightness)) {
                    float v = brightness.cast<float>();
                    validate_python_float(v, "brightness");
                    b_range = {std::max(0.0f, 1.0f - v), 1.0f + v};
                } else {
                    auto tup = brightness.cast<std::tuple<float, float>>();
                    validate_python_float(std::get<0>(tup), "brightness min");
                    validate_python_float(std::get<1>(tup), "brightness max");
                    b_range = {std::get<0>(tup), std::get<1>(tup)};
                }
            }

            // Parse contrast
            std::pair<float, float> c_range = {1.0f, 1.0f};
            if (!contrast.is_none()) {
                if (py::isinstance<py::float_>(contrast) || py::isinstance<py::int_>(contrast)) {
                    float v = contrast.cast<float>();
                    validate_python_float(v, "contrast");
                    c_range = {std::max(0.0f, 1.0f - v), 1.0f + v};
                } else {
                    auto tup = contrast.cast<std::tuple<float, float>>();
                    validate_python_float(std::get<0>(tup), "contrast min");
                    validate_python_float(std::get<1>(tup), "contrast max");
                    c_range = {std::get<0>(tup), std::get<1>(tup)};
                }
            }

            // Parse saturation
            std::pair<float, float> s_range = {1.0f, 1.0f};
            if (!saturation.is_none()) {
                if (py::isinstance<py::float_>(saturation) || py::isinstance<py::int_>(saturation)) {
                    float v = saturation.cast<float>();
                    validate_python_float(v, "saturation");
                    s_range = {std::max(0.0f, 1.0f - v), 1.0f + v};
                } else {
                    auto tup = saturation.cast<std::tuple<float, float>>();
                    validate_python_float(std::get<0>(tup), "saturation min");
                    validate_python_float(std::get<1>(tup), "saturation max");
                    s_range = {std::get<0>(tup), std::get<1>(tup)};
                }
            }

            // Parse hue
            std::pair<float, float> h_range = {0.0f, 0.0f};
            if (!hue.is_none()) {
                if (py::isinstance<py::float_>(hue) || py::isinstance<py::int_>(hue)) {
                    float v = hue.cast<float>();
                    validate_python_float(v, "hue");
                    h_range = {-v, v};
                } else {
                    auto tup = hue.cast<std::tuple<float, float>>();
                    validate_python_float(std::get<0>(tup), "hue min");
                    validate_python_float(std::get<1>(tup), "hue max");
                    h_range = {std::get<0>(tup), std::get<1>(tup)};
                }
            }

            return std::make_shared<ColorJitter>(b_range, c_range, s_range, h_range);
        }),
        py::arg("brightness") = py::none(),
        py::arg("contrast") = py::none(),
        py::arg("saturation") = py::none(),
        py::arg("hue") = py::none(),
        "Create ColorJitter with explicit ranges")
        .def("set_seed", &ColorJitter::set_seed, py::arg("seed"),
            "Set random seed for reproducibility")
        .def("brightness", &ColorJitter::brightness)
        .def("contrast", &ColorJitter::contrast)
        .def("saturation", &ColorJitter::saturation)
        .def("hue", &ColorJitter::hue)
        .def("last_brightness_factor", &ColorJitter::last_brightness_factor)
        .def("last_contrast_factor", &ColorJitter::last_contrast_factor)
        .def("last_saturation_factor", &ColorJitter::last_saturation_factor)
        .def("last_hue_factor", &ColorJitter::last_hue_factor)
        .def("last_order", &ColorJitter::last_order);

    // GaussianBlur
    py::class_<GaussianBlur, Transform, std::shared_ptr<GaussianBlur>>(
        transforms_module, "GaussianBlur")
        .def(py::init([](py::object kernel_size, py::object sigma) {
            int ks_min, ks_max;
            if (py::isinstance<py::int_>(kernel_size)) {
                ks_min = ks_max = kernel_size.cast<int>();
            } else if (py::isinstance<py::tuple>(kernel_size) || py::isinstance<py::list>(kernel_size)) {
                auto tup = kernel_size.cast<std::tuple<int, int>>();
                ks_min = std::get<0>(tup);
                ks_max = std::get<1>(tup);
            } else {
                throw ValidationError("kernel_size must be int or tuple(min, max)");
            }

            std::pair<float, float> sig = {0.1f, 2.0f};
            if (!sigma.is_none()) {
                if (py::isinstance<py::float_>(sigma) || py::isinstance<py::int_>(sigma)) {
                    float s = sigma.cast<float>();
                    validate_python_float(s, "sigma");
                    sig = {s, s};
                } else if (py::isinstance<py::tuple>(sigma) || py::isinstance<py::list>(sigma)) {
                    auto tup = sigma.cast<std::tuple<float, float>>();
                    validate_python_float(std::get<0>(tup), "sigma min");
                    validate_python_float(std::get<1>(tup), "sigma max");
                    sig = {std::get<0>(tup), std::get<1>(tup)};
                }
            }

            return std::make_shared<GaussianBlur>(ks_min, ks_max, sig);
        }),
        py::arg("kernel_size"),
        py::arg("sigma") = py::none(),
        "Apply Gaussian blur with configurable kernel size and sigma")
        .def("set_seed", &GaussianBlur::set_seed, py::arg("seed"),
            "Set random seed for reproducibility")
        .def("kernel_size", &GaussianBlur::kernel_size,
            "Get kernel size range (min, max)")
        .def("sigma", &GaussianBlur::sigma,
            "Get sigma range (min, max)")
        .def("last_kernel_size", &GaussianBlur::last_kernel_size,
            "Get last applied kernel size")
        .def("last_sigma", &GaussianBlur::last_sigma,
            "Get last applied sigma")
        .def("halo_size", &GaussianBlur::halo_size,
            "Get halo size for distributed execution")
        .def("get_kernel_weights", &GaussianBlur::get_kernel_weights,
            "Get 1D Gaussian kernel weights");

    // ========================================================================
    // Functional API
    // ========================================================================
    auto functional_module = m.def_submodule("functional", "Functional transform API");

    functional_module.def("resize_output_shape",
        [](const std::vector<int64_t>& input_shape, std::tuple<int64_t, int64_t> size) {
            return functional::resize_output_shape(input_shape, size);
        },
        py::arg("input_shape"), py::arg("size"),
        "Compute output shape for resize operation");

    functional_module.def("crop_output_shape",
        &functional::crop_output_shape,
        py::arg("input_shape"), py::arg("top"), py::arg("left"),
        py::arg("height"), py::arg("width"),
        "Compute output shape for crop operation");

    functional_module.def("center_crop_output_shape",
        [](const std::vector<int64_t>& input_shape, std::tuple<int64_t, int64_t> size) {
            return functional::center_crop_output_shape(input_shape, size);
        },
        py::arg("input_shape"), py::arg("size"),
        "Compute output shape for center crop operation");

    functional_module.def("normalize_output_shape",
        &functional::normalize_output_shape,
        py::arg("input_shape"), py::arg("mean"), py::arg("std"),
        "Compute output shape for normalize operation");

    functional_module.def("compute_center_crop_bounds",
        &functional::compute_center_crop_bounds,
        py::arg("input_height"), py::arg("input_width"),
        py::arg("crop_height"), py::arg("crop_width"),
        "Compute center crop bounds (top, left, height, width)");

    functional_module.def("compute_resize_scale",
        &functional::compute_resize_scale,
        py::arg("src_height"), py::arg("src_width"),
        py::arg("dst_height"), py::arg("dst_width"),
        "Compute resize scale factors (scale_y, scale_x)");

    // ========================================================================
    // Neural Network (nn) module
    // ========================================================================
    auto nn_module = m.def_submodule("nn", "Neural network layers");

    // TensorSpec
    py::class_<TensorSpec>(nn_module, "TensorSpec")
        .def(py::init<std::vector<int64_t>, std::string>(),
            py::arg("shape"), py::arg("dtype") = "float32")
        .def_readwrite("shape", &TensorSpec::shape)
        .def_readwrite("dtype", &TensorSpec::dtype)
        .def("numel", &TensorSpec::numel, "Total number of elements")
        .def("size_bytes", &TensorSpec::size_bytes, "Size in bytes")
        .def("__repr__", [](const TensorSpec& self) {
            std::ostringstream ss;
            ss << "TensorSpec(shape=[";
            for (size_t i = 0; i < self.shape.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << self.shape[i];
            }
            ss << "], dtype='" << self.dtype << "')";
            return ss.str();
        });

    // Parameter
    py::class_<Parameter>(nn_module, "Parameter")
        .def_readonly("name", &Parameter::name)
        .def_readonly("spec", &Parameter::spec)
        .def_readonly("requires_grad", &Parameter::requires_grad);

    // Module base class
    py::class_<Module, std::shared_ptr<Module>>(nn_module, "Module")
        .def("forward", &Module::forward, py::arg("input"),
            "Compute output shape for given input")
        .def("get_output_shape", [](const Module& self, const std::vector<int64_t>& shape) {
            TensorSpec input{shape, "float32"};
            auto output = self.forward(input);
            return output.shape;
        }, py::arg("input_shape"), "Get output shape for input shape")
        .def("name", &Module::name)
        .def("parameters", &Module::parameters)
        .def("named_parameters", &Module::named_parameters, py::arg("prefix") = "")
        .def("train", &Module::train, py::arg("mode") = true)
        .def("eval", &Module::eval)
        .def("is_training", &Module::is_training)
        .def("__repr__", &Module::repr);

    // Sequential
    py::class_<Sequential, Module, std::shared_ptr<Sequential>>(nn_module, "Sequential")
        .def(py::init<>())
        .def(py::init<std::vector<std::shared_ptr<Module>>>(), py::arg("modules"))
        .def("add", &Sequential::add, py::arg("module"))
        .def("__len__", &Sequential::size)
        .def("__getitem__", &Sequential::operator[])
        .def("empty", &Sequential::empty);

    // Conv2d
    py::class_<Conv2d, Module, std::shared_ptr<Conv2d>>(nn_module, "Conv2d")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool, PaddingMode>(),
            py::arg("in_channels"),
            py::arg("out_channels"),
            py::arg("kernel_size"),
            py::arg("stride") = 1,
            py::arg("padding") = 0,
            py::arg("dilation") = 1,
            py::arg("groups") = 1,
            py::arg("bias") = true,
            py::arg("padding_mode") = PaddingMode::ZEROS)
        .def_property_readonly("in_channels", &Conv2d::in_channels)
        .def_property_readonly("out_channels", &Conv2d::out_channels)
        .def_property_readonly("kernel_size", &Conv2d::kernel_size)
        .def_property_readonly("stride", &Conv2d::stride)
        .def_property_readonly("padding", &Conv2d::padding)
        .def_property_readonly("dilation", &Conv2d::dilation)
        .def_property_readonly("groups", &Conv2d::groups)
        .def("has_bias", &Conv2d::has_bias)
        .def("is_depthwise", &Conv2d::is_depthwise)
        .def("is_pointwise", &Conv2d::is_pointwise)
        .def("halo_size", &Conv2d::halo_size);

    // PaddingMode enum
    py::enum_<PaddingMode>(nn_module, "PaddingMode")
        .value("ZEROS", PaddingMode::ZEROS)
        .value("REFLECT", PaddingMode::REFLECT)
        .value("REPLICATE", PaddingMode::REPLICATE)
        .value("CIRCULAR", PaddingMode::CIRCULAR)
        .export_values();

    // BatchNorm2d
    py::class_<BatchNorm2d, Module, std::shared_ptr<BatchNorm2d>>(nn_module, "BatchNorm2d")
        .def(py::init<int64_t, double, double, bool, bool>(),
            py::arg("num_features"),
            py::arg("eps") = 1e-5,
            py::arg("momentum") = 0.1,
            py::arg("affine") = true,
            py::arg("track_running_stats") = true)
        .def_property_readonly("num_features", &BatchNorm2d::num_features)
        .def_property_readonly("eps", &BatchNorm2d::eps)
        .def_property_readonly("momentum", &BatchNorm2d::momentum)
        .def_property_readonly("affine", &BatchNorm2d::affine)
        .def_property_readonly("track_running_stats", &BatchNorm2d::track_running_stats);

    // ReLU
    py::class_<ReLU, Module, std::shared_ptr<ReLU>>(nn_module, "ReLU")
        .def(py::init<bool>(), py::arg("inplace") = false)
        .def_property_readonly("inplace", &ReLU::inplace);

    // SiLU
    py::class_<SiLU, Module, std::shared_ptr<SiLU>>(nn_module, "SiLU")
        .def(py::init<bool>(), py::arg("inplace") = false)
        .def_property_readonly("inplace", &SiLU::inplace);

    // Sigmoid
    py::class_<Sigmoid, Module, std::shared_ptr<Sigmoid>>(nn_module, "Sigmoid")
        .def(py::init<>());

    // Identity
    py::class_<Identity, Module, std::shared_ptr<Identity>>(nn_module, "Identity")
        .def(py::init<>());

    // Flatten
    py::class_<Flatten, Module, std::shared_ptr<Flatten>>(nn_module, "Flatten")
        .def(py::init<int64_t, int64_t>(),
            py::arg("start_dim") = 1, py::arg("end_dim") = -1)
        .def_property_readonly("start_dim", &Flatten::start_dim)
        .def_property_readonly("end_dim", &Flatten::end_dim);

    // MaxPool2d
    py::class_<MaxPool2d, Module, std::shared_ptr<MaxPool2d>>(nn_module, "MaxPool2d")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, bool>(),
            py::arg("kernel_size"),
            py::arg("stride") = 0,
            py::arg("padding") = 0,
            py::arg("dilation") = 1,
            py::arg("ceil_mode") = false)
        .def_property_readonly("kernel_size", &MaxPool2d::kernel_size)
        .def_property_readonly("stride", &MaxPool2d::stride)
        .def_property_readonly("padding", &MaxPool2d::padding)
        .def_property_readonly("dilation", &MaxPool2d::dilation)
        .def_property_readonly("ceil_mode", &MaxPool2d::ceil_mode)
        .def("halo_size", &MaxPool2d::halo_size);

    // AvgPool2d
    py::class_<AvgPool2d, Module, std::shared_ptr<AvgPool2d>>(nn_module, "AvgPool2d")
        .def(py::init<int64_t, int64_t, int64_t, bool, bool>(),
            py::arg("kernel_size"),
            py::arg("stride") = 0,
            py::arg("padding") = 0,
            py::arg("ceil_mode") = false,
            py::arg("count_include_pad") = true)
        .def_property_readonly("kernel_size", &AvgPool2d::kernel_size)
        .def_property_readonly("stride", &AvgPool2d::stride)
        .def_property_readonly("padding", &AvgPool2d::padding)
        .def_property_readonly("ceil_mode", &AvgPool2d::ceil_mode)
        .def_property_readonly("count_include_pad", &AvgPool2d::count_include_pad);

    // AdaptiveAvgPool2d
    py::class_<AdaptiveAvgPool2d, Module, std::shared_ptr<AdaptiveAvgPool2d>>(nn_module, "AdaptiveAvgPool2d")
        .def(py::init<int64_t>(), py::arg("output_size"))
        .def_property_readonly("output_size", &AdaptiveAvgPool2d::output_size);

    // Linear
    py::class_<Linear, Module, std::shared_ptr<Linear>>(nn_module, "Linear")
        .def(py::init<int64_t, int64_t, bool>(),
            py::arg("in_features"),
            py::arg("out_features"),
            py::arg("bias") = true)
        .def_property_readonly("in_features", &Linear::in_features)
        .def_property_readonly("out_features", &Linear::out_features)
        .def("has_bias", &Linear::has_bias);

    // ========================================================================
    // Models module
    // ========================================================================
    auto models_module = m.def_submodule("models", "Model architectures");

    // ResNetBlockType enum
    py::enum_<ResNetBlockType>(models_module, "ResNetBlockType")
        .value("BASIC", ResNetBlockType::BASIC)
        .value("BOTTLENECK", ResNetBlockType::BOTTLENECK)
        .export_values();

    // ResNet
    py::class_<ResNet, Module, std::shared_ptr<ResNet>>(models_module, "ResNet")
        .def(py::init<ResNetBlockType, std::vector<int64_t>, int64_t, bool, int64_t, int64_t>(),
            py::arg("block_type"),
            py::arg("layers"),
            py::arg("num_classes") = 1000,
            py::arg("zero_init_residual") = false,
            py::arg("groups") = 1,
            py::arg("width_per_group") = 64)
        .def("forward_features", &ResNet::forward_features, py::arg("input"))
        .def("remove_fc", &ResNet::remove_fc)
        .def("has_fc", &ResNet::has_fc)
        .def("num_features", &ResNet::num_features)
        .def_property_readonly("block_type", &ResNet::block_type)
        .def_property_readonly("layer_config", &ResNet::layer_config);

    // ResNet factory functions
    models_module.def("resnet18", &resnet18,
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "Create ResNet-18 model");
    models_module.def("resnet34", &resnet34,
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "Create ResNet-34 model");
    models_module.def("resnet50", &resnet50,
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "Create ResNet-50 model");
    models_module.def("resnet101", &resnet101,
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "Create ResNet-101 model");
    models_module.def("resnet152", &resnet152,
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "Create ResNet-152 model");
    models_module.def("resnext50_32x4d", &resnext50_32x4d,
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "Create ResNeXt-50 (32x4d) model");
    models_module.def("resnext101_32x8d", &resnext101_32x8d,
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "Create ResNeXt-101 (32x8d) model");
    models_module.def("wide_resnet50_2", &wide_resnet50_2,
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "Create Wide ResNet-50-2 model");
    models_module.def("wide_resnet101_2", &wide_resnet101_2,
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "Create Wide ResNet-101-2 model");

    // EfficientNet
    py::class_<EfficientNet, Module, std::shared_ptr<EfficientNet>>(models_module, "EfficientNet")
        .def("forward_features", &EfficientNet::forward_features, py::arg("input"))
        .def("remove_classifier", &EfficientNet::remove_classifier)
        .def("has_classifier", &EfficientNet::has_classifier)
        .def("num_features", &EfficientNet::num_features);

    // EfficientNet factory functions
    models_module.def("efficientnet_b0", &efficientnet_b0,
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "Create EfficientNet-B0 model");
    models_module.def("efficientnet_b1", &efficientnet_b1,
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "Create EfficientNet-B1 model");
    models_module.def("efficientnet_b2", &efficientnet_b2,
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "Create EfficientNet-B2 model");
    models_module.def("efficientnet_b3", &efficientnet_b3,
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "Create EfficientNet-B3 model");
    models_module.def("efficientnet_b4", &efficientnet_b4,
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "Create EfficientNet-B4 model");

    // ========================================================================
    // Phase 4: Specialized Operations module
    // ========================================================================
    auto ops_module = m.def_submodule("ops", "Specialized operations for detection and spatial transformers");

    // Interpolation mode (expose from core if not already)
    py::enum_<core::InterpolationMode>(ops_module, "InterpolationMode")
        .value("NEAREST", core::InterpolationMode::NEAREST)
        .value("BILINEAR", core::InterpolationMode::BILINEAR)
        .value("BICUBIC", core::InterpolationMode::BICUBIC)
        .value("AREA", core::InterpolationMode::AREA)
        .export_values();

    // Padding mode
    py::enum_<core::PaddingMode>(ops_module, "PaddingMode")
        .value("ZEROS", core::PaddingMode::ZEROS)
        .value("BORDER", core::PaddingMode::BORDER)
        .value("REFLECTION", core::PaddingMode::REFLECTION)
        .export_values();

    // GridSample
    py::class_<ops::GridSample>(ops_module, "GridSample",
        "Grid-based spatial sampling operation.\n\n"
        "Samples from input tensor at locations specified by a grid.\n"
        "Grid coordinates are normalized to [-1, 1].\n"
        "Equivalent to torch.nn.functional.grid_sample")
        .def(py::init([](const std::string& mode, const std::string& padding_mode, bool align_corners) {
            core::InterpolationMode interp = core::interpolation_from_string(mode);
            core::PaddingMode pad = core::padding_mode_from_string(padding_mode);
            return std::make_unique<ops::GridSample>(interp, pad, align_corners);
        }),
            py::arg("mode") = "bilinear",
            py::arg("padding_mode") = "zeros",
            py::arg("align_corners") = false)
        .def("get_output_shape", &ops::GridSample::get_output_shape,
            py::arg("input_shape"), py::arg("grid_shape"),
            "Compute output shape given input and grid shapes")
        .def("__repr__", &ops::GridSample::repr)
        .def_property_readonly("mode", [](const ops::GridSample& self) {
            return core::interpolation_name(self.mode());
        })
        .def_property_readonly("padding_mode", [](const ops::GridSample& self) {
            return core::padding_mode_name(self.padding_mode());
        })
        .def_property_readonly("align_corners", &ops::GridSample::align_corners)
        .def("halo_size", &ops::GridSample::halo_size);

    // ROI struct
    py::class_<ops::ROI>(ops_module, "ROI",
        "Region of Interest specification.\n"
        "Format: [batch_index, x1, y1, x2, y2]")
        .def(py::init([](int64_t batch_index, float x1, float y1, float x2, float y2) {
            ops::ROI roi{batch_index, x1, y1, x2, y2};
            roi.validate();
            return roi;
        }),
            py::arg("batch_index"),
            py::arg("x1"), py::arg("y1"),
            py::arg("x2"), py::arg("y2"))
        .def_readwrite("batch_index", &ops::ROI::batch_index)
        .def_readwrite("x1", &ops::ROI::x1)
        .def_readwrite("y1", &ops::ROI::y1)
        .def_readwrite("x2", &ops::ROI::x2)
        .def_readwrite("y2", &ops::ROI::y2)
        .def("width", &ops::ROI::width)
        .def("height", &ops::ROI::height)
        .def("area", &ops::ROI::area);

    // ROIAlign
    py::class_<ops::ROIAlign>(ops_module, "ROIAlign",
        "Region of Interest Align operation.\n\n"
        "Extracts fixed-size feature maps from regions of interest.\n"
        "Equivalent to torchvision.ops.roi_align")
        .def(py::init([](py::object output_size, float spatial_scale, int sampling_ratio, bool aligned) {
            validate_python_float(spatial_scale, "spatial_scale");
            int64_t out_h, out_w;
            if (py::isinstance<py::int_>(output_size)) {
                out_h = out_w = output_size.cast<int64_t>();
            } else if (py::isinstance<py::tuple>(output_size) || py::isinstance<py::list>(output_size)) {
                auto tup = output_size.cast<std::tuple<int64_t, int64_t>>();
                out_h = std::get<0>(tup);
                out_w = std::get<1>(tup);
            } else {
                throw ValidationError("output_size must be int or tuple(height, width)");
            }
            return std::make_unique<ops::ROIAlign>(out_h, out_w, spatial_scale, sampling_ratio, aligned);
        }),
            py::arg("output_size"),
            py::arg("spatial_scale"),
            py::arg("sampling_ratio") = 0,
            py::arg("aligned") = true)
        .def("get_output_shape", &ops::ROIAlign::get_output_shape,
            py::arg("input_shape"), py::arg("num_rois"),
            "Compute output shape given input shape and number of ROIs")
        .def("__repr__", &ops::ROIAlign::repr)
        .def_property_readonly("output_height", &ops::ROIAlign::output_height)
        .def_property_readonly("output_width", &ops::ROIAlign::output_width)
        .def_property_readonly("spatial_scale", &ops::ROIAlign::spatial_scale)
        .def_property_readonly("sampling_ratio", &ops::ROIAlign::sampling_ratio)
        .def_property_readonly("aligned", &ops::ROIAlign::aligned)
        .def("halo_size", &ops::ROIAlign::halo_size);

    // DetectionBox struct
    py::class_<ops::DetectionBox>(ops_module, "DetectionBox",
        "Detection box for NMS.\n"
        "Format: [x1, y1, x2, y2, score, class_id]")
        .def(py::init([](float x1, float y1, float x2, float y2, float score, int64_t class_id) {
            ops::DetectionBox box{x1, y1, x2, y2, score, class_id};
            box.validate();
            return box;
        }),
            py::arg("x1"), py::arg("y1"),
            py::arg("x2"), py::arg("y2"),
            py::arg("score"),
            py::arg("class_id") = 0)
        .def_readwrite("x1", &ops::DetectionBox::x1)
        .def_readwrite("y1", &ops::DetectionBox::y1)
        .def_readwrite("x2", &ops::DetectionBox::x2)
        .def_readwrite("y2", &ops::DetectionBox::y2)
        .def_readwrite("score", &ops::DetectionBox::score)
        .def_readwrite("class_id", &ops::DetectionBox::class_id)
        .def("area", &ops::DetectionBox::area)
        .def("iou", &ops::DetectionBox::iou, py::arg("other"),
            "Compute IoU with another box");

    // NMS
    py::class_<ops::NMS>(ops_module, "NMS",
        "Non-Maximum Suppression operation.\n\n"
        "Filters detection boxes by removing overlapping boxes.\n"
        "Equivalent to torchvision.ops.nms")
        .def(py::init([](float iou_threshold) {
            validate_python_float(iou_threshold, "iou_threshold");
            return std::make_unique<ops::NMS>(iou_threshold);
        }), py::arg("iou_threshold"))
        .def("max_output_size", &ops::NMS::max_output_size,
            py::arg("num_boxes"),
            "Get maximum possible number of kept boxes")
        .def("__repr__", &ops::NMS::repr)
        .def_property_readonly("iou_threshold", &ops::NMS::iou_threshold);

    // BatchedNMS
    py::class_<ops::BatchedNMS>(ops_module, "BatchedNMS",
        "Batched class-aware Non-Maximum Suppression.\n\n"
        "Performs NMS per class within each batch item.")
        .def(py::init([](float iou_threshold) {
            validate_python_float(iou_threshold, "iou_threshold");
            return std::make_unique<ops::BatchedNMS>(iou_threshold);
        }), py::arg("iou_threshold"))
        .def("max_output_size", &ops::BatchedNMS::max_output_size,
            py::arg("num_boxes"))
        .def("__repr__", &ops::BatchedNMS::repr)
        .def_property_readonly("iou_threshold", &ops::BatchedNMS::iou_threshold);

    // SoftNMS
    py::enum_<ops::SoftNMS::Method>(ops_module, "SoftNMSMethod")
        .value("LINEAR", ops::SoftNMS::Method::LINEAR)
        .value("GAUSSIAN", ops::SoftNMS::Method::GAUSSIAN)
        .export_values();

    py::class_<ops::SoftNMS>(ops_module, "SoftNMS",
        "Soft Non-Maximum Suppression.\n\n"
        "Reduces scores of overlapping boxes instead of hard suppression.")
        .def(py::init([](float sigma, float iou_threshold, float score_threshold, const std::string& method) {
            validate_python_float(sigma, "sigma");
            validate_python_float(iou_threshold, "iou_threshold");
            validate_python_float(score_threshold, "score_threshold");
            ops::SoftNMS::Method m = (method == "gaussian") ?
                ops::SoftNMS::Method::GAUSSIAN : ops::SoftNMS::Method::LINEAR;
            return std::make_unique<ops::SoftNMS>(sigma, iou_threshold, score_threshold, m);
        }),
            py::arg("sigma") = 0.5f,
            py::arg("iou_threshold") = 0.3f,
            py::arg("score_threshold") = 0.001f,
            py::arg("method") = "gaussian")
        .def("max_output_size", &ops::SoftNMS::max_output_size,
            py::arg("num_boxes"))
        .def("__repr__", &ops::SoftNMS::repr)
        .def_property_readonly("sigma", &ops::SoftNMS::sigma)
        .def_property_readonly("iou_threshold", &ops::SoftNMS::iou_threshold)
        .def_property_readonly("score_threshold", &ops::SoftNMS::score_threshold);

    // Functional API for ops
    ops_module.def("roi_align",
        [](const std::vector<int64_t>& input_shape,
           int64_t num_rois,
           py::object output_size,
           float spatial_scale,
           int sampling_ratio,
           bool aligned) {
            validate_python_float(spatial_scale, "spatial_scale");
            int64_t out_h, out_w;
            if (py::isinstance<py::int_>(output_size)) {
                out_h = out_w = output_size.cast<int64_t>();
            } else {
                auto tup = output_size.cast<std::tuple<int64_t, int64_t>>();
                out_h = std::get<0>(tup);
                out_w = std::get<1>(tup);
            }
            ops::ROIAlign op(out_h, out_w, spatial_scale, sampling_ratio, aligned);
            return op.get_output_shape(input_shape, num_rois);
        },
        py::arg("input_shape"),
        py::arg("num_rois"),
        py::arg("output_size"),
        py::arg("spatial_scale"),
        py::arg("sampling_ratio") = 0,
        py::arg("aligned") = true,
        "Compute ROI Align output shape");

    ops_module.def("nms",
        [](int64_t num_boxes, float iou_threshold) {
            validate_python_float(iou_threshold, "iou_threshold");
            ops::NMS op(iou_threshold);
            return op.max_output_size(num_boxes);
        },
        py::arg("num_boxes"),
        py::arg("iou_threshold"),
        "Get maximum output size for NMS");

    ops_module.def("batched_nms",
        [](int64_t num_boxes, float iou_threshold) {
            validate_python_float(iou_threshold, "iou_threshold");
            ops::BatchedNMS op(iou_threshold);
            return op.max_output_size(num_boxes);
        },
        py::arg("num_boxes"),
        py::arg("iou_threshold"),
        "Get maximum output size for batched NMS");

    ops_module.def("grid_sample",
        [](const std::vector<int64_t>& input_shape,
           const std::vector<int64_t>& grid_shape,
           const std::string& mode,
           const std::string& padding_mode,
           bool align_corners) {
            core::InterpolationMode interp = core::interpolation_from_string(mode);
            core::PaddingMode pad = core::padding_mode_from_string(padding_mode);
            ops::GridSample op(interp, pad, align_corners);
            return op.get_output_shape(input_shape, grid_shape);
        },
        py::arg("input_shape"),
        py::arg("grid_shape"),
        py::arg("mode") = "bilinear",
        py::arg("padding_mode") = "zeros",
        py::arg("align_corners") = false,
        "Compute grid sample output shape");
}

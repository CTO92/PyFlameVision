/**
 * PyFlameVision Example: ImageNet Classification Transforms
 *
 * This example demonstrates how to create a typical ImageNet
 * preprocessing pipeline using PyFlameVision transforms.
 */

#include <iostream>
#include <pyflame_vision/pyflame_vision.hpp>

using namespace pyflame_vision::transforms;
using namespace pyflame_vision::core;

int main() {
    std::cout << "PyFlameVision ImageNet Transform Example\n";
    std::cout << "========================================\n\n";

    // ImageNet normalization parameters
    std::vector<float> imagenet_mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> imagenet_std = {0.229f, 0.224f, 0.225f};

    // Create transforms for validation (center crop)
    Compose val_transforms({
        std::make_shared<Resize>(256),
        std::make_shared<CenterCrop>(224),
        std::make_shared<Normalize>(imagenet_mean, imagenet_std)
    });

    std::cout << "Validation transforms:\n" << val_transforms.repr() << "\n\n";

    // Create transforms for training (random crop)
    Compose train_transforms({
        std::make_shared<Resize>(256),
        std::make_shared<RandomCrop>(224),
        std::make_shared<Normalize>(imagenet_mean, imagenet_std)
    });

    std::cout << "Training transforms:\n" << train_transforms.repr() << "\n\n";

    // Example input shape (batch of 4 RGB images at 480x640)
    std::vector<int64_t> input_shape = {4, 3, 480, 640};

    std::cout << "Input shape: ["
              << input_shape[0] << ", "
              << input_shape[1] << ", "
              << input_shape[2] << ", "
              << input_shape[3] << "]\n";

    // Get output shapes
    auto val_output = val_transforms.get_output_shape(input_shape);
    auto train_output = train_transforms.get_output_shape(input_shape);

    std::cout << "Validation output shape: ["
              << val_output[0] << ", "
              << val_output[1] << ", "
              << val_output[2] << ", "
              << val_output[3] << "]\n";

    std::cout << "Training output shape: ["
              << train_output[0] << ", "
              << train_output[1] << ", "
              << train_output[2] << ", "
              << train_output[3] << "]\n\n";

    // Check determinism
    std::cout << "Validation pipeline deterministic: "
              << (val_transforms.is_deterministic() ? "yes" : "no") << "\n";
    std::cout << "Training pipeline deterministic: "
              << (train_transforms.is_deterministic() ? "yes" : "no") << "\n\n";

    // Memory estimation
    std::cout << "Input memory: "
              << ImageTensor::size_bytes(input_shape) / 1024.0 / 1024.0
              << " MB\n";
    std::cout << "Output memory: "
              << ImageTensor::size_bytes(val_output) / 1024.0 / 1024.0
              << " MB\n";

    return 0;
}

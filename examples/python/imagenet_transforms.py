"""
PyFlameVision Example: ImageNet Classification Transforms

This example demonstrates how to create a typical ImageNet
preprocessing pipeline using PyFlameVision transforms.
"""

from pyflame_vision.transforms import (
    Resize,
    CenterCrop,
    RandomCrop,
    Normalize,
    Compose,
)


def main():
    print("PyFlameVision ImageNet Transform Example")
    print("=" * 40)
    print()

    # ImageNet normalization parameters
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Create transforms for validation (center crop)
    val_transforms = Compose([
        Resize(256),
        CenterCrop(224),
        Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    print("Validation transforms:")
    print(val_transforms)
    print()

    # Create transforms for training (random crop)
    train_transforms = Compose([
        Resize(256),
        RandomCrop(224),
        Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    print("Training transforms:")
    print(train_transforms)
    print()

    # Example input shape (batch of 4 RGB images at 480x640)
    input_shape = [4, 3, 480, 640]

    print(f"Input shape: {input_shape}")

    # Get output shapes
    val_output = val_transforms.get_output_shape(input_shape)
    train_output = train_transforms.get_output_shape(input_shape)

    print(f"Validation output shape: {val_output}")
    print(f"Training output shape: {train_output}")
    print()

    # Check determinism
    print(f"Validation pipeline deterministic: {val_transforms.is_deterministic()}")
    print(f"Training pipeline deterministic: {train_transforms.is_deterministic()}")
    print()

    # Memory estimation (assuming float32 = 4 bytes)
    def memory_mb(shape):
        elements = 1
        for dim in shape:
            elements *= dim
        return elements * 4 / 1024 / 1024

    print(f"Input memory: {memory_mb(input_shape):.2f} MB")
    print(f"Output memory: {memory_mb(val_output):.2f} MB")


if __name__ == "__main__":
    main()

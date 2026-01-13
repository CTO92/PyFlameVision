"""
Functional API for PyFlameVision transforms.

These are stateless functions that can be used directly without creating transform objects.
"""

from typing import List, Tuple, Union, Optional


def get_output_shape_resize(
    input_shape: List[int],
    size: Union[int, Tuple[int, int]]
) -> List[int]:
    """
    Get output shape for resize operation.

    Args:
        input_shape: Input tensor shape [N, C, H, W]
        size: Target size (int for square, tuple for (height, width))

    Returns:
        Output shape [N, C, new_H, new_W]
    """
    if len(input_shape) != 4:
        raise ValueError(f"Expected 4D NCHW tensor, got shape {input_shape}")

    if isinstance(size, int):
        target_h = target_w = size
    else:
        target_h, target_w = size

    return [input_shape[0], input_shape[1], target_h, target_w]


def get_output_shape_center_crop(
    input_shape: List[int],
    size: Union[int, Tuple[int, int]]
) -> List[int]:
    """
    Get output shape for center crop operation.

    Args:
        input_shape: Input tensor shape [N, C, H, W]
        size: Crop size (int for square, tuple for (height, width))

    Returns:
        Output shape [N, C, crop_H, crop_W]
    """
    if len(input_shape) != 4:
        raise ValueError(f"Expected 4D NCHW tensor, got shape {input_shape}")

    if isinstance(size, int):
        crop_h = crop_w = size
    else:
        crop_h, crop_w = size

    in_h, in_w = input_shape[2], input_shape[3]
    if crop_h > in_h or crop_w > in_w:
        raise ValueError(
            f"Crop size ({crop_h}, {crop_w}) larger than image size ({in_h}, {in_w})"
        )

    return [input_shape[0], input_shape[1], crop_h, crop_w]


def get_output_shape_random_crop(
    input_shape: List[int],
    size: Union[int, Tuple[int, int]],
    padding: int = 0
) -> List[int]:
    """
    Get output shape for random crop operation.

    Args:
        input_shape: Input tensor shape [N, C, H, W]
        size: Crop size (int for square, tuple for (height, width))
        padding: Optional padding applied before crop

    Returns:
        Output shape [N, C, crop_H, crop_W]
    """
    if len(input_shape) != 4:
        raise ValueError(f"Expected 4D NCHW tensor, got shape {input_shape}")

    if isinstance(size, int):
        crop_h = crop_w = size
    else:
        crop_h, crop_w = size

    return [input_shape[0], input_shape[1], crop_h, crop_w]


def get_output_shape_normalize(
    input_shape: List[int],
    mean: List[float],
    std: List[float]
) -> List[int]:
    """
    Get output shape for normalize operation (preserves shape).

    Args:
        input_shape: Input tensor shape [N, C, H, W]
        mean: Per-channel mean values
        std: Per-channel standard deviation values

    Returns:
        Output shape (same as input)
    """
    if len(input_shape) != 4:
        raise ValueError(f"Expected 4D NCHW tensor, got shape {input_shape}")

    num_channels = input_shape[1]
    if len(mean) != num_channels:
        raise ValueError(
            f"mean length ({len(mean)}) does not match "
            f"number of channels ({num_channels})"
        )
    if len(std) != num_channels:
        raise ValueError(
            f"std length ({len(std)}) does not match "
            f"number of channels ({num_channels})"
        )

    return list(input_shape)


def compute_center_crop_bounds(
    input_height: int,
    input_width: int,
    crop_height: int,
    crop_width: int
) -> Tuple[int, int, int, int]:
    """
    Compute crop bounds for center crop.

    Args:
        input_height: Input image height
        input_width: Input image width
        crop_height: Desired crop height
        crop_width: Desired crop width

    Returns:
        Tuple of (top, left, height, width)
    """
    if crop_height > input_height or crop_width > input_width:
        raise ValueError(
            f"Crop size ({crop_height}, {crop_width}) larger than "
            f"image size ({input_height}, {input_width})"
        )

    top = (input_height - crop_height) // 2
    left = (input_width - crop_width) // 2

    return (top, left, crop_height, crop_width)


def validate_image_shape(shape: List[int]) -> None:
    """
    Validate that shape is a valid NCHW image tensor shape.

    Args:
        shape: Shape to validate

    Raises:
        ValueError: If shape is invalid
    """
    if len(shape) != 4:
        raise ValueError(f"Expected 4D NCHW tensor, got {len(shape)}D shape {shape}")

    for i, dim in enumerate(shape):
        if dim <= 0:
            raise ValueError(
                f"All dimensions must be positive, got {dim} at index {i}"
            )


def validate_normalize_params(
    mean: List[float],
    std: List[float],
    num_channels: Optional[int] = None
) -> None:
    """
    Validate normalization parameters.

    Args:
        mean: Per-channel mean values
        std: Per-channel standard deviation values
        num_channels: Optional expected number of channels

    Raises:
        ValueError: If parameters are invalid
    """
    if len(mean) != len(std):
        raise ValueError(
            f"mean and std must have same length, got {len(mean)} and {len(std)}"
        )

    if not mean:
        raise ValueError("mean cannot be empty")

    for i, s in enumerate(std):
        if s <= 0:
            raise ValueError(f"std values must be positive, got {s} at index {i}")

    if num_channels is not None and len(mean) != num_channels:
        raise ValueError(
            f"mean/std length ({len(mean)}) does not match "
            f"number of channels ({num_channels})"
        )

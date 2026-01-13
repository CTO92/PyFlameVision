"""
Pure Python implementations of PyFlameVision transforms.

These are fallback implementations used when the C++ extension is not available.
They provide the same API but without CSL code generation.
"""

from typing import List, Tuple, Union, Optional
from abc import ABC, abstractmethod
import random
import math


class Size:
    """Size specification for transforms."""

    def __init__(self, height: int, width: Optional[int] = None):
        """
        Create a Size.

        Args:
            height: Height or size for square
            width: Width (if None, creates square size)
        """
        if width is None:
            width = height
        self.height = height
        self.width = width

    def is_valid(self) -> bool:
        """Check if size is valid (positive dimensions)."""
        return self.height > 0 and self.width > 0

    def __repr__(self) -> str:
        if self.height == self.width:
            return str(self.height)
        return f"({self.height}, {self.width})"


def _parse_size(size: Union[int, Tuple[int, int], List[int]]) -> Size:
    """Parse size argument into Size object."""
    if isinstance(size, int):
        return Size(size)
    elif isinstance(size, (tuple, list)):
        if len(size) == 1:
            return Size(size[0])
        elif len(size) == 2:
            return Size(size[0], size[1])
    raise ValueError(f"Size must be int or (height, width), got {size}")


def _validate_image_shape(shape: List[int]) -> None:
    """Validate that shape is a valid NCHW image."""
    if len(shape) != 4:
        raise ValueError(f"Expected 4D NCHW tensor, got shape {shape}")
    for dim in shape:
        if dim <= 0:
            raise ValueError(f"All dimensions must be positive, got {shape}")


class Transform(ABC):
    """Base class for all transforms."""

    @abstractmethod
    def get_output_shape(self, input_shape: List[int]) -> List[int]:
        """Get output shape for given input shape."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Get transform name."""
        pass

    def is_deterministic(self) -> bool:
        """Check if transform is deterministic."""
        return True

    @abstractmethod
    def __repr__(self) -> str:
        """Get string representation."""
        pass


class Resize(Transform):
    """Resize image to given size."""

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        interpolation: str = "bilinear",
        antialias: bool = True
    ):
        self._size = _parse_size(size)
        self._interpolation = interpolation
        self._antialias = antialias

    def get_output_shape(self, input_shape: List[int]) -> List[int]:
        _validate_image_shape(input_shape)
        return [
            input_shape[0],  # batch
            input_shape[1],  # channels
            self._size.height,
            self._size.width,
        ]

    def name(self) -> str:
        return "Resize"

    def interpolation(self) -> str:
        return self._interpolation

    def antialias(self) -> bool:
        return self._antialias

    def __repr__(self) -> str:
        return f"Resize(size={self._size}, interpolation={self._interpolation})"


class CenterCrop(Transform):
    """Center crop image to given size."""

    def __init__(self, size: Union[int, Tuple[int, int]]):
        self._size = _parse_size(size)

    def get_output_shape(self, input_shape: List[int]) -> List[int]:
        _validate_image_shape(input_shape)
        in_h, in_w = input_shape[2], input_shape[3]

        if self._size.height > in_h or self._size.width > in_w:
            raise ValueError(
                f"Crop size ({self._size.height}, {self._size.width}) "
                f"larger than image size ({in_h}, {in_w})"
            )

        return [
            input_shape[0],
            input_shape[1],
            self._size.height,
            self._size.width,
        ]

    def compute_bounds(
        self, input_height: int, input_width: int
    ) -> Tuple[int, int, int, int]:
        """Compute crop bounds (top, left, height, width)."""
        top = (input_height - self._size.height) // 2
        left = (input_width - self._size.width) // 2
        return (top, left, self._size.height, self._size.width)

    def name(self) -> str:
        return "CenterCrop"

    def __repr__(self) -> str:
        return f"CenterCrop(size={self._size})"


class RandomCrop(Transform):
    """Random crop image to given size."""

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        padding: int = 0,
        pad_if_needed: bool = False,
        fill: float = 0.0
    ):
        self._size = _parse_size(size)
        self._padding = padding
        self._pad_if_needed = pad_if_needed
        self._fill = fill
        self._seed = None

    def get_output_shape(self, input_shape: List[int]) -> List[int]:
        _validate_image_shape(input_shape)
        return [
            input_shape[0],
            input_shape[1],
            self._size.height,
            self._size.width,
        ]

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self._seed = seed
        random.seed(seed)

    def padding(self) -> int:
        return self._padding

    def pad_if_needed(self) -> bool:
        return self._pad_if_needed

    def name(self) -> str:
        return "RandomCrop"

    def is_deterministic(self) -> bool:
        return False

    def __repr__(self) -> str:
        parts = [f"size={self._size}"]
        if self._padding > 0:
            parts.append(f"padding={self._padding}")
        if self._pad_if_needed:
            parts.append("pad_if_needed=True")
        return f"RandomCrop({', '.join(parts)})"


class Normalize(Transform):
    """Normalize tensor with mean and standard deviation."""

    def __init__(
        self,
        mean: List[float],
        std: List[float],
        inplace: bool = False
    ):
        if len(mean) != len(std):
            raise ValueError(
                f"mean and std must have same length, "
                f"got {len(mean)} and {len(std)}"
            )
        if not mean:
            raise ValueError("mean cannot be empty")

        for i, s in enumerate(std):
            if s <= 0:
                raise ValueError(f"std values must be positive, got {s} at index {i}")

        self._mean = list(mean)
        self._std = list(std)
        self._inv_std = [1.0 / s for s in std]
        self._inplace = inplace

    def get_output_shape(self, input_shape: List[int]) -> List[int]:
        _validate_image_shape(input_shape)
        num_channels = input_shape[1]
        if num_channels != len(self._mean):
            raise ValueError(
                f"mean/std length ({len(self._mean)}) does not match "
                f"number of channels ({num_channels})"
            )
        return input_shape  # Normalize preserves shape

    def mean(self) -> List[float]:
        return self._mean

    def std(self) -> List[float]:
        return self._std

    def inv_std(self) -> List[float]:
        return self._inv_std

    def inplace(self) -> bool:
        return self._inplace

    def name(self) -> str:
        return "Normalize"

    def __repr__(self) -> str:
        mean_str = "[" + ", ".join(f"{m:.4f}" for m in self._mean) + "]"
        std_str = "[" + ", ".join(f"{s:.4f}" for s in self._std) + "]"
        return f"Normalize(mean={mean_str}, std={std_str})"


class Compose(Transform):
    """Compose multiple transforms into a pipeline."""

    def __init__(self, transforms: List[Transform]):
        for i, t in enumerate(transforms):
            if t is None:
                raise ValueError(f"Transform at index {i} is None")
        self._transforms = list(transforms)

    def get_output_shape(self, input_shape: List[int]) -> List[int]:
        shape = input_shape
        for transform in self._transforms:
            shape = transform.get_output_shape(shape)
        return shape

    def name(self) -> str:
        return "Compose"

    def is_deterministic(self) -> bool:
        return all(t.is_deterministic() for t in self._transforms)

    def __len__(self) -> int:
        return len(self._transforms)

    def __getitem__(self, index: int) -> Transform:
        return self._transforms[index]

    def transforms(self) -> List[Transform]:
        return self._transforms

    def empty(self) -> bool:
        return len(self._transforms) == 0

    def __repr__(self) -> str:
        lines = ["Compose(["]
        for i, t in enumerate(self._transforms):
            comma = "," if i < len(self._transforms) - 1 else ""
            lines.append(f"    {t}{comma}")
        lines.append("])")
        return "\n".join(lines)

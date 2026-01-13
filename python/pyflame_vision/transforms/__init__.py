"""
PyFlameVision Transforms

Image transforms for preprocessing and data augmentation.
API is compatible with torchvision.transforms.

Example:
    >>> import pyflame_vision.transforms as T
    >>> transform = T.Compose([
    ...     T.Resize(256),
    ...     T.CenterCrop(224),
    ...     T.Normalize(mean=[0.485, 0.456, 0.406],
    ...                 std=[0.229, 0.224, 0.225])
    ... ])
"""

try:
    from .._pyflame_vision_cpp.transforms import (
        Transform,
        Size,
        Resize,
        CenterCrop,
        RandomCrop,
        Normalize,
        Compose,
    )
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False
    # Provide Python fallback implementations
    from .transforms import (
        Transform,
        Size,
        Resize,
        CenterCrop,
        RandomCrop,
        Normalize,
        Compose,
    )

from . import functional

__all__ = [
    "Transform",
    "Size",
    "Resize",
    "CenterCrop",
    "RandomCrop",
    "Normalize",
    "Compose",
    "functional",
]

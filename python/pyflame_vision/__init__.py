"""
PyFlameVision: Cerebras-native Computer Vision Library

**PRE-RELEASE ALPHA SOFTWARE**

This project is currently in an early alpha stage of development.
APIs may change without notice, features may be incomplete, and the software
is not yet recommended for production use. Use at your own risk.

A standalone computer vision library with optional PyFlame integration
for hardware-accelerated execution on Cerebras WSE.

PyFlameVision can operate in two modes:
    1. Standalone: Pure Python with NumPy/PIL (no PyFlame required)
    2. Integrated: Hardware acceleration via PyFlame tensors and CSL

Example:
    >>> import pyflame_vision.transforms as T
    >>> transform = T.Compose([
    ...     T.Resize(256),
    ...     T.CenterCrop(224),
    ...     T.Normalize(mean=[0.485, 0.456, 0.406],
    ...                 std=[0.229, 0.224, 0.225])
    ... ])
    >>> # Get output shape
    >>> output_shape = transform.get_output_shape([1, 3, 512, 512])
    >>> print(output_shape)  # [1, 3, 224, 224]
"""

from ._version import __version__, version_info

# Check for PyFlame availability (optional dependency)
try:
    import pyflame
    HAS_PYFLAME = True
except ImportError:
    HAS_PYFLAME = False

# Import C++ bindings (optional, requires build from source)
try:
    from . import _pyflame_vision_cpp as _cpp
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False
    # Only warn if PyFlame is available (user likely expects C++ acceleration)
    if HAS_PYFLAME:
        import warnings
        warnings.warn(
            "PyFlameVision C++ extension not found. "
            "Install from source with PyFlame for hardware acceleration.",
            ImportWarning
        )

# Import submodules
from . import transforms
from .transforms import functional
from . import ops
from . import datasets
from . import io

# Core utilities
if _HAS_CPP:
    from ._pyflame_vision_cpp.core import (
        InterpolationMode,
        ColorSpace,
        ImageTensor,
    )

# Convenient imports for datasets module
from .datasets import (
    Dataset,
    DataLoader,
    ImageFolder,
)

# Convenient imports for io module
from .io import (
    read_image,
    write_image,
    decode_image,
    encode_image,
    ImageReadMode,
)

__all__ = [
    "__version__",
    "version_info",
    # Feature detection flags
    "HAS_PYFLAME",
    "_HAS_CPP",
    # Submodules
    "transforms",
    "functional",
    "ops",
    "datasets",
    "io",
    # Dataset shortcuts
    "Dataset",
    "DataLoader",
    "ImageFolder",
    # IO shortcuts
    "read_image",
    "write_image",
    "decode_image",
    "encode_image",
    "ImageReadMode",
]

if _HAS_CPP:
    __all__.extend([
        "InterpolationMode",
        "ColorSpace",
        "ImageTensor",
    ])

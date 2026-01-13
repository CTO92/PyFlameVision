"""
Image I/O utilities for PyFlameVision.

Provides functions for reading and writing images in various formats.
"""

from enum import Enum, auto
from pathlib import Path
from typing import Union, Optional
import os

# Try to import numpy, fall back to list-based implementation
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Try to import PIL for image loading
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ============================================================================
# Security Limits
# ============================================================================

class IOSecurityLimits:
    """Security limits for I/O operations."""
    MAX_IMAGE_WIDTH = 65536
    MAX_IMAGE_HEIGHT = 65536
    MAX_PIXEL_COUNT = 100_000_000  # 100 megapixels (~400MB for RGBA uint8)
    MAX_FILE_SIZE = 1 << 30  # 1 GB
    MAX_PATH_LENGTH = 4096
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif'}


# Set PIL decompression bomb limit if PIL is available
if HAS_PIL:
    Image.MAX_IMAGE_PIXELS = IOSecurityLimits.MAX_PIXEL_COUNT


# ============================================================================
# Enums
# ============================================================================

class ImageReadMode(Enum):
    """Image reading modes."""
    UNCHANGED = auto()  # Keep original format
    GRAY = auto()       # Convert to grayscale
    GRAY_ALPHA = auto() # Grayscale with alpha
    RGB = auto()        # Convert to RGB
    RGB_ALPHA = auto()  # RGB with alpha (RGBA)


# ============================================================================
# Validation Functions
# ============================================================================

def _validate_path(path: Union[str, Path], must_exist: bool = True) -> Path:
    """Validate file path.

    Args:
        path: Path to validate
        must_exist: Whether the file must exist

    Returns:
        Validated Path object

    Raises:
        ValueError: If path is invalid
        FileNotFoundError: If file doesn't exist and must_exist=True
        PermissionError: If file cannot be read
    """
    path_str = str(path)

    # Security: Check for null bytes which could be interpreted differently
    if '\x00' in path_str:
        raise ValueError("Null bytes not allowed in path")

    path = Path(path)

    # Check path length
    if len(path_str) > IOSecurityLimits.MAX_PATH_LENGTH:
        raise ValueError(
            f"Path too long: {len(path_str)} > {IOSecurityLimits.MAX_PATH_LENGTH}"
        )

    if must_exist:
        # Security: Check for symlinks to prevent symlink attacks
        if path.is_symlink():
            raise ValueError(f"Symbolic links not allowed: {path}")

        # Check existence
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Check is file
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        # Check readable
        if not os.access(path, os.R_OK):
            raise PermissionError(f"Cannot read file: {path}")

        # Check file size
        file_size = path.stat().st_size
        if file_size > IOSecurityLimits.MAX_FILE_SIZE:
            raise ValueError(
                f"File too large: {file_size} > {IOSecurityLimits.MAX_FILE_SIZE}"
            )

    return path


def _validate_image_dimensions(width: int, height: int) -> None:
    """Validate image dimensions.

    Args:
        width: Image width
        height: Image height

    Raises:
        ValueError: If dimensions exceed limits
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")

    if width > IOSecurityLimits.MAX_IMAGE_WIDTH:
        raise ValueError(
            f"Image width ({width}) exceeds maximum ({IOSecurityLimits.MAX_IMAGE_WIDTH})"
        )

    if height > IOSecurityLimits.MAX_IMAGE_HEIGHT:
        raise ValueError(
            f"Image height ({height}) exceeds maximum ({IOSecurityLimits.MAX_IMAGE_HEIGHT})"
        )

    # Check total pixel count to prevent memory exhaustion
    pixel_count = width * height
    if pixel_count > IOSecurityLimits.MAX_PIXEL_COUNT:
        raise ValueError(
            f"Image pixel count ({pixel_count:,}) exceeds maximum "
            f"({IOSecurityLimits.MAX_PIXEL_COUNT:,})"
        )


def _validate_extension(path: Path) -> None:
    """Validate file extension is supported.

    Args:
        path: File path

    Raises:
        ValueError: If extension is not supported
    """
    ext = path.suffix.lower()
    if ext not in IOSecurityLimits.SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: {ext}. "
            f"Supported: {IOSecurityLimits.SUPPORTED_EXTENSIONS}"
        )


# ============================================================================
# Image Reading
# ============================================================================

def read_image(
    path: Union[str, Path],
    mode: ImageReadMode = ImageReadMode.UNCHANGED
) -> "np.ndarray":
    """Read an image file into a tensor.

    Args:
        path: Path to image file
        mode: How to interpret image data
            - UNCHANGED: Keep original format
            - GRAY: Convert to grayscale
            - RGB: Convert to RGB
            - RGB_ALPHA: RGB with alpha

    Returns:
        NumPy array of shape [C, H, W] with dtype uint8

    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If file is not a valid image
        PermissionError: If file cannot be read
        ImportError: If PIL or numpy is not available

    Example:
        >>> img = read_image("photo.jpg")
        >>> img.shape
        (3, 480, 640)
    """
    if not HAS_PIL:
        raise ImportError("PIL (Pillow) is required for image I/O. Install with: pip install Pillow")
    if not HAS_NUMPY:
        raise ImportError("NumPy is required for image I/O. Install with: pip install numpy")

    # Validate path
    path = _validate_path(path, must_exist=True)
    _validate_extension(path)

    # Load image
    try:
        img = Image.open(path)
    except Exception as e:
        raise ValueError(f"Failed to open image: {path}. Error: {e}")

    # Validate dimensions
    _validate_image_dimensions(img.width, img.height)

    # Convert mode
    if mode == ImageReadMode.GRAY:
        img = img.convert("L")
    elif mode == ImageReadMode.GRAY_ALPHA:
        img = img.convert("LA")
    elif mode == ImageReadMode.RGB:
        img = img.convert("RGB")
    elif mode == ImageReadMode.RGB_ALPHA:
        img = img.convert("RGBA")
    # UNCHANGED keeps original

    # Convert to numpy array [H, W, C] or [H, W]
    arr = np.array(img)

    # Convert to [C, H, W] format
    if arr.ndim == 2:
        # Grayscale: [H, W] -> [1, H, W]
        arr = arr[np.newaxis, :, :]
    elif arr.ndim == 3:
        # Color: [H, W, C] -> [C, H, W]
        arr = arr.transpose(2, 0, 1)
    else:
        raise ValueError(f"Unexpected image dimensions: {arr.ndim}")

    return arr


def decode_image(
    data: bytes,
    mode: ImageReadMode = ImageReadMode.UNCHANGED
) -> "np.ndarray":
    """Decode image bytes into a tensor.

    Useful for streaming data or in-memory images.

    Args:
        data: Raw image bytes (JPEG, PNG, etc.)
        mode: How to interpret image data

    Returns:
        NumPy array of shape [C, H, W] with dtype uint8

    Raises:
        ValueError: If data is not a valid image
        ImportError: If PIL or numpy is not available

    Example:
        >>> with open("photo.jpg", "rb") as f:
        ...     data = f.read()
        >>> img = decode_image(data)
    """
    if not HAS_PIL:
        raise ImportError("PIL (Pillow) is required for image I/O")
    if not HAS_NUMPY:
        raise ImportError("NumPy is required for image I/O")

    import io as pyio

    # Validate data size
    if len(data) > IOSecurityLimits.MAX_FILE_SIZE:
        raise ValueError(
            f"Image data too large: {len(data)} > {IOSecurityLimits.MAX_FILE_SIZE}"
        )

    # Decode image
    try:
        img = Image.open(pyio.BytesIO(data))
    except Exception as e:
        raise ValueError(f"Failed to decode image. Error: {e}")

    # Validate dimensions
    _validate_image_dimensions(img.width, img.height)

    # Convert mode
    if mode == ImageReadMode.GRAY:
        img = img.convert("L")
    elif mode == ImageReadMode.GRAY_ALPHA:
        img = img.convert("LA")
    elif mode == ImageReadMode.RGB:
        img = img.convert("RGB")
    elif mode == ImageReadMode.RGB_ALPHA:
        img = img.convert("RGBA")

    # Convert to numpy array [H, W, C] or [H, W]
    arr = np.array(img)

    # Convert to [C, H, W] format
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim == 3:
        arr = arr.transpose(2, 0, 1)
    else:
        raise ValueError(f"Unexpected image dimensions: {arr.ndim}")

    return arr


# ============================================================================
# Image Writing
# ============================================================================

def write_image(
    tensor: "np.ndarray",
    path: Union[str, Path],
    format: Optional[str] = None,
    quality: int = 95
) -> None:
    """Write a tensor to an image file.

    Args:
        tensor: Image tensor [C, H, W] or [H, W]
        path: Output file path
        format: Image format (inferred from path if None)
        quality: JPEG quality (1-100)

    Raises:
        ValueError: If tensor has invalid shape
        PermissionError: If file cannot be written
        ImportError: If PIL or numpy is not available

    Example:
        >>> img = np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8)
        >>> write_image(img, "output.png")
    """
    if not HAS_PIL:
        raise ImportError("PIL (Pillow) is required for image I/O")
    if not HAS_NUMPY:
        raise ImportError("NumPy is required for image I/O")

    # Validate path (don't require existence)
    path = _validate_path(path, must_exist=False)

    # Security: Validate output extension if format not explicitly specified
    if format is None:
        ext = path.suffix.lower()
        if ext and ext not in IOSecurityLimits.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported output format: {ext}. "
                f"Supported: {IOSecurityLimits.SUPPORTED_EXTENSIONS}. "
                "Use the 'format' parameter to specify format explicitly."
            )

    # Validate quality
    if not 1 <= quality <= 100:
        raise ValueError(f"Quality must be in [1, 100], got {quality}")

    # Validate tensor shape
    if tensor.ndim == 2:
        # Grayscale [H, W]
        height, width = tensor.shape
        img_arr = tensor
        pil_mode = "L"
    elif tensor.ndim == 3:
        # [C, H, W] -> [H, W, C]
        channels, height, width = tensor.shape
        if channels == 1:
            img_arr = tensor[0]
            pil_mode = "L"
        elif channels == 3:
            img_arr = tensor.transpose(1, 2, 0)
            pil_mode = "RGB"
        elif channels == 4:
            img_arr = tensor.transpose(1, 2, 0)
            pil_mode = "RGBA"
        else:
            raise ValueError(f"Unsupported number of channels: {channels}")
    else:
        raise ValueError(f"Tensor must be 2D or 3D, got {tensor.ndim}D")

    # Validate dimensions
    _validate_image_dimensions(width, height)

    # Ensure uint8
    if img_arr.dtype != np.uint8:
        if img_arr.max() <= 1.0:
            # Assume [0, 1] range
            img_arr = (img_arr * 255).astype(np.uint8)
        else:
            img_arr = img_arr.astype(np.uint8)

    # Create PIL image
    img = Image.fromarray(img_arr, mode=pil_mode)

    # Determine format
    if format is None:
        format = path.suffix.lower().lstrip('.')
        if format in ('jpg', 'jpeg'):
            format = 'JPEG'
        elif format == 'png':
            format = 'PNG'
        elif format == 'bmp':
            format = 'BMP'
        elif format == 'gif':
            format = 'GIF'
        elif format == 'webp':
            format = 'WEBP'
        elif format in ('tiff', 'tif'):
            format = 'TIFF'
        else:
            format = 'PNG'  # Default

    # Create parent directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    save_kwargs = {}
    if format == 'JPEG':
        save_kwargs['quality'] = quality
    elif format == 'WEBP':
        save_kwargs['quality'] = quality

    try:
        img.save(path, format=format, **save_kwargs)
    except Exception as e:
        raise ValueError(f"Failed to save image: {path}. Error: {e}")


def encode_image(
    tensor: "np.ndarray",
    format: str = "PNG",
    quality: int = 95
) -> bytes:
    """Encode a tensor to image bytes.

    Args:
        tensor: Image tensor [C, H, W] or [H, W]
        format: Image format (PNG, JPEG, etc.)
        quality: JPEG/WebP quality (1-100)

    Returns:
        Encoded image bytes

    Raises:
        ValueError: If tensor has invalid shape
        ImportError: If PIL or numpy is not available

    Example:
        >>> img = np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8)
        >>> data = encode_image(img, format="JPEG", quality=90)
    """
    if not HAS_PIL:
        raise ImportError("PIL (Pillow) is required for image I/O")
    if not HAS_NUMPY:
        raise ImportError("NumPy is required for image I/O")

    import io as pyio

    # Validate tensor shape
    if tensor.ndim == 2:
        height, width = tensor.shape
        img_arr = tensor
        pil_mode = "L"
    elif tensor.ndim == 3:
        channels, height, width = tensor.shape
        if channels == 1:
            img_arr = tensor[0]
            pil_mode = "L"
        elif channels == 3:
            img_arr = tensor.transpose(1, 2, 0)
            pil_mode = "RGB"
        elif channels == 4:
            img_arr = tensor.transpose(1, 2, 0)
            pil_mode = "RGBA"
        else:
            raise ValueError(f"Unsupported number of channels: {channels}")
    else:
        raise ValueError(f"Tensor must be 2D or 3D, got {tensor.ndim}D")

    # Validate dimensions
    _validate_image_dimensions(width, height)

    # Ensure uint8
    if img_arr.dtype != np.uint8:
        if img_arr.max() <= 1.0:
            img_arr = (img_arr * 255).astype(np.uint8)
        else:
            img_arr = img_arr.astype(np.uint8)

    # Create PIL image
    img = Image.fromarray(img_arr, mode=pil_mode)

    # Encode to bytes
    buffer = pyio.BytesIO()
    save_kwargs = {}
    if format.upper() in ('JPEG', 'JPG'):
        format = 'JPEG'
        save_kwargs['quality'] = quality
    elif format.upper() == 'WEBP':
        save_kwargs['quality'] = quality

    img.save(buffer, format=format.upper(), **save_kwargs)
    return buffer.getvalue()

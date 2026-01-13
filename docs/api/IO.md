# IO API Reference

PyFlameVision provides image I/O utilities for reading and writing images in various formats.

## Quick Start

```python
from pyflame_vision import read_image, write_image, ImageReadMode

# Read an image
image = read_image("photo.jpg")
print(image.shape)  # (3, 480, 640) - [C, H, W] format

# Read as grayscale
gray = read_image("photo.jpg", mode=ImageReadMode.GRAY)
print(gray.shape)   # (1, 480, 640)

# Write an image
write_image(image, "output.png")
```

---

## Image Reading

### read_image

Read an image file into a numpy array.

```python
from pyflame_vision.io import read_image, ImageReadMode

# Basic usage
image = read_image("photo.jpg")

# With specific mode
image = read_image("photo.jpg", mode=ImageReadMode.RGB)
gray = read_image("photo.jpg", mode=ImageReadMode.GRAY)
rgba = read_image("photo.png", mode=ImageReadMode.RGB_ALPHA)
```

**Parameters:**
- `path` (str | Path) - Path to image file
- `mode` (ImageReadMode) - How to interpret image data (default: UNCHANGED)

**Returns:**
- `np.ndarray` - Image array with shape [C, H, W] and dtype uint8

**Raises:**
- `FileNotFoundError` - If file doesn't exist
- `ValueError` - If file is not a valid image or exceeds size limits
- `PermissionError` - If file cannot be read
- `ImportError` - If PIL or numpy is not available

**Supported Formats:**
`.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.webp`, `.tiff`, `.tif`

---

### decode_image

Decode image bytes into a numpy array. Useful for in-memory images.

```python
from pyflame_vision.io import decode_image, ImageReadMode

# From file bytes
with open("photo.jpg", "rb") as f:
    data = f.read()
image = decode_image(data)

# From network response
import requests
response = requests.get(image_url)
image = decode_image(response.content, mode=ImageReadMode.RGB)
```

**Parameters:**
- `data` (bytes) - Raw image bytes (JPEG, PNG, etc.)
- `mode` (ImageReadMode) - How to interpret image data (default: UNCHANGED)

**Returns:**
- `np.ndarray` - Image array with shape [C, H, W] and dtype uint8

**Raises:**
- `ValueError` - If data is not a valid image or exceeds size limits
- `ImportError` - If PIL or numpy is not available

---

## Image Writing

### write_image

Write a numpy array to an image file.

```python
from pyflame_vision.io import write_image
import numpy as np

# Basic usage
image = np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8)
write_image(image, "output.png")

# With JPEG quality
write_image(image, "output.jpg", quality=95)

# Explicit format
write_image(image, "output", format="PNG")

# Grayscale image
gray = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
write_image(gray, "gray.png")
```

**Parameters:**
- `tensor` (np.ndarray) - Image array [C, H, W] or [H, W]
- `path` (str | Path) - Output file path
- `format` (str, optional) - Image format (inferred from path if None)
- `quality` (int) - JPEG/WebP quality 1-100 (default: 95)

**Raises:**
- `ValueError` - If tensor has invalid shape or quality is out of range
- `PermissionError` - If file cannot be written
- `ImportError` - If PIL or numpy is not available

**Supported Formats:**
- `JPEG` / `JPG` - Lossy compression, supports quality setting
- `PNG` - Lossless compression
- `BMP` - Uncompressed bitmap
- `GIF` - Limited to 256 colors
- `WEBP` - Modern format, supports quality setting
- `TIFF` - High-quality archival format

**Notes:**
- Parent directories are created automatically if they don't exist
- Float tensors with max value <= 1.0 are scaled to 0-255

---

### encode_image

Encode a numpy array to image bytes.

```python
from pyflame_vision.io import encode_image

# Encode to PNG bytes
png_bytes = encode_image(image, format="PNG")

# Encode to JPEG with quality
jpg_bytes = encode_image(image, format="JPEG", quality=90)

# Save to file or send over network
with open("output.png", "wb") as f:
    f.write(png_bytes)
```

**Parameters:**
- `tensor` (np.ndarray) - Image array [C, H, W] or [H, W]
- `format` (str) - Image format (default: "PNG")
- `quality` (int) - JPEG/WebP quality 1-100 (default: 95)

**Returns:**
- `bytes` - Encoded image data

**Raises:**
- `ValueError` - If tensor has invalid shape
- `ImportError` - If PIL or numpy is not available

---

## ImageReadMode

Enum specifying how to interpret image data when reading.

```python
from pyflame_vision.io import ImageReadMode

ImageReadMode.UNCHANGED   # Keep original format
ImageReadMode.GRAY        # Convert to grayscale [1, H, W]
ImageReadMode.GRAY_ALPHA  # Grayscale with alpha [2, H, W]
ImageReadMode.RGB         # Convert to RGB [3, H, W]
ImageReadMode.RGB_ALPHA   # RGB with alpha [4, H, W]
```

| Mode | Channels | Description |
|------|----------|-------------|
| `UNCHANGED` | Varies | Keep original format and channels |
| `GRAY` | 1 | Convert to grayscale |
| `GRAY_ALPHA` | 2 | Grayscale with alpha channel |
| `RGB` | 3 | Convert to RGB (discards alpha) |
| `RGB_ALPHA` | 4 | RGB with alpha channel |

---

## Tensor Format

All images use **[C, H, W]** format (channels-first):

```python
image = read_image("photo.jpg")
print(image.shape)  # (3, 480, 640)

channels = image.shape[0]  # 3
height = image.shape[1]    # 480
width = image.shape[2]     # 640

# Access a single channel
red_channel = image[0]     # shape: (480, 640)
```

This matches PyTorch's expected format and is consistent with PyFlameVision transforms.

---

## Security Limits

All I/O operations are validated against security limits.

```python
from pyflame_vision.io.io import IOSecurityLimits

IOSecurityLimits.MAX_IMAGE_WIDTH = 65536
IOSecurityLimits.MAX_IMAGE_HEIGHT = 65536
IOSecurityLimits.MAX_FILE_SIZE = 1 << 30  # 1 GB
IOSecurityLimits.MAX_PATH_LENGTH = 4096
```

**Validation Checks:**
- Path length validation
- File size validation
- Image dimension validation
- File existence and permissions
- Supported extension validation

---

## Dependencies

The io module requires:
- **NumPy** - For array operations
- **Pillow (PIL)** - For image decoding/encoding

Install with:
```bash
pip install numpy Pillow
```

---

## Complete Example

```python
from pyflame_vision import read_image, write_image, ImageReadMode
from pyflame_vision.io import decode_image, encode_image
import numpy as np

# Read image in different modes
rgb_image = read_image("input.jpg", mode=ImageReadMode.RGB)
gray_image = read_image("input.jpg", mode=ImageReadMode.GRAY)

print(f"RGB shape: {rgb_image.shape}")   # (3, H, W)
print(f"Gray shape: {gray_image.shape}") # (1, H, W)

# Process image (simple brightness adjustment)
brightened = np.clip(rgb_image.astype(np.int16) + 30, 0, 255).astype(np.uint8)

# Save processed image
write_image(brightened, "brightened.jpg", quality=95)
write_image(gray_image, "grayscale.png")

# Encode to bytes for transmission
png_bytes = encode_image(rgb_image, format="PNG")
jpg_bytes = encode_image(rgb_image, format="JPEG", quality=85)

print(f"PNG size: {len(png_bytes)} bytes")
print(f"JPEG size: {len(jpg_bytes)} bytes")

# Decode from bytes
decoded = decode_image(jpg_bytes, mode=ImageReadMode.RGB)
print(f"Decoded shape: {decoded.shape}")
```

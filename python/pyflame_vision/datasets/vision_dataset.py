"""
VisionDataset base class for image datasets.

Provides common functionality for image-based datasets.
"""

from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union
import os

from .dataset import Dataset


# ============================================================================
# Security Limits
# ============================================================================

class DatasetSecurityLimits:
    """Security limits for dataset operations."""
    MAX_PATH_LENGTH = 4096
    MAX_DIRECTORY_DEPTH = 100
    MAX_DATASET_SIZE = 1 << 30  # ~1 billion samples


def _validate_root_directory(root: Union[str, Path]) -> Path:
    """Validate dataset root directory.

    Args:
        root: Root directory path

    Returns:
        Validated Path object

    Raises:
        FileNotFoundError: If directory doesn't exist
        PermissionError: If directory isn't accessible
        ValueError: If path is too long
    """
    root = Path(root)

    # Check path length
    if len(str(root)) > DatasetSecurityLimits.MAX_PATH_LENGTH:
        raise ValueError(
            f"Path too long: {len(str(root))} > {DatasetSecurityLimits.MAX_PATH_LENGTH}"
        )

    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    if not os.access(root, os.R_OK | os.X_OK):
        raise PermissionError(f"Cannot access directory: {root}")

    return root


def _safe_path_join(root: Path, *parts: str) -> Path:
    """Safely join paths, preventing directory traversal attacks.

    Args:
        root: Root directory
        *parts: Path components to join

    Returns:
        Joined path

    Raises:
        ValueError: If resulting path escapes root
    """
    result = root
    for part in parts:
        # Security: Check for null bytes
        if '\x00' in part:
            raise ValueError(f"Null bytes not allowed in path: {part}")

        # Reject absolute paths in parts
        if os.path.isabs(part):
            raise ValueError(f"Absolute path not allowed: {part}")

        # Security: Normalize separators and check for parent directory references
        # This handles both Unix (/) and Windows (\) separators
        normalized = part.replace('\\', '/').split('/')
        if '..' in normalized:
            raise ValueError(f"Parent directory reference not allowed: {part}")

        # Security: Windows-specific checks
        if os.name == 'nt':
            # Check for alternate data streams (file.txt:stream)
            if ':' in part:
                # Allow drive letter (C:) but reject streams
                if not (len(part) >= 2 and part[1] == ':' and part[0].isalpha()):
                    raise ValueError(f"Alternate data streams not allowed: {part}")

            # Check for reserved device names
            reserved_names = {
                'CON', 'PRN', 'AUX', 'NUL',
                'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
                'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
            }
            # Get base name without extension
            base_name = part.split('.')[0].upper()
            if base_name in reserved_names:
                raise ValueError(f"Reserved device name not allowed: {part}")

        result = result / part

    # Verify result is under root
    try:
        result.resolve().relative_to(root.resolve())
    except ValueError:
        raise ValueError(f"Path escapes root directory: {result}")

    return result


class VisionDataset(Dataset):
    """Base class for image-based datasets.

    Provides common functionality for vision datasets:
        - Root directory handling
        - Transform application
        - Target transform application

    Args:
        root: Root directory of dataset
        transform: Transform to apply to images
        target_transform: Transform to apply to targets/labels

    Attributes:
        root: Path to dataset root directory
        transform: Image transform
        target_transform: Target transform

    Example:
        >>> class MyImageDataset(VisionDataset):
        ...     def __init__(self, root, transform=None):
        ...         super().__init__(root, transform=transform)
        ...         self.samples = self._load_samples()
        ...
        ...     def __getitem__(self, index):
        ...         image, target = self.samples[index]
        ...         return self._apply_transforms(image, target)
        ...
        ...     def __len__(self):
        ...         return len(self.samples)
    """

    _repr_indent = 4

    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        self.root = _validate_root_directory(root)
        self.transform = transform
        self.target_transform = target_transform

    def _apply_transforms(
        self,
        image: Any,
        target: Any
    ) -> Tuple[Any, Any]:
        """Apply transforms to image and target.

        Args:
            image: Input image
            target: Target/label

        Returns:
            Tuple of (transformed_image, transformed_target)
        """
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __repr__(self) -> str:
        head = f"Dataset {self.__class__.__name__}"
        body = [f"Number of datapoints: {len(self)}"]
        body.append(f"Root location: {self.root}")
        if self.transform is not None:
            body.append(f"Transform: {self.transform}")
        if self.target_transform is not None:
            body.append(f"Target transform: {self.target_transform}")
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def __getitem__(self, index: int) -> Any:
        """Get sample at index. Must be implemented by subclass."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return dataset length. Must be implemented by subclass."""
        raise NotImplementedError

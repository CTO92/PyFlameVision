"""
Collate functions for PyFlameVision DataLoader.

Provides functions to merge a list of samples into a mini-batch.
"""

from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union
import collections.abc

# Try to import numpy for array handling
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ============================================================================
# Security Limits
# ============================================================================

class CollateSecurityLimits:
    """Security limits for collate operations."""
    MAX_COLLATE_DEPTH = 100  # Maximum recursion depth for nested structures
    MAX_PAD_DIMENSION = 65536  # Maximum size for any padded dimension


def default_collate(batch: List[Any], _depth: int = 0) -> Any:
    """Default collation function for creating batches.

    Handles different data types:
        - NumPy arrays: Stack along new batch dimension (axis 0)
        - Numbers (int, float): Convert to list
        - Strings: Return as list
        - Mappings (dict): Recursively collate values
        - Sequences (tuple, list): Recursively collate elements

    Args:
        batch: List of samples to collate

    Returns:
        Batched data in the same structure as input samples

    Raises:
        TypeError: If element type is not supported
        ValueError: If batch is empty

    Example:
        >>> # Batch of (image, label) tuples
        >>> batch = [(np.zeros((3, 224, 224)), 0), (np.zeros((3, 224, 224)), 1)]
        >>> images, labels = default_collate(batch)
        >>> images.shape
        (2, 3, 224, 224)
        >>> labels
        [0, 1]
    """
    # Security: Prevent stack overflow from deeply nested structures
    if _depth > CollateSecurityLimits.MAX_COLLATE_DEPTH:
        raise ValueError(
            f"Collation depth ({_depth}) exceeds maximum "
            f"({CollateSecurityLimits.MAX_COLLATE_DEPTH}). "
            "Data structure may be too deeply nested."
        )

    if len(batch) == 0:
        raise ValueError("Cannot collate empty batch")

    elem = batch[0]
    elem_type = type(elem)

    # Handle numpy arrays
    if HAS_NUMPY and isinstance(elem, np.ndarray):
        # Stack along new batch dimension
        try:
            return np.stack(batch, axis=0)
        except ValueError as e:
            # Arrays have different shapes - can't stack
            raise ValueError(
                f"Cannot stack arrays with different shapes: {[b.shape for b in batch]}"
            ) from e

    # Handle numeric types
    elif isinstance(elem, (int, float, complex)):
        return batch  # Return as list

    # Handle boolean
    elif isinstance(elem, bool):
        return batch

    # Handle strings
    elif isinstance(elem, str):
        return batch

    # Handle bytes
    elif isinstance(elem, bytes):
        return batch

    # Handle None
    elif elem is None:
        return batch

    # Handle mappings (dict-like)
    elif isinstance(elem, Mapping):
        try:
            # Recursively collate each key
            return {key: default_collate([d[key] for d in batch], _depth + 1) for key in elem}
        except KeyError as e:
            raise ValueError(f"Inconsistent dict keys in batch") from e

    # Handle tuples (including namedtuple)
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
        # Named tuple
        return elem_type(*(default_collate(list(samples), _depth + 1) for samples in zip(*batch)))

    elif isinstance(elem, tuple):
        # Regular tuple - collate each position
        return tuple(default_collate(list(samples), _depth + 1) for samples in zip(*batch))

    # Handle sequences (list-like, but not string/bytes)
    elif isinstance(elem, collections.abc.Sequence):
        # Transpose and collate
        transposed = list(zip(*batch))
        return [default_collate(list(samples), _depth + 1) for samples in transposed]

    else:
        # Unknown type - return as list
        return batch


def pad_collate(
    batch: List[Any],
    padding_value: float = 0.0,
    padding_mode: str = "end",
    _depth: int = 0
) -> Any:
    """Collate tensors with different sizes by padding.

    Useful for batching images of different sizes or variable-length sequences.
    Pads all tensors to the maximum size in the batch.

    Args:
        batch: List of numpy arrays to collate
        padding_value: Value to use for padding (default: 0.0)
        padding_mode: Where to add padding - "end" or "start" (default: "end")

    Returns:
        Padded and stacked array of shape [batch_size, *max_shape]

    Raises:
        ValueError: If batch is empty or arrays have different numbers of dimensions
        ImportError: If numpy is not available

    Example:
        >>> # Variable size images
        >>> batch = [np.zeros((3, 100, 100)), np.zeros((3, 150, 120))]
        >>> padded = pad_collate(batch)
        >>> padded.shape
        (2, 3, 150, 120)
    """
    # Security: Prevent stack overflow from deeply nested structures
    if _depth > CollateSecurityLimits.MAX_COLLATE_DEPTH:
        raise ValueError(
            f"Collation depth ({_depth}) exceeds maximum "
            f"({CollateSecurityLimits.MAX_COLLATE_DEPTH}). "
            "Data structure may be too deeply nested."
        )

    if not HAS_NUMPY:
        raise ImportError("NumPy is required for pad_collate")

    if len(batch) == 0:
        raise ValueError("Cannot collate empty batch")

    if padding_mode not in ("end", "start"):
        raise ValueError(f"padding_mode must be 'end' or 'start', got {padding_mode}")

    elem = batch[0]

    # Handle numpy arrays
    if isinstance(elem, np.ndarray):
        # Check all have same number of dimensions
        ndims = [arr.ndim for arr in batch]
        if len(set(ndims)) > 1:
            raise ValueError(
                f"All arrays must have same number of dimensions. Got: {ndims}"
            )

        # Find maximum size for each dimension
        max_shape = [max(arr.shape[i] for arr in batch) for i in range(elem.ndim)]

        # Security: Validate max dimensions to prevent memory exhaustion
        for i, dim in enumerate(max_shape):
            if dim > CollateSecurityLimits.MAX_PAD_DIMENSION:
                raise ValueError(
                    f"Padded dimension {i} ({dim}) exceeds maximum "
                    f"({CollateSecurityLimits.MAX_PAD_DIMENSION})"
                )

        # Pad each array
        padded = []
        for arr in batch:
            if arr.shape == tuple(max_shape):
                padded.append(arr)
            else:
                # Create padded array filled with padding_value
                padded_arr = np.full(max_shape, padding_value, dtype=arr.dtype)

                # Build slices for inserting original array
                if padding_mode == "end":
                    slices = tuple(slice(0, s) for s in arr.shape)
                else:  # start
                    slices = tuple(slice(m - s, m) for s, m in zip(arr.shape, max_shape))

                padded_arr[slices] = arr
                padded.append(padded_arr)

        return np.stack(padded, axis=0)

    # Handle tuples
    elif isinstance(elem, tuple):
        return tuple(pad_collate(list(samples), padding_value, padding_mode, _depth + 1) for samples in zip(*batch))

    # Handle lists
    elif isinstance(elem, list):
        return [pad_collate(list(samples), padding_value, padding_mode, _depth + 1) for samples in zip(*batch)]

    # For other types, fall back to default_collate
    else:
        return default_collate(batch, _depth + 1)


def stack_collate(batch: List[Any], axis: int = 0, _depth: int = 0) -> Any:
    """Simple stack collation - just stacks arrays along specified axis.

    Args:
        batch: List of numpy arrays
        axis: Axis along which to stack (default: 0)

    Returns:
        Stacked array

    Example:
        >>> batch = [np.array([1, 2]), np.array([3, 4])]
        >>> stack_collate(batch)
        array([[1, 2], [3, 4]])
    """
    # Security: Prevent stack overflow from deeply nested structures
    if _depth > CollateSecurityLimits.MAX_COLLATE_DEPTH:
        raise ValueError(
            f"Collation depth ({_depth}) exceeds maximum "
            f"({CollateSecurityLimits.MAX_COLLATE_DEPTH}). "
            "Data structure may be too deeply nested."
        )

    if not HAS_NUMPY:
        raise ImportError("NumPy is required for stack_collate")

    if len(batch) == 0:
        raise ValueError("Cannot collate empty batch")

    elem = batch[0]

    if isinstance(elem, np.ndarray):
        return np.stack(batch, axis=axis)
    elif isinstance(elem, tuple):
        return tuple(stack_collate(list(samples), axis, _depth + 1) for samples in zip(*batch))
    elif isinstance(elem, list):
        return [stack_collate(list(samples), axis, _depth + 1) for samples in zip(*batch)]
    else:
        return default_collate(batch, _depth + 1)


def no_collate(batch: List[Any]) -> List[Any]:
    """No-op collation - returns batch as-is.

    Useful when you want to handle batching yourself or need
    samples without any transformation.

    Args:
        batch: List of samples

    Returns:
        The batch unchanged

    Example:
        >>> batch = [{"path": "a.jpg"}, {"path": "b.jpg"}]
        >>> no_collate(batch)
        [{"path": "a.jpg"}, {"path": "b.jpg"}]
    """
    return batch

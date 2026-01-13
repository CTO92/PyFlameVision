"""
ImageFolder and DatasetFolder implementations.

Provides directory-based dataset loading where subdirectories represent classes.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os

from .vision_dataset import VisionDataset, DatasetSecurityLimits


# Default image extensions
IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.bmp', '.gif',
    '.webp', '.tiff', '.tif', '.ppm', '.pgm', '.pbm'
)


def _has_valid_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Check if filename has a valid extension.

    Args:
        filename: Filename to check
        extensions: Tuple of valid extensions (lowercase with dot)

    Returns:
        True if extension is valid
    """
    return filename.lower().endswith(extensions)


def _find_classes(directory: Path) -> Tuple[List[str], Dict[str, int]]:
    """Find class subdirectories in a directory.

    Args:
        directory: Root directory to search

    Returns:
        Tuple of (class_names, class_to_idx mapping)

    Raises:
        FileNotFoundError: If no valid classes found
    """
    classes = sorted([
        d.name for d in directory.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])

    if not classes:
        raise FileNotFoundError(
            f"No class subdirectories found in {directory}. "
            "Expected directory structure: root/class_name/images..."
        )

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    return classes, class_to_idx


def _make_dataset(
    directory: Path,
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
    max_samples: Optional[int] = None
) -> List[Tuple[Path, int]]:
    """Build list of (path, class_idx) samples.

    Args:
        directory: Root directory
        class_to_idx: Class name to index mapping
        extensions: Valid file extensions
        is_valid_file: Custom file validation function
        max_samples: Maximum number of samples (for security)

    Returns:
        List of (file_path, class_index) tuples
    """
    if extensions is None:
        extensions = IMG_EXTENSIONS

    if max_samples is None:
        max_samples = DatasetSecurityLimits.MAX_DATASET_SIZE

    samples = []
    for class_name, class_idx in sorted(class_to_idx.items()):
        class_dir = directory / class_name

        if not class_dir.is_dir():
            continue

        # Security: Skip symlinked class directories
        if class_dir.is_symlink():
            continue

        # Security: Don't follow symlinks in os.walk to prevent directory escape
        for root, _, filenames in os.walk(class_dir, followlinks=False):
            root_path = Path(root)

            # Security: Skip symlinked directories
            if root_path.is_symlink():
                continue

            # Security: Verify path hasn't escaped via symlink resolution
            try:
                root_path.resolve().relative_to(directory.resolve())
            except ValueError:
                continue  # Path escaped root directory

            # Security: Limit directory depth
            try:
                depth = len(root_path.relative_to(directory).parts)
                if depth > DatasetSecurityLimits.MAX_DIRECTORY_DEPTH:
                    continue
            except ValueError:
                continue

            for filename in sorted(filenames):
                if len(samples) >= max_samples:
                    break

                file_path = root_path / filename

                # Security: Skip symlinked files
                if file_path.is_symlink():
                    continue

                # Validate file
                if is_valid_file is not None:
                    if is_valid_file(str(file_path)):
                        samples.append((file_path, class_idx))
                elif _has_valid_extension(filename, extensions):
                    samples.append((file_path, class_idx))

            if len(samples) >= max_samples:
                break

        if len(samples) >= max_samples:
            break

    return samples


def _default_loader(path: Path) -> Any:
    """Default image loader using PIL.

    Args:
        path: Path to image file

    Returns:
        PIL Image in RGB mode
    """
    try:
        from PIL import Image
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except ImportError:
        raise ImportError(
            "PIL (Pillow) is required for ImageFolder. "
            "Install with: pip install Pillow"
        )


class DatasetFolder(VisionDataset):
    """Generic dataset for files organized in class subdirectories.

    Expected directory structure:
        root/
        ├── class_a/
        │   ├── file1.ext
        │   ├── file2.ext
        │   └── ...
        ├── class_b/
        │   ├── file1.ext
        │   └── ...
        └── ...

    Args:
        root: Root directory path
        loader: Function to load a sample from path
        extensions: Valid file extensions
        transform: Transform to apply to samples
        target_transform: Transform to apply to labels
        is_valid_file: Custom file validation function

    Attributes:
        classes: List of class names (subdirectory names)
        class_to_idx: Dict mapping class name to index
        samples: List of (path, class_index) tuples
        targets: List of class indices

    Example:
        >>> def my_loader(path):
        ...     return open(path).read()
        >>> dataset = DatasetFolder("data", loader=my_loader, extensions=(".txt",))
    """

    def __init__(
        self,
        root: Union[str, Path],
        loader: Callable[[Path], Any],
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.loader = loader
        self.extensions = extensions

        classes, class_to_idx = _find_classes(self.root)
        samples = _make_dataset(
            self.root,
            class_to_idx,
            extensions,
            is_valid_file
        )

        if len(samples) == 0:
            msg = f"Found 0 files in subdirectories of: {self.root}\n"
            if extensions is not None:
                msg += f"Supported extensions: {extensions}"
            raise RuntimeError(msg)

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """Get sample and target at index.

        Args:
            index: Sample index

        Returns:
            Tuple of (sample, target) where target is the class index
        """
        if index < 0:
            if -index > len(self):
                raise IndexError(f"Index {index} out of range")
            index = len(self) + index

        if index >= len(self.samples):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")

        path, target = self.samples[index]
        sample = self.loader(path)
        sample, target = self._apply_transforms(sample, target)
        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

    def find_classes(self, directory: Union[str, Path]) -> Tuple[List[str], Dict[str, int]]:
        """Find classes in directory. Can be overridden for custom behavior."""
        return _find_classes(Path(directory))


class ImageFolder(DatasetFolder):
    """Dataset for images organized in class subdirectories.

    Expected directory structure:
        root/
        ├── dog/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        ├── cat/
        │   ├── image1.jpg
        │   └── ...
        └── ...

    The class names are inferred from subdirectory names.

    Args:
        root: Root directory path
        transform: Transform to apply to images
        target_transform: Transform to apply to labels
        loader: Function to load images (default: PIL loader)
        is_valid_file: Custom file validation function

    Attributes:
        classes: List of class names (subdirectory names)
        class_to_idx: Dict mapping class name to index
        samples: List of (path, class_index) tuples
        targets: List of class indices
        imgs: Alias for samples (for compatibility)

    Example:
        >>> from pyflame_vision import transforms as T
        >>> transform = T.Compose([
        ...     T.Resize(256),
        ...     T.CenterCrop(224),
        ...     T.ToTensor(),
        ...     T.Normalize(mean=[0.485, 0.456, 0.406],
        ...                 std=[0.229, 0.224, 0.225])
        ... ])
        >>> dataset = ImageFolder("data/train", transform=transform)
        >>> image, label = dataset[0]
        >>> print(f"Classes: {dataset.classes}")
        >>> print(f"Number of images: {len(dataset)}")
    """

    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[Path], Any] = _default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None
    ) -> None:
        super().__init__(
            root,
            loader=loader,
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file
        )
        # Alias for compatibility
        self.imgs = self.samples

    def __repr__(self) -> str:
        head = f"Dataset {self.__class__.__name__}"
        body = [
            f"Number of datapoints: {len(self)}",
            f"Root location: {self.root}",
            f"Number of classes: {len(self.classes)}"
        ]
        if self.transform is not None:
            body.append(f"Transform: {self.transform}")
        if self.target_transform is not None:
            body.append(f"Target transform: {self.target_transform}")
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

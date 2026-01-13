# Datasets API Reference

PyFlameVision provides PyTorch-compatible dataset and data loading utilities for efficient training pipelines.

## Quick Start

```python
from pyflame_vision import datasets, transforms as T

# Create dataset with transforms
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder("data/train", transform=transform)

# Create DataLoader
loader = datasets.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Iterate over batches
for images, labels in loader:
    # images: [32, 3, 224, 224]
    # labels: [32]
    output = model(images)
```

---

## Base Dataset Classes

### Dataset

Abstract base class for map-style datasets that support random access.

```python
from pyflame_vision.datasets import Dataset

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
```

**Abstract Methods:**
- `__getitem__(index: int) -> T` - Get sample at index
- `__len__() -> int` - Return dataset length

**Operators:**
- `+` - Concatenate datasets: `combined = dataset1 + dataset2`

---

### IterableDataset

Abstract base class for iterable-style datasets that support sequential access.

```python
from pyflame_vision.datasets import IterableDataset

class StreamingDataset(IterableDataset):
    def __init__(self, url):
        self.url = url

    def __iter__(self):
        for chunk in stream_from_url(self.url):
            yield process(chunk)
```

**Abstract Methods:**
- `__iter__() -> Iterator[T]` - Iterate over samples

**Operators:**
- `+` - Chain datasets: `combined = dataset1 + dataset2`

---

### VisionDataset

Base class for image-based datasets with transform support.

```python
from pyflame_vision.datasets import VisionDataset

class MyImageDataset(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)
        self.samples = self._load_samples()

    def __getitem__(self, index):
        image, target = self.samples[index]
        return self._apply_transforms(image, target)

    def __len__(self):
        return len(self.samples)
```

**Parameters:**
- `root` (str | Path) - Root directory of dataset
- `transform` (Callable, optional) - Transform to apply to images
- `target_transform` (Callable, optional) - Transform to apply to targets

**Attributes:**
- `root` - Path to dataset root
- `transform` - Image transform
- `target_transform` - Target transform

**Methods:**
- `_apply_transforms(image, target)` - Apply transforms to image and target

---

## Dataset Utilities

### ConcatDataset

Concatenate multiple datasets into one.

```python
from pyflame_vision.datasets import ConcatDataset

combined = ConcatDataset([dataset1, dataset2, dataset3])
print(len(combined))  # len(d1) + len(d2) + len(d3)
```

---

### Subset

Create a subset of a dataset at specified indices.

```python
from pyflame_vision.datasets import Subset

# Split dataset manually
train_indices = list(range(0, 800))
val_indices = list(range(800, 1000))

train_set = Subset(full_dataset, train_indices)
val_set = Subset(full_dataset, val_indices)
```

---

### random_split

Randomly split a dataset into non-overlapping subsets.

```python
from pyflame_vision.datasets import random_split

# Split 1000 samples into 800/100/100
train_set, val_set, test_set = random_split(dataset, [800, 100, 100])

# With reproducibility
import random
gen = random.Random(42)
train_set, val_set = random_split(dataset, [800, 200], generator=gen)
```

**Parameters:**
- `dataset` (Dataset) - Dataset to split
- `lengths` (Sequence[int]) - Lengths of splits (must sum to len(dataset))
- `generator` (random.Random, optional) - RNG for reproducibility

---

## Image Datasets

### ImageFolder

Dataset for images organized in class subdirectories.

```
data/train/
├── dog/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── cat/
│   ├── image1.jpg
│   └── ...
└── ...
```

```python
from pyflame_vision.datasets import ImageFolder

dataset = ImageFolder(
    root="data/train",
    transform=transform,
    target_transform=None,
    loader=None,  # Default PIL loader
    is_valid_file=None
)

# Access class information
print(dataset.classes)      # ['cat', 'dog', ...]
print(dataset.class_to_idx) # {'cat': 0, 'dog': 1, ...}
print(len(dataset))         # Number of images

# Get sample
image, label = dataset[0]
```

**Parameters:**
- `root` (str | Path) - Root directory
- `transform` (Callable, optional) - Image transform
- `target_transform` (Callable, optional) - Label transform
- `loader` (Callable, optional) - Function to load images
- `is_valid_file` (Callable, optional) - Custom file validation

**Attributes:**
- `classes` - List of class names
- `class_to_idx` - Dict mapping class name to index
- `samples` - List of (path, class_idx) tuples
- `targets` - List of class indices
- `imgs` - Alias for samples

**Supported Extensions:**
`.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.webp`, `.tiff`, `.tif`, `.ppm`, `.pgm`, `.pbm`

---

### DatasetFolder

Generic dataset for files organized in class subdirectories.

```python
from pyflame_vision.datasets import DatasetFolder

def my_loader(path):
    with open(path) as f:
        return f.read()

dataset = DatasetFolder(
    root="data",
    loader=my_loader,
    extensions=(".txt", ".json")
)
```

---

## Samplers

Samplers define the order in which dataset indices are accessed.

### SequentialSampler

Sample elements sequentially, always in the same order.

```python
from pyflame_vision.datasets import SequentialSampler

sampler = SequentialSampler(dataset)
list(sampler)  # [0, 1, 2, 3, ...]
```

---

### RandomSampler

Sample elements randomly.

```python
from pyflame_vision.datasets import RandomSampler
import random

# Without replacement (default) - each index appears once
sampler = RandomSampler(dataset)

# With replacement - allows duplicates
sampler = RandomSampler(dataset, replacement=True, num_samples=1000)

# With reproducibility
gen = random.Random(42)
sampler = RandomSampler(dataset, generator=gen)
```

**Parameters:**
- `data_source` (Sized) - Dataset to sample from
- `replacement` (bool) - Sample with replacement (default: False)
- `num_samples` (int, optional) - Number of samples (default: len(data_source))
- `generator` (random.Random, optional) - RNG for reproducibility

---

### SubsetRandomSampler

Sample randomly from a subset of indices.

```python
from pyflame_vision.datasets import SubsetRandomSampler

# Sample only from these indices
indices = [0, 5, 10, 15, 20]
sampler = SubsetRandomSampler(indices)
```

---

### WeightedRandomSampler

Sample with given probabilities (weights). Useful for handling class imbalance.

```python
from pyflame_vision.datasets import WeightedRandomSampler

# Handle imbalanced dataset: class 0 has 1000 samples, class 1 has 100
class_counts = [1000, 100]
weights = [1.0 / class_counts[label] for label in all_labels]

sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=len(weights),
    replacement=True
)
```

**Parameters:**
- `weights` (Sequence[float]) - Weight for each sample
- `num_samples` (int) - Number of samples to draw
- `replacement` (bool) - Sample with replacement (default: True)
- `generator` (random.Random, optional) - RNG for reproducibility

---

### BatchSampler

Wrap a sampler to yield batches of indices.

```python
from pyflame_vision.datasets import SequentialSampler, BatchSampler

sampler = SequentialSampler(dataset)
batch_sampler = BatchSampler(sampler, batch_size=4, drop_last=False)

list(batch_sampler)  # [[0,1,2,3], [4,5,6,7], [8,9]]
```

**Parameters:**
- `sampler` (Sampler) - Base sampler
- `batch_size` (int) - Batch size
- `drop_last` (bool) - Drop last incomplete batch (default: False)

---

## DataLoader

Combines dataset and sampler to provide batched iteration.

```python
from pyflame_vision.datasets import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    sampler=None,           # Custom sampler (exclusive with shuffle)
    batch_sampler=None,     # Custom batch sampler
    num_workers=4,          # Parallel loading threads
    collate_fn=None,        # Custom batch creation function
    drop_last=False,        # Drop incomplete last batch
    timeout=0,              # Worker timeout in seconds
    prefetch_factor=2       # Batches to prefetch per worker
)

# Iterate
for batch in loader:
    process(batch)

# Get length
num_batches = len(loader)
```

**Parameters:**
- `dataset` (Dataset | IterableDataset) - Dataset to load from
- `batch_size` (int) - Samples per batch (default: 1)
- `shuffle` (bool) - Shuffle at each epoch (default: False)
- `sampler` (Sampler, optional) - Custom sampler
- `batch_sampler` (Sampler, optional) - Custom batch sampler
- `num_workers` (int) - Worker threads (default: 0 = main thread)
- `collate_fn` (Callable, optional) - Batch creation function
- `drop_last` (bool) - Drop incomplete batch (default: False)
- `timeout` (float) - Worker timeout in seconds (default: 0 = infinite)
- `prefetch_factor` (int) - Batches to prefetch per worker (default: 2)

**Notes:**
- `shuffle` and `sampler` are mutually exclusive
- `batch_sampler` overrides `batch_size`, `shuffle`, `sampler`, `drop_last`
- When `num_workers > 0`, data loading uses thread workers

---

## Collate Functions

Functions to merge samples into batches.

### default_collate

Default collation that handles various data types.

```python
from pyflame_vision.datasets import default_collate

# NumPy arrays are stacked
batch = [np.zeros((3, 224, 224)), np.zeros((3, 224, 224))]
result = default_collate(batch)  # shape: (2, 3, 224, 224)

# Tuples are recursively collated
batch = [(image1, label1), (image2, label2)]
images, labels = default_collate(batch)

# Dicts are recursively collated
batch = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
result = default_collate(batch)  # {"x": [1, 3], "y": [2, 4]}
```

---

### pad_collate

Collate tensors with different sizes by padding to max size.

```python
from pyflame_vision.datasets import pad_collate

# Variable size images
batch = [np.zeros((3, 100, 100)), np.zeros((3, 150, 120))]
result = pad_collate(batch)  # shape: (2, 3, 150, 120)

# Custom padding value
result = pad_collate(batch, padding_value=-1.0)
```

**Parameters:**
- `batch` (List) - List of arrays to collate
- `padding_value` (float) - Value for padding (default: 0.0)
- `padding_mode` (str) - "end" or "start" (default: "end")

---

### stack_collate

Simple stack along specified axis.

```python
from pyflame_vision.datasets.collate import stack_collate

result = stack_collate(batch, axis=0)
```

---

### no_collate

No-op collation - returns batch unchanged.

```python
from pyflame_vision.datasets.collate import no_collate

# Keep samples as list without transformation
loader = DataLoader(dataset, collate_fn=no_collate)
```

---

## Security Limits

All dataset operations are validated against security limits.

```python
# Dataset limits
DatasetSecurityLimits.MAX_PATH_LENGTH = 4096
DatasetSecurityLimits.MAX_DIRECTORY_DEPTH = 100
DatasetSecurityLimits.MAX_DATASET_SIZE = 1 << 30  # ~1 billion

# DataLoader limits
DataLoaderSecurityLimits.MAX_NUM_WORKERS = 64
DataLoaderSecurityLimits.MAX_BATCH_SIZE = 65536
DataLoaderSecurityLimits.MAX_PREFETCH_FACTOR = 100
DataLoaderSecurityLimits.MAX_TIMEOUT = 3600  # 1 hour
```

---

## Complete Example

```python
from pyflame_vision import datasets, transforms as T, read_image

# Define transforms
train_transform = T.Compose([
    T.Resize(256),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = datasets.ImageFolder("data/train", transform=train_transform)
val_dataset = datasets.ImageFolder("data/val", transform=val_transform)

# Create data loaders
train_loader = datasets.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    drop_last=True
)

val_loader = datasets.DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)

# Training loop
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Train step
        pass

    for images, labels in val_loader:
        # Validation step
        pass
```

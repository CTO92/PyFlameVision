"""
PyFlameVision Datasets Module.

This module provides dataset abstractions for loading image data.
"""

from .dataset import Dataset, IterableDataset, ConcatDataset, Subset, random_split
from .vision_dataset import VisionDataset
from .image_folder import ImageFolder, DatasetFolder
from .samplers import (
    Sampler,
    SequentialSampler,
    RandomSampler,
    SubsetRandomSampler,
    WeightedRandomSampler,
    BatchSampler,
)
from .collate import (
    default_collate,
    pad_collate,
    stack_collate,
    no_collate,
)
from .dataloader import DataLoader, DataLoaderIterator

__all__ = [
    # Base classes
    "Dataset",
    "IterableDataset",
    "ConcatDataset",
    "Subset",
    "random_split",
    "VisionDataset",
    # Datasets
    "ImageFolder",
    "DatasetFolder",
    # Samplers
    "Sampler",
    "SequentialSampler",
    "RandomSampler",
    "SubsetRandomSampler",
    "WeightedRandomSampler",
    "BatchSampler",
    # Collate functions
    "default_collate",
    "pad_collate",
    "stack_collate",
    "no_collate",
    # DataLoader
    "DataLoader",
    "DataLoaderIterator",
]

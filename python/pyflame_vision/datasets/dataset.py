"""
Base Dataset classes for PyFlameVision.

Provides PyTorch-compatible Dataset and IterableDataset abstractions.
"""

from abc import ABC, abstractmethod
from typing import Any, Iterator, List, TypeVar, Generic, Optional, Sequence
import bisect
import random

T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)


class Dataset(ABC, Generic[T_co]):
    """Abstract base class for map-style datasets.

    A map-style dataset is one that implements __getitem__ and __len__.
    It allows random access by index.

    Subclasses must implement:
        __getitem__(index) -> sample
        __len__() -> int

    Example:
        >>> class MyDataset(Dataset):
        ...     def __init__(self, data):
        ...         self.data = data
        ...     def __getitem__(self, index):
        ...         return self.data[index]
        ...     def __len__(self):
        ...         return len(self.data)
    """

    @abstractmethod
    def __getitem__(self, index: int) -> T_co:
        """Get sample at index.

        Args:
            index: Sample index

        Returns:
            Sample at the given index
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """Return dataset length.

        Returns:
            Number of samples in dataset
        """
        raise NotImplementedError

    def __add__(self, other: "Dataset[T_co]") -> "ConcatDataset[T_co]":
        """Concatenate two datasets.

        Args:
            other: Another dataset to concatenate

        Returns:
            ConcatDataset containing both datasets
        """
        return ConcatDataset([self, other])


class IterableDataset(ABC, Generic[T_co]):
    """Abstract base class for iterable-style datasets.

    An iterable-style dataset is one that implements __iter__.
    It is designed for streaming data that doesn't fit in memory
    or data that requires sequential access.

    Subclasses must implement:
        __iter__() -> Iterator[sample]

    Example:
        >>> class StreamingDataset(IterableDataset):
        ...     def __init__(self, url):
        ...         self.url = url
        ...     def __iter__(self):
        ...         for chunk in stream_from_url(self.url):
        ...             yield process(chunk)
    """

    @abstractmethod
    def __iter__(self) -> Iterator[T_co]:
        """Iterate over samples.

        Returns:
            Iterator yielding samples
        """
        raise NotImplementedError

    def __add__(self, other: "IterableDataset[T_co]") -> "ChainDataset[T_co]":
        """Chain two iterable datasets.

        Args:
            other: Another iterable dataset to chain

        Returns:
            ChainDataset containing both datasets
        """
        return ChainDataset([self, other])


class ConcatDataset(Dataset[T_co]):
    """Dataset that concatenates multiple datasets.

    Allows treating multiple datasets as a single dataset.

    Args:
        datasets: List of datasets to concatenate

    Example:
        >>> ds1 = MyDataset(data1)
        >>> ds2 = MyDataset(data2)
        >>> combined = ConcatDataset([ds1, ds2])
        >>> len(combined) == len(ds1) + len(ds2)
        True
    """

    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    def __init__(self, datasets: Sequence[Dataset[T_co]]) -> None:
        super().__init__()
        if len(datasets) == 0:
            raise ValueError("ConcatDataset requires at least one dataset")

        self.datasets = list(datasets)
        self.cumulative_sizes = self._cumsum(self.datasets)

    @staticmethod
    def _cumsum(datasets: List[Dataset]) -> List[int]:
        """Compute cumulative sum of dataset lengths."""
        result = []
        total = 0
        for ds in datasets:
            total += len(ds)
            result.append(total)
        return result

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> T_co:
        if idx < 0:
            if -idx > len(self):
                raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
            idx = len(self) + idx

        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        # Find which dataset contains this index
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)

        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx][sample_idx]


class ChainDataset(IterableDataset[T_co]):
    """Dataset that chains multiple iterable datasets.

    Args:
        datasets: List of iterable datasets to chain

    Example:
        >>> ds1 = StreamingDataset(url1)
        >>> ds2 = StreamingDataset(url2)
        >>> combined = ChainDataset([ds1, ds2])
        >>> for sample in combined:
        ...     process(sample)
    """

    def __init__(self, datasets: Sequence[IterableDataset[T_co]]) -> None:
        super().__init__()
        self.datasets = list(datasets)

    def __iter__(self) -> Iterator[T_co]:
        for dataset in self.datasets:
            yield from dataset


class Subset(Dataset[T_co]):
    """Subset of a dataset at specified indices.

    Args:
        dataset: The full dataset
        indices: Indices in the full dataset to include

    Example:
        >>> full_dataset = MyDataset(data)
        >>> train_indices = list(range(0, 800))
        >>> val_indices = list(range(800, 1000))
        >>> train_dataset = Subset(full_dataset, train_indices)
        >>> val_dataset = Subset(full_dataset, val_indices)
    """

    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx: int) -> T_co:
        if idx < 0:
            if -idx > len(self):
                raise IndexError(f"Index {idx} out of range")
            idx = len(self) + idx
        if idx >= len(self.indices):
            raise IndexError(f"Index {idx} out of range for subset of size {len(self.indices)}")
        return self.dataset[self.indices[idx]]

    def __len__(self) -> int:
        return len(self.indices)


def random_split(
    dataset: Dataset[T],
    lengths: Sequence[int],
    generator: Optional[random.Random] = None
) -> List[Subset[T]]:
    """Randomly split a dataset into non-overlapping subsets.

    Args:
        dataset: Dataset to split
        lengths: Lengths of splits to create
        generator: Random number generator for reproducibility

    Returns:
        List of Subset datasets

    Example:
        >>> dataset = MyDataset(data)  # len = 1000
        >>> train_set, val_set, test_set = random_split(dataset, [800, 100, 100])
    """
    if sum(lengths) != len(dataset):
        raise ValueError(
            f"Sum of split lengths ({sum(lengths)}) must equal "
            f"dataset length ({len(dataset)})"
        )

    # Create shuffled indices
    indices = list(range(len(dataset)))
    if generator is not None:
        generator.shuffle(indices)
    else:
        random.shuffle(indices)

    # Create subsets
    subsets = []
    offset = 0
    for length in lengths:
        subset_indices = indices[offset:offset + length]
        subsets.append(Subset(dataset, subset_indices))
        offset += length

    return subsets

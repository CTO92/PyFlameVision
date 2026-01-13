"""
Samplers for PyFlameVision DataLoader.

Provides various sampling strategies for iterating over datasets.
"""

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Sequence, Sized
import random
import math


# ============================================================================
# Security Limits
# ============================================================================

class SamplerSecurityLimits:
    """Security limits for sampler operations."""
    MAX_NUM_SAMPLES = 1 << 30  # ~1 billion samples maximum


class Sampler(ABC):
    """Base class for all samplers.

    Samplers generate indices to access dataset samples.
    They define the order in which samples are accessed during training.

    Subclasses must implement:
        __iter__() -> Iterator[int]
        __len__() -> int
    """

    @abstractmethod
    def __iter__(self) -> Iterator[int]:
        """Iterate over sample indices.

        Returns:
            Iterator yielding sample indices
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """Return number of samples.

        Returns:
            Number of samples that will be yielded
        """
        raise NotImplementedError


class SequentialSampler(Sampler):
    """Sample elements sequentially, always in the same order.

    Args:
        data_source: Dataset to sample from

    Example:
        >>> sampler = SequentialSampler(dataset)
        >>> list(sampler)
        [0, 1, 2, 3, 4, ...]
    """

    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler):
    """Sample elements randomly.

    If replacement is False (default), samples without replacement,
    giving a random permutation of indices.

    If replacement is True, samples with replacement, allowing
    the same index to be sampled multiple times.

    Args:
        data_source: Dataset to sample from
        replacement: If True, sample with replacement
        num_samples: Number of samples to draw (default: len(data_source))
        generator: Random generator for reproducibility

    Example:
        >>> sampler = RandomSampler(dataset)
        >>> list(sampler)  # Random order
        [3, 1, 4, 0, 2, ...]

        >>> sampler = RandomSampler(dataset, replacement=True, num_samples=10)
        >>> list(sampler)  # May have duplicates
        [2, 2, 4, 1, 3, 4, 0, 2, 1, 3]
    """

    data_source: Sized
    replacement: bool
    _num_samples: Optional[int]
    generator: Optional[random.Random]

    def __init__(
        self,
        data_source: Sized,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        generator: Optional[random.Random] = None
    ) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        # Security: Validate num_samples upper bound
        if num_samples is not None:
            if num_samples > SamplerSecurityLimits.MAX_NUM_SAMPLES:
                raise ValueError(
                    f"num_samples ({num_samples}) exceeds maximum "
                    f"({SamplerSecurityLimits.MAX_NUM_SAMPLES})"
                )

        if not replacement and num_samples is not None:
            if num_samples > len(data_source):
                raise ValueError(
                    f"num_samples ({num_samples}) cannot exceed dataset size "
                    f"({len(data_source)}) when replacement=False"
                )

    @property
    def num_samples(self) -> int:
        """Return number of samples to draw."""
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        gen = self.generator if self.generator else random

        if self.replacement:
            for _ in range(self.num_samples):
                yield gen.randint(0, n - 1)
        else:
            indices = list(range(n))
            gen.shuffle(indices)
            for i in range(self.num_samples):
                yield indices[i]

    def __len__(self) -> int:
        return self.num_samples


class SubsetRandomSampler(Sampler):
    """Sample randomly from a subset of indices.

    Useful for train/validation splits.

    Args:
        indices: Sequence of indices to sample from
        generator: Random generator for reproducibility

    Example:
        >>> train_indices = list(range(0, 800))
        >>> sampler = SubsetRandomSampler(train_indices)
    """

    indices: Sequence[int]
    generator: Optional[random.Random]

    def __init__(
        self,
        indices: Sequence[int],
        generator: Optional[random.Random] = None
    ) -> None:
        # Security: Validate indices
        if len(indices) == 0:
            raise ValueError("indices cannot be empty")

        for i, idx in enumerate(indices):
            if not isinstance(idx, int):
                raise TypeError(
                    f"Index at position {i} is not an integer: {type(idx).__name__}"
                )
            if idx < 0:
                raise ValueError(
                    f"Negative index not allowed at position {i}: {idx}"
                )

        self.indices = indices
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        gen = self.generator if self.generator else random
        shuffled = list(self.indices)
        gen.shuffle(shuffled)
        return iter(shuffled)

    def __len__(self) -> int:
        return len(self.indices)


class WeightedRandomSampler(Sampler):
    """Sample elements with given probabilities (weights).

    Useful for handling class imbalance.

    Args:
        weights: Sequence of weights (probabilities) for each sample
        num_samples: Number of samples to draw
        replacement: If True, sample with replacement
        generator: Random generator for reproducibility

    Example:
        >>> # Handle imbalanced dataset
        >>> class_counts = [1000, 100]  # Class 0 has 10x more samples
        >>> weights = [1.0/class_counts[label] for label in labels]
        >>> sampler = WeightedRandomSampler(weights, num_samples=len(weights))
    """

    weights: Sequence[float]
    num_samples: int
    replacement: bool
    generator: Optional[random.Random]

    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int,
        replacement: bool = True,
        generator: Optional[random.Random] = None
    ) -> None:
        # Security: Validate num_samples upper bound
        if num_samples > SamplerSecurityLimits.MAX_NUM_SAMPLES:
            raise ValueError(
                f"num_samples ({num_samples}) exceeds maximum "
                f"({SamplerSecurityLimits.MAX_NUM_SAMPLES})"
            )

        if not replacement and num_samples > len(weights):
            raise ValueError(
                f"num_samples ({num_samples}) cannot exceed weights length "
                f"({len(weights)}) when replacement=False"
            )

        if any(w < 0 for w in weights):
            raise ValueError("All weights must be non-negative")

        self.weights = list(weights)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        gen = self.generator if self.generator else random

        # Normalize weights
        total = sum(self.weights)
        if total == 0:
            raise ValueError("Sum of weights cannot be zero")
        probs = [w / total for w in self.weights]

        # Sample indices based on weights
        indices = list(range(len(self.weights)))

        if self.replacement:
            for _ in range(self.num_samples):
                # Weighted random choice
                r = gen.random()
                cumsum = 0.0
                for i, p in enumerate(probs):
                    cumsum += p
                    if r < cumsum:
                        yield indices[i]
                        break
                else:
                    yield indices[-1]
        else:
            # Sample without replacement using reservoir sampling with weights
            selected = []
            remaining_indices = list(indices)
            remaining_probs = list(probs)

            for _ in range(self.num_samples):
                if not remaining_indices:
                    break

                # Normalize remaining probs
                total = sum(remaining_probs)
                if total == 0:
                    break

                r = gen.random() * total
                cumsum = 0.0
                for i, p in enumerate(remaining_probs):
                    cumsum += p
                    if r < cumsum:
                        selected.append(remaining_indices[i])
                        remaining_indices.pop(i)
                        remaining_probs.pop(i)
                        break

            yield from selected

    def __len__(self) -> int:
        return self.num_samples


class BatchSampler(Sampler):
    """Wrap a sampler to yield batches of indices.

    Args:
        sampler: Base sampler
        batch_size: Size of mini-batch
        drop_last: If True, drop the last incomplete batch

    Example:
        >>> sampler = SequentialSampler(dataset)
        >>> batch_sampler = BatchSampler(sampler, batch_size=4, drop_last=False)
        >>> list(batch_sampler)
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    """

    sampler: Sampler
    batch_size: int
    drop_last: bool

    def __init__(
        self,
        sampler: Sampler,
        batch_size: int,
        drop_last: bool = False
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return math.ceil(len(self.sampler) / self.batch_size)

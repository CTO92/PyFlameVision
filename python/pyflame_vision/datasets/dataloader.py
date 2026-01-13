"""
DataLoader for PyFlameVision.

Provides efficient data loading with batching, shuffling, and optional multiprocessing.
"""

from typing import Any, Callable, Iterator, List, Optional, Union, Sized
import queue
import threading
import multiprocessing
import traceback
from contextlib import contextmanager

from .dataset import Dataset, IterableDataset
from .samplers import (
    Sampler,
    SequentialSampler,
    RandomSampler,
    BatchSampler,
)
from .collate import default_collate


# ============================================================================
# Security Limits
# ============================================================================

class DataLoaderSecurityLimits:
    """Security limits for DataLoader operations."""
    MAX_NUM_WORKERS = 64
    MAX_BATCH_SIZE = 65536
    MAX_PREFETCH_FACTOR = 100
    MAX_TIMEOUT = 3600  # 1 hour
    MAX_QUEUE_SIZE = 1000
    MAX_OUT_OF_ORDER_RESULTS = 1000  # Maximum out-of-order results to buffer


# ============================================================================
# Worker Functions
# ============================================================================

def _worker_loop(
    dataset: Dataset,
    index_queue: multiprocessing.Queue,
    output_queue: multiprocessing.Queue,
    collate_fn: Callable,
    worker_id: int,
    done_event: threading.Event
) -> None:
    """Worker process main loop for loading data.

    Args:
        dataset: Dataset to load from
        index_queue: Queue to receive batch indices from
        output_queue: Queue to send loaded batches to
        collate_fn: Function to collate samples
        worker_id: ID of this worker
        done_event: Event signaling shutdown
    """
    try:
        while not done_event.is_set():
            try:
                # Get next batch of indices
                task = index_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if task is None:
                # Shutdown signal
                break

            batch_idx, indices = task

            try:
                # Load samples
                samples = [dataset[idx] for idx in indices]
                # Collate into batch
                batch = collate_fn(samples)
                # Send result
                output_queue.put((batch_idx, batch, None))
            except Exception as e:
                # Send error
                tb = traceback.format_exc()
                output_queue.put((batch_idx, None, (e, tb)))

    except Exception as e:
        # Fatal error in worker
        tb = traceback.format_exc()
        try:
            output_queue.put((-1, None, (e, tb)))
        except Exception:
            pass


def _thread_worker_loop(
    dataset: Dataset,
    index_queue: queue.Queue,
    output_queue: queue.Queue,
    collate_fn: Callable,
    worker_id: int,
    done_event: threading.Event
) -> None:
    """Thread worker main loop for loading data.

    Args:
        dataset: Dataset to load from
        index_queue: Queue to receive batch indices from
        output_queue: Queue to send loaded batches to
        collate_fn: Function to collate samples
        worker_id: ID of this worker
        done_event: Event signaling shutdown
    """
    try:
        while not done_event.is_set():
            try:
                # Get next batch of indices
                task = index_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if task is None:
                # Shutdown signal
                break

            batch_idx, indices = task

            try:
                # Load samples
                samples = [dataset[idx] for idx in indices]
                # Collate into batch
                batch = collate_fn(samples)
                # Send result
                output_queue.put((batch_idx, batch, None))
            except Exception as e:
                # Send error
                tb = traceback.format_exc()
                output_queue.put((batch_idx, None, (e, tb)))

    except Exception as e:
        # Fatal error in worker
        tb = traceback.format_exc()
        try:
            output_queue.put((-1, None, (e, tb)))
        except Exception:
            pass


# ============================================================================
# DataLoader
# ============================================================================

class DataLoader:
    """Data loader with batching, shuffling, and multiprocessing support.

    Combines a dataset and a sampler to provide an iterable over
    the given dataset with automatic batching.

    Args:
        dataset: Dataset to load from
        batch_size: Number of samples per batch (default: 1)
        shuffle: Whether to shuffle data at each epoch (default: False)
        sampler: Custom sampler (mutually exclusive with shuffle)
        batch_sampler: Custom batch sampler (overrides batch_size, shuffle, sampler, drop_last)
        num_workers: Worker processes/threads for loading (default: 0 = main process)
        collate_fn: Function to merge samples into batch (default: default_collate)
        drop_last: Drop the last incomplete batch (default: False)
        timeout: Timeout for worker data collection in seconds (default: 0 = infinite)
        prefetch_factor: Number of batches to prefetch per worker (default: 2)
        persistent_workers: Keep workers alive between iterations (default: False)
        multiprocessing_context: Multiprocessing start method (default: None = spawn)

    Example:
        >>> dataset = ImageFolder("data/train", transform=transform)
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
        >>> for images, labels in loader:
        ...     output = model(images)

    Note:
        - When num_workers > 0, data loading uses separate processes/threads
        - shuffle and sampler are mutually exclusive
        - batch_sampler is mutually exclusive with batch_size, shuffle, sampler, drop_last
    """

    def __init__(
        self,
        dataset: Union[Dataset, IterableDataset],
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[Sampler] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        drop_last: bool = False,
        timeout: float = 0,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        multiprocessing_context: Optional[str] = None
    ) -> None:
        # Validate parameters
        self._validate_params(
            batch_size, num_workers, timeout, prefetch_factor,
            shuffle, sampler, batch_sampler
        )

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn if collate_fn is not None else default_collate
        self.drop_last = drop_last
        self.timeout = timeout if timeout > 0 else None
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.multiprocessing_context = multiprocessing_context

        # Handle iterable datasets differently
        self._is_iterable = isinstance(dataset, IterableDataset)

        # Set up sampler
        if batch_sampler is not None:
            # batch_sampler overrides everything else
            if batch_size != 1:
                raise ValueError(
                    "batch_size must be 1 when batch_sampler is specified"
                )
            if shuffle:
                raise ValueError(
                    "shuffle must be False when batch_sampler is specified"
                )
            if sampler is not None:
                raise ValueError(
                    "sampler must be None when batch_sampler is specified"
                )
            if drop_last:
                raise ValueError(
                    "drop_last must be False when batch_sampler is specified"
                )

            self.batch_sampler = batch_sampler
            self.sampler = None
        else:
            # Determine sampler
            if sampler is not None:
                if shuffle:
                    raise ValueError(
                        "sampler and shuffle are mutually exclusive"
                    )
                self.sampler = sampler
            else:
                if self._is_iterable:
                    # Iterable datasets don't use samplers
                    self.sampler = None
                elif shuffle:
                    self.sampler = RandomSampler(dataset)
                else:
                    self.sampler = SequentialSampler(dataset)

            # Create batch sampler
            if self.sampler is not None:
                self.batch_sampler = BatchSampler(
                    self.sampler, batch_size, drop_last
                )
            else:
                self.batch_sampler = None

        # Worker state
        self._workers = []
        self._worker_done_event = None
        self._index_queue = None
        self._output_queue = None

    def _validate_params(
        self,
        batch_size: int,
        num_workers: int,
        timeout: float,
        prefetch_factor: int,
        shuffle: bool,
        sampler: Optional[Sampler],
        batch_sampler: Optional[Sampler]
    ) -> None:
        """Validate DataLoader parameters."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_size > DataLoaderSecurityLimits.MAX_BATCH_SIZE:
            raise ValueError(
                f"batch_size ({batch_size}) exceeds maximum "
                f"({DataLoaderSecurityLimits.MAX_BATCH_SIZE})"
            )

        if num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {num_workers}")
        if num_workers > DataLoaderSecurityLimits.MAX_NUM_WORKERS:
            raise ValueError(
                f"num_workers ({num_workers}) exceeds maximum "
                f"({DataLoaderSecurityLimits.MAX_NUM_WORKERS})"
            )

        if timeout < 0:
            raise ValueError(f"timeout must be non-negative, got {timeout}")
        if timeout > DataLoaderSecurityLimits.MAX_TIMEOUT:
            raise ValueError(
                f"timeout ({timeout}) exceeds maximum "
                f"({DataLoaderSecurityLimits.MAX_TIMEOUT})"
            )

        if prefetch_factor <= 0:
            raise ValueError(f"prefetch_factor must be positive, got {prefetch_factor}")
        if prefetch_factor > DataLoaderSecurityLimits.MAX_PREFETCH_FACTOR:
            raise ValueError(
                f"prefetch_factor ({prefetch_factor}) exceeds maximum "
                f"({DataLoaderSecurityLimits.MAX_PREFETCH_FACTOR})"
            )

    def __iter__(self) -> Iterator:
        """Return iterator over batches."""
        if self._is_iterable:
            return self._iter_iterable_dataset()
        elif self.num_workers == 0:
            return self._single_process_iter()
        else:
            return self._multi_process_iter()

    def __len__(self) -> int:
        """Return number of batches.

        Only available for map-style datasets with batch_sampler.
        """
        if self._is_iterable:
            raise TypeError(
                "len() not supported for IterableDataset. "
                "Use a map-style Dataset for known length."
            )
        if self.batch_sampler is not None:
            return len(self.batch_sampler)
        else:
            # Shouldn't happen, but fallback
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __enter__(self) -> "DataLoader":
        """Context manager entry. Returns self for use in 'with' statement.

        Example:
            >>> with DataLoader(dataset, num_workers=4) as loader:
            ...     for batch in loader:
            ...         process(batch)
            ... # Workers are automatically cleaned up
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit. Ensures worker cleanup on iteration abandonment."""
        self.shutdown()
        return None  # Don't suppress exceptions

    def shutdown(self) -> None:
        """Explicitly shut down worker threads.

        Call this method when you're done iterating or if you abandon iteration early.
        This is automatically called when using the DataLoader as a context manager.

        Example:
            >>> loader = DataLoader(dataset, num_workers=4)
            >>> for i, batch in enumerate(loader):
            ...     if should_stop:
            ...         break  # Iteration abandoned
            >>> loader.shutdown()  # Clean up workers
        """
        # Signal workers to stop if event exists
        if self._worker_done_event is not None:
            self._worker_done_event.set()

        # Clear any pending items from queues
        if self._index_queue is not None:
            try:
                while True:
                    self._index_queue.get_nowait()
            except queue.Empty:
                pass

        if self._output_queue is not None:
            try:
                while True:
                    self._output_queue.get_nowait()
            except queue.Empty:
                pass

        # Wait for workers to finish
        for w in self._workers:
            if w.is_alive():
                w.join(timeout=1.0)

        # Reset state
        self._workers = []
        self._worker_done_event = None
        self._index_queue = None
        self._output_queue = None

    def _single_process_iter(self) -> Iterator:
        """Single-process iteration (no workers)."""
        for batch_indices in self.batch_sampler:
            # Load samples
            samples = [self.dataset[idx] for idx in batch_indices]
            # Collate and yield
            yield self.collate_fn(samples)

    def _iter_iterable_dataset(self) -> Iterator:
        """Iterate over an iterable dataset."""
        batch = []
        for sample in self.dataset:
            batch.append(sample)
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []

        # Handle last incomplete batch
        if len(batch) > 0 and not self.drop_last:
            yield self.collate_fn(batch)

    def _multi_process_iter(self) -> Iterator:
        """Multi-process iteration using thread workers.

        Uses threads instead of processes for Windows compatibility
        and simpler implementation.
        """
        # Create queues and store in instance for cleanup
        self._index_queue = queue.Queue(
            maxsize=self.num_workers * self.prefetch_factor
        )
        self._output_queue = queue.Queue(
            maxsize=self.num_workers * self.prefetch_factor
        )

        # Done event
        self._worker_done_event = threading.Event()

        # Start workers and store in instance for cleanup
        self._workers = []
        for worker_id in range(self.num_workers):
            w = threading.Thread(
                target=_thread_worker_loop,
                args=(
                    self.dataset,
                    self._index_queue,
                    self._output_queue,
                    self.collate_fn,
                    worker_id,
                    self._worker_done_event
                ),
                daemon=True
            )
            w.start()
            self._workers.append(w)

        try:
            # Submit batch indices to workers
            results = {}
            next_batch_idx = 0
            batches_submitted = 0
            batches_received = 0
            total_batches = len(self.batch_sampler)

            # Iterator over batch indices
            batch_iter = iter(self.batch_sampler)

            # Pre-fill the queue
            for _ in range(min(self.num_workers * self.prefetch_factor, total_batches)):
                try:
                    indices = next(batch_iter)
                    self._index_queue.put((batches_submitted, indices))
                    batches_submitted += 1
                except StopIteration:
                    break

            # Yield batches in order
            while next_batch_idx < total_batches:
                # If we already have the next batch, yield it
                if next_batch_idx in results:
                    batch, error = results.pop(next_batch_idx)
                    if error is not None:
                        exc, tb = error
                        raise RuntimeError(
                            f"Worker error:\n{tb}"
                        ) from exc
                    next_batch_idx += 1

                    # Submit more work
                    if batches_submitted < total_batches:
                        try:
                            indices = next(batch_iter)
                            self._index_queue.put((batches_submitted, indices))
                            batches_submitted += 1
                        except StopIteration:
                            pass

                    yield batch
                else:
                    # Wait for next result
                    try:
                        batch_idx, batch, error = self._output_queue.get(
                            timeout=self.timeout
                        )
                    except queue.Empty:
                        raise RuntimeError(
                            f"DataLoader timed out after {self.timeout} seconds"
                        )

                    if batch_idx == -1:
                        # Fatal worker error
                        exc, tb = error
                        raise RuntimeError(f"Worker error:\n{tb}") from exc

                    if batch_idx == next_batch_idx:
                        # Got the one we need
                        if error is not None:
                            exc, tb = error
                            raise RuntimeError(
                                f"Worker error:\n{tb}"
                            ) from exc
                        next_batch_idx += 1

                        # Submit more work
                        if batches_submitted < total_batches:
                            try:
                                indices = next(batch_iter)
                                self._index_queue.put((batches_submitted, indices))
                                batches_submitted += 1
                            except StopIteration:
                                pass

                        yield batch
                    else:
                        # Store for later
                        results[batch_idx] = (batch, error)

                        # Security: Check for excessive out-of-order results
                        if len(results) > DataLoaderSecurityLimits.MAX_OUT_OF_ORDER_RESULTS:
                            raise RuntimeError(
                                f"Too many out-of-order results ({len(results)}). "
                                f"Maximum allowed: {DataLoaderSecurityLimits.MAX_OUT_OF_ORDER_RESULTS}. "
                                "This may indicate a worker deadlock or data loading issue."
                            )

        finally:
            # Signal workers to stop
            self._worker_done_event.set()
            for _ in self._workers:
                try:
                    self._index_queue.put(None, timeout=0.1)
                except queue.Full:
                    pass

            # Wait for workers to finish
            for w in self._workers:
                w.join(timeout=1.0)


class DataLoaderIterator:
    """Iterator for DataLoader that supports resuming and checkpointing."""

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self._batch_idx = 0
        self._iter = iter(loader)

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self._iter)
        self._batch_idx += 1
        return batch

    @property
    def batch_idx(self) -> int:
        """Current batch index (0-based)."""
        return self._batch_idx

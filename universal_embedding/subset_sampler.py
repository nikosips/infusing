import numpy as np
from typing import Optional
from grain._src.core import sharding
from grain._src.python import record
from grain._src.python.dataset import dataset



class SubsetSampler:
    """Sampler for subsampling a random subset of indices."""

    def __init__(
        self,
        num_records: int,
        fraction: float,
        shard_options: sharding.ShardOptions,
        shuffle: bool = True,
        num_epochs: Optional[int] = None,
        seed: Optional[int] = None,
        selected_indices: Optional[np.ndarray] = None,
    ):
        if num_records <= 0:
            raise ValueError("num_records must be greater than 0.")
        if not (0 < fraction <= 1.0):
            raise ValueError("fraction must be between 0 and 1.")
        if shuffle and seed is None:
            raise ValueError("Shuffling requires specifying a seed.")
        if seed is not None and (seed < 0 or seed.bit_length() > 32):
            raise ValueError("Seed should be positive 32-bit integer.")

        self._num_records_total = num_records
        self._fraction = fraction
        self._shard_options = shard_options
        self._shuffle = shuffle
        self._num_epochs = num_epochs
        self._seed = seed

        self._subset_size = int(np.ceil(num_records * fraction))
    
        rng = np.random.default_rng(seed)
        
        if self._subset_size <= 0:
            raise ValueError("fraction too small, subset size is zero.")
        if selected_indices is not None:
            if len(selected_indices) != self._subset_size:
                raise ValueError(
                    f"selected_indices must have length {self._subset_size}."
                )
            self._subset_indices = selected_indices
        else:
            # Generate random subset of indices upfront
            self._subset_indices = rng.choice(
                num_records, size=self._subset_size, replace=False
            )
        if shuffle:
            rng.shuffle(self._subset_indices)

        # Apply sharding
        start, end = sharding.even_split(self._subset_size, shard_options)
        self._subset_indices_sharded = self._subset_indices[start:end]

        self._effective_size = len(self._subset_indices_sharded)
        if num_epochs is None:
            self._max_index = None
        else:
            self._max_index = self._effective_size * num_epochs

    def __repr__(self) -> str:
        return (
            f"SubsetSampler(num_records={self._num_records_total}, "
            f"fraction={self._fraction}, shard_options={self._shard_options!r}, "
            f"shuffle={self._shuffle}, num_epochs={self._num_epochs}, "
            f"seed={self._seed})"
        )

    def __getitem__(self, index: int) -> record.RecordMetadata:
        if index < 0 or (self._max_index is not None and index >= self._max_index):
            raise IndexError(
                f"RecordMetadata index out of bounds; Got {index},"
                f" allowed indices are [0, {self._max_index})."
            )

        subset_index = index % self._effective_size
        epoch = index // self._effective_size

        # Compute record key (original dataset index)
        record_key = int(self._subset_indices_sharded[subset_index])

        # Initialize RNG for the current index, consistent with Grain design
        rng = None
        if self._seed is not None:
            rng = np.random.Generator(np.random.Philox(key=self._seed + index))

        next_record = record.RecordMetadata(
            index=index, record_key=record_key, rng=rng
        )
        return next_record

"""Batch composition, so the region term is never inert.

Tversky is computed only over images that have a non-empty mask. A batch
containing no defective image therefore contributes nothing through it. Only
18.6% of the training split is defective, so at batch 8 an unconstrained
sampler leaves ``0.814 ** 8 = 19%`` of batches with no region signal at all.

This sampler places a fixed number of defective images in every batch. That is
a deliberate, small distortion of the class prior -- at batch 8 it gives 25%
defective against a natural 18.6% -- and :meth:`StratifiedBatchSampler.realised_rate`
reports it so the deviation appears in the run log rather than hiding. It
matters because module 03 calibrates against the model's confidences, and the
prior the model trained under is part of what shapes them.
"""

from __future__ import annotations

import math
import random
from typing import Iterator, List, Sequence

from torch.utils.data import Sampler


class StratifiedBatchSampler(Sampler):
    """Yield index batches holding a fixed number of defective images.

    Clean images are drawn without replacement and define the epoch length, so
    every clean image is seen once per epoch. Defective images are recycled
    within an epoch, since there are far fewer of them; they are reshuffled each
    time the pool is exhausted rather than repeating in a fixed order.
    """

    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int = 8,
        defective_per_batch: int = 0,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if batch_size < 2:
            raise ValueError(f"batch_size must be at least 2, got {batch_size}")

        self.defective = [i for i, label in enumerate(labels) if label == 1]
        self.clean = [i for i, label in enumerate(labels) if label != 1]
        if not self.defective:
            raise ValueError(
                "no defective images in this split; the region term would never "
                "fire and the model has nothing to segment"
            )
        if not self.clean:
            raise ValueError("no clean images in this split")

        natural_rate = len(self.defective) / len(labels)
        if defective_per_batch <= 0:
            # Round up so the count is never zero and defect exposure is never
            # below the natural rate.
            defective_per_batch = max(1, math.ceil(batch_size * natural_rate))
        if defective_per_batch >= batch_size:
            raise ValueError(
                f"defective_per_batch ({defective_per_batch}) must leave room "
                f"for clean images in a batch of {batch_size}"
            )

        self.batch_size = batch_size
        self.defective_per_batch = defective_per_batch
        self.clean_per_batch = batch_size - defective_per_batch
        self.natural_rate = natural_rate
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    @property
    def realised_rate(self) -> float:
        """Fraction of each batch that is defective, for the run log."""
        return self.defective_per_batch / self.batch_size

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle deterministically per epoch, so a run is reproducible."""
        self.epoch = epoch

    def __len__(self) -> int:
        whole = len(self.clean) // self.clean_per_batch
        if self.drop_last or len(self.clean) % self.clean_per_batch == 0:
            return whole
        return whole + 1

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random((self.seed, self.epoch).__hash__())
        clean = list(self.clean)
        rng.shuffle(clean)

        defective_pool: List[int] = []

        def take_defective(count: int) -> List[int]:
            nonlocal defective_pool
            drawn: List[int] = []
            while len(drawn) < count:
                if not defective_pool:
                    defective_pool = list(self.defective)
                    rng.shuffle(defective_pool)
                drawn.append(defective_pool.pop())
            return drawn

        for start in range(0, len(clean), self.clean_per_batch):
            chunk = clean[start : start + self.clean_per_batch]
            if len(chunk) < self.clean_per_batch and self.drop_last:
                return
            batch = chunk + take_defective(self.defective_per_batch)
            rng.shuffle(batch)
            yield batch

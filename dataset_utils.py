from typing import List, Optional

import math
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class SupervisedContrastiveDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class LabelAwareBatchSampler(Sampler[List[int]]):
    """Ensure every batch provides multiple samples per label for contrastive learning."""

    def __init__(self,
                 labels: np.ndarray,
                 batch_size: int,
                 num_views: int = 2,
                 steps_per_epoch: Optional[int] = None,
                 seed: Optional[int] = None):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if num_views < 1:
            raise ValueError("num_views must be positive")

        self.labels = np.asarray(labels, dtype=np.int64)
        if self.labels.size == 0:
            raise ValueError("labels array must be non-empty")

        self.batch_size = batch_size
        self.num_views = num_views
        self.unique_labels = np.unique(self.labels)
        self.label_to_indices = {
            label: np.flatnonzero(self.labels == label)
            for label in self.unique_labels
        }
        self.steps_per_epoch = steps_per_epoch or max(1, math.ceil(len(self.labels) / self.batch_size))
        base_seed = seed if seed is not None else np.random.SeedSequence().entropy
        self._base_seed = int(base_seed) % (2 ** 32)
        self._epoch = 0

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(self._base_seed + self._epoch)
        self._epoch += 1

        pairs_per_batch = max(1, self.batch_size // self.num_views)
        for _ in range(self.steps_per_epoch):
            batch_indices: List[int] = []

            chosen_labels = rng.choice(self.unique_labels, size=pairs_per_batch, replace=True)
            for label in chosen_labels:
                indices = self.label_to_indices[label]
                replace = len(indices) < self.num_views
                sampled = rng.choice(indices, size=self.num_views, replace=replace)
                batch_indices.extend(int(idx) for idx in sampled.tolist())

            remainder = self.batch_size - len(batch_indices)
            if remainder > 0:
                label = rng.choice(self.unique_labels)
                indices = self.label_to_indices[label]
                replace = len(indices) < remainder
                sampled = rng.choice(indices, size=remainder, replace=replace)
                batch_indices.extend(int(idx) for idx in sampled.tolist())

            yield batch_indices

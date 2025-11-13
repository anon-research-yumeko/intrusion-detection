"""chipoccon package

Split modules from original `main.py` for clarity and reuse.
"""

from .datasets import DataBundle, resolve_data_dir, load_and_sample_data
from .models import OCConNet
from .losses import OCConLossFn
from .dataset_utils import SupervisedContrastiveDataset, LabelAwareBatchSampler
from .chip import Chip
from .trainer import train_occon_and_embed
from .scorers import spherical_kmeans_scores, chip_pvalue_scores
from .utils import set_seed, to_pvalue_scores, compute_metrics, make_occon_grid

__all__ = [
    "DataBundle",
    "load_and_sample_data",
    "resolve_data_dir",
    "OCConNet",
    "OCConLossFn",
    "SupervisedContrastiveDataset",
    "LabelAwareBatchSampler",
    "Chip",
    "train_occon_and_embed",
    "spherical_kmeans_scores",
    "chip_pvalue_scores",
    "set_seed",
    "to_pvalue_scores",
    "compute_metrics",
    "make_occon_grid",
]

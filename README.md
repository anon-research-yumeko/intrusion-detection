# chipoccon

This package contains a modular split of the `main.py` from the repository. It separates dataset loaders, model, loss, training and scoring utilities into independent modules so they can be published to GitHub and reused as a package.

Modules
- `chipoccon.datasets` - dataset loaders and `DataBundle` dataclass
- `chipoccon.dataset_utils` - dataset and sampler helpers for supervised contrastive training
- `chipoccon.models` - `OCConNet` architecture
- `chipoccon.losses` - `OCConLossFn` implementation
- `chipoccon.chip` - `Chip` class (chi-square based) implementation
- `chipoccon.scorers` - SKMeans and Chip scorers
- `chipoccon.trainer` - `train_occon_and_embed` trainer
- `chipoccon.utils` - helper utilities and metric computations
- `chipoccon.evaluate` - high-level grid evaluation (wrapping the original `evaluate_grid`)

Usage

You can import the package from the repository root (same level as `main.py`):

```python
from chipoccon import train_occon_and_embed, evaluate_grid
```

Dependencies
See `requirements.txt` for a minimal set of packages required to run training/evaluation scripts.

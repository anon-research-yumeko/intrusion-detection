# chipoccon
Modules
- `chipoccon.datasets` - dataset loaders and `DataBundle` dataclass
- `chipoccon.dataset_utils` - dataset and sampler helpers for contrastive training
- `chipoccon.models` - `OCConNet` architecture
- `chipoccon.losses` - `OCConLossFn` implementation
- `chipoccon.chip` - `Chip` class (chi-square based) implementation
- `chipoccon.scorers` - SKMeans and Chip scorers
- `chipoccon.trainer` - `train_occon_and_embed` trainer
- `chipoccon.utils` - helper utilities and metric computations
- `chipoccon.evaluate` - high-level grid evaluation

Usage

You can import the package from the repository root (same level as `main.py`):

```python
py main.py --dataset unsw --cut_off_list 0.3 --epochs 30 --enable_chip
```

Dependencies
See `requirements.txt` for a minimal set of packages required to run training/evaluation scripts.

Link to Datset
NF-CSE-CIC-IDS2018-v3: https://staff.itee.uq.edu.au/marius/NIDS_datasets/


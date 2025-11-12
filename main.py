import os
import math

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, Sampler

from numba import jit

# -------------------------
# Dataset + Model utilities
# -------------------------

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


class OCConNet(nn.Module):
    def __init__(self, input_dim, emb_dim, proj_dim=256):
        super().__init__()

        def _round_to_mult(value: float, mult: int = 8) -> int:
            return int(max(mult, mult * round(value / mult)))

        l1_est = (input_dim ** (2 / 3)) * (emb_dim ** (1 / 3))
        l2_est = (input_dim ** (1 / 3)) * (emb_dim ** (2 / 3))
        enc_l1 = _round_to_mult(l1_est)
        enc_l2 = _round_to_mult(l2_est)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, enc_l1),
            nn.BatchNorm1d(enc_l1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(enc_l1, enc_l2),
            nn.BatchNorm1d(enc_l2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(enc_l2, emb_dim),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, proj_dim),
        )

    def forward(self, x):
        emb = self.encoder(x)
        emb = F.normalize(emb, p=2, dim=1, eps=1e-12)
        proj = self.projection_head(emb)
        proj = F.normalize(proj, p=2, dim=1, eps=1e-12)
        return emb, proj


class OCConLossFn(nn.Module):
    def __init__(self, temperature: float = 0.07, device: str = "cpu"):
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(self.device)

        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=self.device)
        mask = mask * logits_mask

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        denom = torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        log_prob = logits - denom

        sum_mask = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / torch.clamp(sum_mask, min=1.0)
        if sum_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return -mean_log_prob_pos[sum_mask > 0].mean()


# -------------------------
# Chip Algorithm
# -------------------------


class Chip:
    def __init__(self):
        self.nameAlg = 'Chip'
        self.ts = 1
        self.current = {}
        self.total = {}
        self.time_table = {}
        self.time_cache = {}
        self.last_ts = None
        self.last_edge_cache = {}

    @staticmethod
    @jit(nopython=True)
    def chi_squared_test(a, s, t):
        mask = (s != 0) & (t != 1)
        result = np.zeros_like(a, dtype=np.float64)
        result[mask] = ((a[mask] - s[mask] / t[mask]) * t[mask]) ** 2 / (s[mask] * (t[mask] - 1))
        return result

    def chip_no_collision(self, data_src, data_dst, ts):
        data_src = data_src.to_numpy() if hasattr(data_src, 'to_numpy') else np.array(data_src)
        data_dst = data_dst.to_numpy() if hasattr(data_dst, 'to_numpy') else np.array(data_dst)
        ts = ts.to_numpy() if hasattr(ts, 'to_numpy') else np.array(ts)

        batch_size = len(data_src)
        scores = np.zeros(batch_size)

        edge_cache = self.last_edge_cache if ts[0] == self.last_ts else {}
        current_ts = ts[0]
        seg_start = 0  # start index of the current timestamp segment

        for i in range(batch_size):
            if ts[i] != current_ts:
                # finalize the just-finished segment [seg_start, i)
                self._update_scores(scores, edge_cache, data_src, data_dst, ts, current_ts, seg_start, i)
                edge_cache = {}
                current_ts = ts[i]
                seg_start = i  # new segment starts here

            if self.ts < ts[i]:
                self.current.clear()
                self.time_cache.clear()
                self.ts = ts[i]

            edge = (data_src[i], data_dst[i])

            if edge not in self.time_cache:
                self.time_cache[edge] = True
                self.time_table[edge] = self.time_table.get(edge, 0) + 1

            self.current[edge] = self.current.get(edge, 0) + 1
            self.total[edge] = self.total.get(edge, 0) + 1

            current_count = self.current[edge]
            total_count = self.total[edge]
            time_count = self.time_table[edge]

            scores[i] = self.chi_squared_test(
                np.array([current_count]),
                np.array([total_count]),
                np.array([time_count])
            )[0]
            edge_cache[edge] = scores[i]

        # finalize the last open segment [seg_start, batch_size)
        self._update_scores(scores, edge_cache, data_src, data_dst, ts, current_ts, seg_start, batch_size)

        self.last_ts = ts[-1]
        self.last_edge_cache = edge_cache if ts[-1] == current_ts else {}

        return scores

    def _update_scores(self, scores, edge_cache, data_src, data_dst, ts, current_ts, start, end):
        for i in range(start, end):
            if ts[i] == current_ts:
                edge = (data_src[i], data_dst[i])
                scores[i] = edge_cache.get(edge, scores[i])

# -------------------------
# Helpers
# -------------------------

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_pvalue_scores(test_scores: np.ndarray, ref_scores: np.ndarray) -> np.ndarray:
    sorted_ref = np.sort(ref_scores)
    n_ref = len(sorted_ref)
    idx = np.searchsorted(sorted_ref, test_scores, side='left')
    # count >= test_score
    counts_ge = n_ref - idx
    # (cnt+1)/(n+1) for stability
    p_values = (counts_ge + 1.0) / (n_ref + 1.0)
    return -np.log(p_values)


def compute_metrics(scores: np.ndarray, y_true: np.ndarray, f1_percentile: float) -> Dict[str, float]:
    """
    Compute AUROC/AUPR and point metrics at the same percentile threshold used for F1.
    - Threshold = np.percentile(scores, f1_percentile)
    - ACC/Precision/Recall/F1 are all computed at this single threshold.

    Parameters
    ----------
    scores : np.ndarray
        Anomaly scores (higher = more anomalous).
    y_true : np.ndarray
        Ground-truth binary labels (0=benign, 1=attack).
    f1_percentile : float
        Percentile in [0, 100] used to set the prediction threshold.

    Returns
    -------
    Dict[str, float]
        {"AUROC", "AUPR", "ACC", "Precision", "Recall", f"F1@{f1_percentile:.2f}"}
    """
    # Ranking metrics (threshold-free)
    auroc = roc_auc_score(y_true, scores)
    aupr = average_precision_score(y_true, scores)

    # Single threshold used for all point metrics
    thr = np.percentile(scores, f1_percentile)
    y_pred = (scores >= thr).astype(int)

    # Confusion matrix components with fixed label order to ensure 2x2 output
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Point metrics with safe divisions
    total = max(1, (tp + tn + fp + fn))
    acc = (tp + tn) / total
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = f1_score(y_true, y_pred)

    return {
        "AUROC": float(auroc),
        "AUPR": float(aupr),
        "ACC": float(acc),
        "Precision": float(precision),
        "Recall": float(recall),
        f"F1@{f1_percentile:.2f}": float(f1),
    }


# -------------------------
# Data loading / preprocessing
# -------------------------

@dataclass
class DataBundle:
    benign_df: pd.DataFrame
    merged_df: pd.DataFrame
    columns_numeric: List[str]
    scaler: MinMaxScaler
    label_encoder: LabelEncoder
    X_benign_scaled: np.ndarray
    y_benign_edges: np.ndarray
    input_dim: int
    f1_percentile: float 




SUPPORTED_DATASETS = ("unsw", "cicids2017", "cicids2018")

DATASET_DEFAULT_DIRS = {
    "unsw": os.path.join("data", "NF-UNSWNB15-v3"),
    "cicids2017": os.path.join("data", "CICIDS2017"),
    "cicids2018": os.path.join("data", "NF-2018-v3"),
}


def _load_unsw_data(data_dir: str, cut_off: float, seed: int) -> DataBundle:
    merged_path = os.path.join(data_dir, "merged_df.parquet")
    merged_df = pd.read_parquet(merged_path)
    if "Edge" not in merged_df.columns:
        merged_df["Edge"] = merged_df["Source IP"] + "->" + merged_df["Destination IP"]
    time_split = 144

    data1 = merged_df[merged_df["Timestamp"] <= time_split]
    benign_df = data1.groupby("Edge").filter(lambda df: df["Label"].max() == 0).copy()
    max_ts = benign_df["Timestamp"].max()
    cutoff_ts = cut_off * max_ts
    early_df = benign_df[benign_df["Timestamp"] <= cutoff_ts]


    ts_per_edge = early_df.groupby("Edge")["Timestamp"].nunique()
    eligible_edges = ts_per_edge[ts_per_edge >= 5].index
    benign_df = early_df[early_df["Edge"].isin(eligible_edges)].copy()

    columns_text = [
        "Source IP",
        "Destination IP",
        "Source Port",
        "Destination Port",
        "Timestamp",
        "FLOW_END_MILLISECONDS",

        "DNS_QUERY_ID", # Noise
        "FTP_COMMAND_RET_CODE", # Only meaningful for FTP traffic

        "Label",
        "Attack",
        "Edge"]

    columns_cat = [
        "PROTOCOL",
        "L7_PROTO",
        "TCP_FLAGS",
        "ICMP_TYPE",
        "CLIENT_TCP_FLAGS",
        "SERVER_TCP_FLAGS",
        "DNS_QUERY_TYPE",
        "ICMP_IPV4_TYPE",
    ]

    columns_numeric = [col for col in merged_df.columns if col not in columns_text]

    X_benign_numeric = benign_df[columns_numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scaler = MinMaxScaler()
    X_benign_scaled = scaler.fit_transform(X_benign_numeric).astype(np.float32)
    edge_labels = benign_df["Edge"].copy()
    label_encoder = LabelEncoder()
    y_benign_edges = label_encoder.fit_transform(edge_labels).astype(np.int64)
    input_dim = X_benign_scaled.shape[1]

    benign_ratio = (merged_df["Label"] == 0).mean() * 100.0
    f1_percentile = benign_ratio

    return DataBundle(
        benign_df=benign_df,
        merged_df=merged_df,
        columns_numeric=columns_numeric,
        scaler=scaler,
        label_encoder=label_encoder,
        X_benign_scaled=X_benign_scaled,
        y_benign_edges=y_benign_edges,
        input_dim=input_dim,
        f1_percentile=f1_percentile,
    )


def _load_cicids2017_data(data_dir: str, cut_off: float, seed: int) -> DataBundle:
    data1_path = os.path.join(data_dir, "Tuesday.parquet")
    merged_path = os.path.join(data_dir, "merged_df.parquet")
    data1 = pd.read_parquet(data1_path)
    merged_df = pd.read_parquet(merged_path)

    columns_text = [
        "Flow ID", " Source IP", " Source Port", " Destination IP", " Destination Port",
        " Protocol", " Timestamp", " Label", "Edge"
    ]
    columns_categorical = [col for col in merged_df if "Flag" in col]
    columns_numeric = [col for col in merged_df if col not in columns_text + columns_categorical]

    if "Edge" not in merged_df.columns:
        merged_df["Edge"] = merged_df[" Source IP"] + "->" + merged_df[" Destination IP"]

    benign_df = data1.groupby("Edge").filter(lambda df: df[" Label"].max() == 0).copy()

    max_ts = benign_df[" Timestamp"].max()
    cutoff_ts = cut_off * max_ts
    early_df = benign_df[benign_df[" Timestamp"] <= cutoff_ts]
    ts_per_edge = early_df.groupby("Edge")[" Timestamp"].nunique()
    eligible_edges = ts_per_edge[ts_per_edge >= 5].index
    if len(eligible_edges) == 0:
        raise ValueError("No eligible edges found for OCCon training (≥5 timestamps).")

    benign_df = early_df[early_df["Edge"].isin(eligible_edges)].copy()

    X_benign_numeric = benign_df[columns_numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scaler = MinMaxScaler()
    X_benign_scaled = scaler.fit_transform(X_benign_numeric).astype(np.float32)
    edge_labels = benign_df["Edge"].copy()
    label_encoder = LabelEncoder()
    y_benign_edges = label_encoder.fit_transform(edge_labels).astype(np.int64)
    input_dim = X_benign_scaled.shape[1]

    benign_ratio = (merged_df[" Label"] == 0).mean() * 100.0
    f1_percentile = benign_ratio

    return DataBundle(
        benign_df=benign_df,
        merged_df=merged_df,
        columns_numeric=columns_numeric,
        scaler=scaler,
        label_encoder=label_encoder,
        X_benign_scaled=X_benign_scaled,
        y_benign_edges=y_benign_edges,
        input_dim=input_dim,
        f1_percentile=f1_percentile,
    )


def _load_cicids2018_data(data_dir: str, cut_off: float, seed: int) -> DataBundle:
    merged_path = os.path.join(data_dir, "merged_df.parquet")
    merged_df = pd.read_parquet(merged_path, engine="fastparquet")

    columns_text = [
        " Source IP", " Destination IP", " Source Port", " Destination Port",
        "DNS_QUERY_ID", "DNS_QUERY_TYPE", "FTP_COMMAND_RET_CODE", "PROTOCOL",
        " Timestamp", "FLOW_END_MILLISECONDS", " Label", "Attack", "Edge",
    ]
    columns_numeric = [col for col in merged_df if col not in columns_text]

    merged_df["Edge"] = merged_df[" Source IP"] + "->" + merged_df[" Destination IP"]

    time_split = 144
    data1 = merged_df[merged_df[" Timestamp"] <= time_split]

    benign_df = data1.groupby("Edge").filter(lambda df: df[" Label"].max() == 0).copy()

    max_ts = benign_df[" Timestamp"].max()
    cutoff_ts = cut_off * max_ts
    early_df = benign_df[benign_df[" Timestamp"] <= cutoff_ts]
    ts_per_edge = early_df.groupby("Edge")[" Timestamp"].nunique()
    eligible_edges = ts_per_edge[ts_per_edge >= 5].index
    benign_df = early_df[early_df["Edge"].isin(eligible_edges)].copy()

    X_benign_numeric = benign_df[columns_numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scaler = MinMaxScaler()
    X_benign_scaled = scaler.fit_transform(X_benign_numeric).astype(np.float32)
    edge_labels = benign_df["Edge"].copy()
    label_encoder = LabelEncoder()
    y_benign_edges = label_encoder.fit_transform(edge_labels).astype(np.int64)
    input_dim = X_benign_scaled.shape[1]

    benign_ratio = (merged_df[" Label"] == 0).mean() * 100.0
    f1_percentile = benign_ratio

    return DataBundle(
        benign_df=benign_df,
        merged_df=merged_df,
        columns_numeric=columns_numeric,
        scaler=scaler,
        label_encoder=label_encoder,
        X_benign_scaled=X_benign_scaled,
        y_benign_edges=y_benign_edges,
        input_dim=input_dim,
        f1_percentile=f1_percentile,
    )


_DATASET_LOADERS = {
    "unsw": _load_unsw_data,
    "cicids2017": _load_cicids2017_data,
    "cicids2018": _load_cicids2018_data,
}


def resolve_data_dir(dataset: str, user_data_dir: str) -> str:
    dataset_key = dataset.lower()
    if dataset_key not in DATASET_DEFAULT_DIRS:
        raise ValueError(f"Unsupported dataset '{dataset}'. Choose from: {', '.join(SUPPORTED_DATASETS)}")

    if user_data_dir and user_data_dir.lower() != "auto":
        return user_data_dir

    return DATASET_DEFAULT_DIRS[dataset_key]


def load_and_sample_data(dataset: str,
                         data_dir: str,
                         cut_off: float,
                         seed: int) -> DataBundle:
    dataset_key = dataset.lower()
    if dataset_key not in _DATASET_LOADERS:
        raise ValueError(f"Unsupported dataset '{dataset}'. Choose from: {', '.join(SUPPORTED_DATASETS)}")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found for dataset '{dataset_key}': {data_dir}")

    # Keep signature to match existing call sites while delegating to dataset-specific loader.
    return _DATASET_LOADERS[dataset_key](data_dir, cut_off, seed)


# -------------------------
# OCCon training + embedding
# -------------------------

@dataclass
class OCConConfig:
    batch_size: int = 128
    embedding_dim: int = 32
    projection_dim: int = 256
    temperature_occon: float = 0.5
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 200
    early_stopping_patience: int = 10
    min_epochs: int = 80
    num_workers: int = 4



def train_occon_and_embed(db: DataBundle,
                          cfg: OCConConfig,
                           seed: int,
                           device: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    X = db.X_benign_scaled
    y = db.y_benign_edges
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=seed
    )

    train_ds = SupervisedContrastiveDataset(X_tr, y_tr)
    val_ds = SupervisedContrastiveDataset(X_val, y_val)

    eff_bs = min(cfg.batch_size, len(train_ds))
    eff_bs = max(2, eff_bs)

    train_labels_np = train_ds.labels.cpu().numpy()
    steps_per_epoch = max(1, math.ceil(len(train_ds) / eff_bs))
    train_sampler = LabelAwareBatchSampler(
        labels=train_labels_np,
        batch_size=eff_bs,
        num_views=2,
        steps_per_epoch=steps_per_epoch,
        seed=seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eff_bs,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.num_workers > 0,
        drop_last=False,
    )

    model = OCConNet(db.input_dim, cfg.embedding_dim, cfg.projection_dim).to(device)
    criterion = OCConLossFn(temperature=cfg.temperature_occon, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_state = None
    epochs_since_improve = 0

    for epoch in range(cfg.epochs):
        model.train()
        for feats, labels in train_loader:
            if feats.size(0) <= 1:
                continue
            feats = feats.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            _, proj = model(feats)
            loss = criterion(proj, labels)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss_accum = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                if feats.size(0) <= 1:
                    continue
                feats = feats.to(device)
                labels = labels.to(device)
                _, proj = model(feats)
                vloss = criterion(proj, labels)
                if torch.isnan(vloss) or torch.isinf(vloss):
                    continue
                val_loss_accum += float(vloss.item())
                n_val_batches += 1

        mean_val_loss = (val_loss_accum / max(1, n_val_batches)) if n_val_batches > 0 else float("inf")
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if (epoch + 1) >= cfg.min_epochs and epochs_since_improve >= cfg.early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        benign_tensor = torch.tensor(db.X_benign_scaled, dtype=torch.float32, device=device)
        emb_benign, _ = model(benign_tensor)
        emb_benign = emb_benign.cpu().numpy()

        X_full = db.merged_df[db.columns_numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        X_full = db.scaler.transform(X_full).astype(np.float32)
        batch_size = 8192
        chunks = []
        for start in range(0, len(X_full), batch_size):
            chunk = torch.tensor(X_full[start:start + batch_size], dtype=torch.float32, device=device)
            emb, _ = model(chunk)
            chunks.append(emb.cpu().numpy())
        emb_full = np.vstack(chunks)

    return emb_benign, emb_full



# -------------------------
# Scorers: SKMeans, CHIP
# -------------------------

def spherical_kmeans_scores(emb_benign: np.ndarray,
                            emb_full: np.ndarray,
                            k: int,
                            seed: int) -> Tuple[np.ndarray, np.ndarray]:
    benign_norm = normalize(emb_benign, norm="l2")
    full_norm = normalize(emb_full, norm="l2")

    km = KMeans(
        n_clusters=k,
        n_init=20,
        random_state=seed,
        tol=1e-4,
    ).fit(benign_norm)
    centers = normalize(km.cluster_centers_, norm="l2")

    d_full = pairwise_distances(full_norm, centers, metric="cosine").min(axis=1)
    d_benign = pairwise_distances(benign_norm, centers, metric="cosine").min(axis=1)

    pv_scores = to_pvalue_scores(d_full, d_benign)
    return pv_scores, d_full


def chip_pvalue_scores(benign_df: pd.DataFrame,
                       merged_df: pd.DataFrame) -> np.ndarray:
    ref = Chip()
    scores_ref = np.zeros(len(benign_df))
    batch = 10000
    for start in range(0, len(benign_df), batch):
        end = min(start + batch, len(benign_df))
        scores_ref[start:end] = ref.chip_no_collision(
            benign_df["Source IP"].iloc[start:end],
            benign_df["Destination IP"].iloc[start:end],
            benign_df["Timestamp"].iloc[start:end],
        )

    chip = Chip()
    scores_full = np.zeros(len(merged_df))
    for start in range(0, len(merged_df), batch):
        end = min(start + batch, len(merged_df))
        scores_full[start:end] = chip.chip_no_collision(
            merged_df["Source IP"].iloc[start:end],
            merged_df["Destination IP"].iloc[start:end],
            merged_df["Timestamp"].iloc[start:end],
        )

    return to_pvalue_scores(scores_full, scores_ref)





def make_occon_grid(args) -> List[Tuple[int, float, int]]:
    return [(d, t, b)
            for d in args.embedding_dim_list
            for t in args.temperature_list
            for b in args.batch_size_list]


def evaluate_grid(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    seeds = args.seeds if args.seeds else [13, 23, 37, 42, 59]

    try:
        occon_triples = make_occon_grid(args)
    except AttributeError:
        occon_triples = [(d, t, b)
                         for d in getattr(args, "embedding_dim_list", [64])
                         for t in getattr(args, "temperature_list", [0.1])
                         for b in getattr(args, "batch_size_list", [256])]

    dataset_list = list(args.cut_off_list)
    shards = int(getattr(args, "array_shards", 1))
    array_id = int(getattr(args, "array_id", 0))
    if shards > 1:
        dataset_list = [val for idx, val in enumerate(dataset_list) if (idx % shards) == array_id]
        if not dataset_list:
            print(f"[WARN] Shard {array_id}/{shards} received no dataset items.")
            return

    all_rows = []
    allowed_methods = {"Chip_only", "OCCon_only", "OCCon+Chip"}
    f1_key_global = None

    for seed in seeds:
        set_seed(seed)
        print(f"\n========== Seed {seed} ==========")

        for cut_off in dataset_list:
            try:
                db = load_and_sample_data(args.dataset, args.data_dir, cut_off, seed)
            except Exception as exc:
                print(f"[ERROR] Failed to load data (cut_off={cut_off}): {exc}")
                continue

            y_true = db.merged_df["Label"].values.astype(int)
            f1_percentile = db.f1_percentile if args.f1_percentile == "auto" else float(args.f1_percentile)

            chip_scores = None
            if getattr(args, "enable_chip", False):
                chip_scores = chip_pvalue_scores(db.benign_df, db.merged_df)

            if chip_scores is not None:
                chip_metrics = compute_metrics(chip_scores, y_true, f1_percentile)
                if f1_key_global is None:
                    f1_key_global = next(k for k in chip_metrics if str(k).startswith("F1@"))
                all_rows.append({
                    "seed": seed,
                    "method": "Chip_only",
                    "cut_off": cut_off,
                    "embedding_dim": None,
                    "temperature": None,
                    "batch_size": None,
                    "k_skmeans": None,
                    "k_odaes": None,
                    "AUROC": chip_metrics["AUROC"],
                    "AUPR": chip_metrics["AUPR"],
                    "ACC": chip_metrics["ACC"],
                    "Precision": chip_metrics["Precision"],
                    "Recall": chip_metrics["Recall"],
                    f1_key_global: chip_metrics[f1_key_global],
                })

            for emb_dim, temp, batch_size in occon_triples:
                cfg = OCConConfig(
                    batch_size=batch_size,
                    embedding_dim=emb_dim,
                    projection_dim=256,
                    temperature_occon=temp,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    epochs=args.epochs,
                    early_stopping_patience=args.early_stopping_patience,
                    min_epochs=args.min_epochs,
                    num_workers=args.num_workers,
                )

                emb_benign, emb_full = train_occon_and_embed(db, cfg, seed, device=device)

                sk_scores_by_k = {}
                for k in args.k_skmeans_list:
                    sk_scores, _ = spherical_kmeans_scores(emb_benign, emb_full, k, seed=seed)
                    sk_scores_by_k[k] = sk_scores

                for k, scores in sk_scores_by_k.items():
                    oc_metrics = compute_metrics(scores, y_true, f1_percentile)
                    if f1_key_global is None:
                        f1_key_global = next(k for k in oc_metrics if str(k).startswith("F1@"))
                    all_rows.append({
                        "seed": seed,
                        "method": "OCCon_only",
                        "cut_off": cut_off,
                        "embedding_dim": emb_dim,
                        "temperature": temp,
                        "batch_size": batch_size,
                        "k_skmeans": k,
                        "k_odaes": None,
                        "AUROC": oc_metrics["AUROC"],
                        "AUPR": oc_metrics["AUPR"],
                        "ACC": oc_metrics["ACC"],
                        "Precision": oc_metrics["Precision"],
                        "Recall": oc_metrics["Recall"],
                        f1_key_global: oc_metrics[f1_key_global],
                    })

                if chip_scores is not None:
                    for k, scores in sk_scores_by_k.items():
                        fused = scores + chip_scores
                        fused_metrics = compute_metrics(fused, y_true, f1_percentile)
                        if f1_key_global is None:
                            f1_key_global = next(k for k in fused_metrics if str(k).startswith("F1@"))
                        all_rows.append({
                            "seed": seed,
                            "method": "OCCon+Chip",
                            "cut_off": cut_off,
                            "embedding_dim": emb_dim,
                            "temperature": temp,
                            "batch_size": batch_size,
                            "k_skmeans": k,
                            "k_odaes": None,
                            "AUROC": fused_metrics["AUROC"],
                            "AUPR": fused_metrics["AUPR"],
                            "ACC": fused_metrics["ACC"],
                            "Precision": fused_metrics["Precision"],
                            "Recall": fused_metrics["Recall"],
                            f1_key_global: fused_metrics[f1_key_global],
                        })

    if not all_rows:
        print("[ERROR] No results were produced.")
        return

    flat_df = pd.DataFrame(all_rows)
    flat_df = flat_df[flat_df["method"].isin(allowed_methods)].copy()

    out_flat = os.path.join(args.out_dir, "all_scores_flat.csv")
    flat_df.to_csv(out_flat, index=False)
    print("=== (1) All scores flat (saved):", out_flat)

    f1_base = next(col for col in flat_df.columns if isinstance(col, str) and col.startswith("F1@"))

    hp_keys = [
        "method",
        "cut_off",
        "embedding_dim",
        "temperature",
        "batch_size",
        "k_skmeans",
        "k_odaes",
    ]

    def _std_or_zero(values: pd.Series) -> float:
        return values.std(ddof=1) if len(values) > 1 else 0.0

    grouped = flat_df.groupby(hp_keys, dropna=False)
    hp_agg = grouped.agg(
        AUROC_mean=("AUROC", "mean"),
        AUROC_std=("AUROC", _std_or_zero),
        AUPR_mean=("AUPR", "mean"),
        AUPR_std=("AUPR", _std_or_zero),
        ACC_mean=("ACC", "mean"),
        ACC_std=("ACC", _std_or_zero),
        Precision_mean=("Precision", "mean"),
        Precision_std=("Precision", _std_or_zero),
        Recall_mean=("Recall", "mean"),
        Recall_std=("Recall", _std_or_zero),
        **{f"{f1_base}_mean": (f1_base, "mean"), f"{f1_base}_std": (f1_base, _std_or_zero)},
        num_seeds=("seed", "nunique"),
    ).reset_index()

    sel_method = "OCCon+Chip" if "OCCon+Chip" in hp_agg["method"].unique() else "OCCon_only"
    subset_pick = hp_agg[hp_agg["method"] == sel_method]
    if len(subset_pick) == 0:
        selected_params_df = pd.DataFrame()
        print("[WARN] No rows available to select parameters for CSV(2); skipping.")
    else:
        idx_best = subset_pick["AUPR_mean"].idxmax()
        best_row = subset_pick.loc[idx_best]

        sel_cut = best_row["cut_off"]
        sel_emb = best_row["embedding_dim"]
        sel_temp = best_row["temperature"]
        sel_bs = best_row["batch_size"]
        sel_ksk = best_row["k_skmeans"]

        rows_for_output = []
        for method in ["Chip_only", "OCCon_only", "OCCon+Chip"]:
            if method == "Chip_only":
                sub = hp_agg[(hp_agg["method"] == method) & (hp_agg["cut_off"] == sel_cut)]
            else:
                sub = hp_agg[
                    (hp_agg["method"] == method)
                    & (hp_agg["cut_off"] == sel_cut)
                    & (hp_agg["embedding_dim"].fillna(-1) == (sel_emb if pd.notna(sel_emb) else -1))
                    & (hp_agg["temperature"].fillna(-1) == (sel_temp if pd.notna(sel_temp) else -1))
                    & (hp_agg["batch_size"].fillna(-1) == (sel_bs if pd.notna(sel_bs) else -1))
                    & (hp_agg["k_skmeans"].fillna(-1) == (sel_ksk if pd.notna(sel_ksk) else -1))
                ]
            if len(sub) == 0:
                continue
            row = sub.iloc[0].to_dict()
            row.update({
                "embedding_dim": sel_emb,
                "temperature": sel_temp,
                "batch_size": sel_bs,
                "k_skmeans": sel_ksk,
            })
            rows_for_output.append(row)

        selected_params_df = pd.DataFrame(rows_for_output)

    out_selected = os.path.join(args.out_dir, "selected_params_means_all_methods.csv")
    if len(selected_params_df) > 0:
        selected_params_df.to_csv(out_selected, index=False)
        print("=== (2) Selected params (saved):", out_selected)
    else:
        print("=== (2) Selected params CSV not created (no rows).")

    best_rows = []
    for method in ["Chip_only", "OCCon_only", "OCCon+Chip"]:
        sub = flat_df[flat_df["method"] == method]
        if len(sub) == 0:
            continue
        idx_best = sub["AUPR"].idxmax()
        best = sub.loc[idx_best].to_dict()
        best_rows.append({
            "method": method,
            "seed": best["seed"],
            "cut_off": best["cut_off"],
            "embedding_dim": best.get("embedding_dim"),
            "temperature": best.get("temperature"),
            "batch_size": best.get("batch_size"),
            "k_skmeans": best.get("k_skmeans"),
            "AUROC": best["AUROC"],
            "AUPR": best["AUPR"],
            "ACC": best["ACC"],
            "Precision": best["Precision"],
            "Recall": best["Recall"],
            f1_base: best[f1_base],
        })

    best_aupr_df = pd.DataFrame(best_rows).reset_index(drop=True)
    out_best = os.path.join(args.out_dir, "best_aupr_per_method.csv")
    best_aupr_df.to_csv(out_best, index=False)
    print("=== (3) Best AUPR per method (saved):", out_best)


def parse_arguments():
    p = argparse.ArgumentParser(description="Comprehensive evaluation for OCCon + {SKMeans, CHIP, ODAES}")
    p.add_argument("--dataset", type=str, default="unsw", choices=SUPPORTED_DATASETS,
                   help="Dataset to preprocess: unsw, cicids2017, or cicids2018")
    p.add_argument("--data_dir", type=str, default="auto",
                   help="Path to dataset files. Use 'auto' to pick the default directory for the chosen dataset.")
    p.add_argument("--out_dir", type=str, default="chipoccon_outputs",
                   help="Where to save CSV/JSON outputs")
    p.add_argument("--array_id", type=int, default=0,
                   help="Index of this shard (0-based). Used only if --array_shards>1")
    p.add_argument("--array_shards", type=int, default=1,
                   help="How many shards to split the OCCon train grid into. "
                        "Shards split the (embedding_dim x temperature) combos.")
    # Seeds
    p.add_argument("--seeds", type=int, nargs="+", default=[13], help="List of seeds")

    # Training-dataset grid
    p.add_argument("--cut_off_list", type=float, nargs="+", default=[1],
                   help="Cutoff over [0,1] of max timestamp")

    # OCCon grid
    p.add_argument("--embedding_dim_list", type=int, nargs="+", default=[128],
                   help="Embedding dims for OCCon encoder")
    p.add_argument("--temperature_list", type=float, nargs="+", default=[0.1],
                   help="OCCon temperatures")

    # SKMeans grid
    p.add_argument("--k_skmeans_list", type=int, nargs="+", default=[2, 3, 5, 10, 20, 30],
                   help="Cluster counts for Spherical KMeans")

    # F1 percentile
    p.add_argument("--f1_percentile", type=str, default="auto",
                   help="'auto' -> benign arrival percentage; or pass a numeric like 75.77")

    # Training hyperparams
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)


    p.add_argument("--epochs", type=int, default=200, help="Maximum epochs for OCCon")
    p.add_argument("--min_epochs", type=int, default=30, help="Train at least this many epochs before early stopping")
    p.add_argument("--early_stopping_patience", type=int, default=10, help="Early stopping patience after min_epochs")

    p.add_argument("--num_workers", type=int, default=0)
    # Toggles
    p.add_argument("--enable_chip", action="store_true", dest="enable_chip",
                   help="Enable OCCon+Chip combined evaluation")

    # Batch-size grid for OCCon
    p.add_argument("--batch_size_list", type=int, nargs="+", default=[256],
                   help="Batch sizes for OCCon encoder training (grid).")

    return p.parse_args()


def main():
    args = parse_arguments()
    args.data_dir = resolve_data_dir(args.dataset, args.data_dir)

    if args.array_shards > 1:
        args.out_dir = os.path.join(args.out_dir, f"shard_{args.array_id}_of_{args.array_shards}")
    os.makedirs(args.out_dir, exist_ok=True)
    evaluate_grid(args)


if __name__ == "__main__":
    main()
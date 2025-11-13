import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from .dataset_utils import SupervisedContrastiveDataset, LabelAwareBatchSampler
from .models import OCConNet
from .losses import OCConLossFn
import pandas as pd

def set_seed_local(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_occon_and_embed(db, cfg, seed: int, device: str = None):
    set_seed_local(seed)
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
    steps_per_epoch = max(1, int(np.ceil(len(train_ds) / eff_bs)))
    train_sampler = LabelAwareBatchSampler(
        labels=train_labels_np,
        batch_size=eff_bs,
        num_views=2,
        steps_per_epoch=steps_per_epoch,
        seed=seed,
    )
    from torch.utils.data import DataLoader
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
        X_full = db.scaler.transform(X_full).astype("float32")
        batch_size = 8192
        chunks = []
        for start in range(0, len(X_full), batch_size):
            chunk = torch.tensor(X_full[start:start + batch_size], dtype=torch.float32, device=device)
            emb, _ = model(chunk)
            chunks.append(emb.cpu().numpy())
        emb_full = np.vstack(chunks)

    return emb_benign, emb_full

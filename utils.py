import random
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix, f1_score
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def to_pvalue_scores(test_scores: np.ndarray, ref_scores: np.ndarray) -> np.ndarray:
    sorted_ref = np.sort(ref_scores)
    n_ref = len(sorted_ref)
    idx = np.searchsorted(sorted_ref, test_scores, side='left')
    counts_ge = n_ref - idx
    p_values = (counts_ge + 1.0) / (n_ref + 1.0)
    return -np.log(p_values)


def compute_metrics(scores: np.ndarray, y_true: np.ndarray, f1_percentile: float):
    auroc = roc_auc_score(y_true, scores)
    aupr = average_precision_score(y_true, scores)

    thr = np.percentile(scores, f1_percentile)
    y_pred = (scores >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

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


def make_occon_grid(args):
    return [(d, t, b)
            for d in args.embedding_dim_list
            for t in args.temperature_list
            for b in args.batch_size_list]

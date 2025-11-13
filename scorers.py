from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
import numpy as np

from .utils import to_pvalue_scores
from .chip import Chip


def spherical_kmeans_scores(emb_benign: np.ndarray,
                            emb_full: np.ndarray,
                            k: int,
                            seed: int):
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


def chip_pvalue_scores(benign_df, merged_df):
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

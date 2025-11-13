import os
from dataclasses import dataclass
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


@dataclass
class DataBundle:
    benign_df: pd.DataFrame
    merged_df: pd.DataFrame
    columns_numeric: List[str]
    scaler: MinMaxScaler
    label_encoder: LabelEncoder
    X_benign_scaled: "np.ndarray"
    y_benign_edges: "np.ndarray"
    input_dim: int
    f1_percentile: float


SUPPORTED_DATASETS = ("unsw", "cicids2017", "cicids2018")

_PACKAGE_ROOT = Path(__file__).resolve().parent
_DATA_ROOT = _PACKAGE_ROOT.parent / "data"

DATASET_DEFAULT_DIRS = {
    "unsw": str(_DATA_ROOT / "NF-UNSWNB15-v3"),
    "cicids2017": str(_DATA_ROOT / "CICIDS2017"),
    "cicids2018": str(_DATA_ROOT / "NF-2018-v3"),
}


def resolve_data_dir(dataset: str, user_data_dir: str) -> str:
    dataset_key = dataset.lower()
    if dataset_key not in DATASET_DEFAULT_DIRS:
        raise ValueError(f"Unsupported dataset '{dataset}'. Choose from: {', '.join(SUPPORTED_DATASETS)}")

    if user_data_dir and user_data_dir.lower() != "auto":
        return user_data_dir

    return DATASET_DEFAULT_DIRS[dataset_key]


# The dataset loaders below are intentionally direct translations of the
# corresponding functions in the original `main.py`. They return a
# `DataBundle` instance ready for training/evaluation.


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

        "DNS_QUERY_ID",
        "FTP_COMMAND_RET_CODE",

        "Label",
        "Attack",
        "Edge"]

    columns_numeric = [col for col in merged_df.columns if col not in columns_text]

    X_benign_numeric = benign_df[columns_numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scaler = MinMaxScaler()
    X_benign_scaled = scaler.fit_transform(X_benign_numeric).astype("float32")
    edge_labels = benign_df["Edge"].copy()
    label_encoder = LabelEncoder()
    y_benign_edges = label_encoder.fit_transform(edge_labels).astype("int64")
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
    X_benign_scaled = scaler.fit_transform(X_benign_numeric).astype("float32")
    edge_labels = benign_df["Edge"].copy()
    label_encoder = LabelEncoder()
    y_benign_edges = label_encoder.fit_transform(edge_labels).astype("int64")
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
    X_benign_scaled = scaler.fit_transform(X_benign_numeric).astype("float32")
    edge_labels = benign_df["Edge"].copy()
    label_encoder = LabelEncoder()
    y_benign_edges = label_encoder.fit_transform(edge_labels).astype("int64")
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


def load_and_sample_data(dataset: str, data_dir: str, cut_off: float, seed: int) -> DataBundle:
    dataset_key = dataset.lower()
    if dataset_key not in _DATASET_LOADERS:
        raise ValueError(f"Unsupported dataset '{dataset}'. Choose from: {', '.join(SUPPORTED_DATASETS)}")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found for dataset '{dataset_key}': {data_dir}")

    return _DATASET_LOADERS[dataset_key](data_dir, cut_off, seed)

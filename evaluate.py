import os
import pandas as pd
import numpy as np

from .datasets import load_and_sample_data
from .trainer import train_occon_and_embed
from .scorers import spherical_kmeans_scores, chip_pvalue_scores
from .utils import set_seed, make_occon_grid, compute_metrics


def evaluate_grid(args):
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
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
                from types import SimpleNamespace
                cfg = SimpleNamespace(
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

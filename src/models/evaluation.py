from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def prediction_error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    return {"rmse": rmse, "mae": mae}


def precision_recall_at_k(
    recommendations: dict[int, list[int]],
    test_df: pd.DataFrame,
    k: int = 10,
    like_threshold: float = 4.0,
) -> dict[str, float]:
    liked = test_df[test_df["rating"] >= like_threshold]
    truth: dict[int, set[int]] = defaultdict(set)
    for row in liked.itertuples(index=False):
        truth[int(row.userId)].add(int(row.movieId))

    precisions: list[float] = []
    recalls: list[float] = []
    for user_id, recs in recommendations.items():
        target = truth.get(int(user_id), set())
        if not target:
            continue
        topk = recs[:k]
        if not topk:
            continue
        hits = len(set(topk) & target)
        precisions.append(hits / len(topk))
        recalls.append(hits / len(target))

    if not precisions:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "users_evaluated": 0}
    return {
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls)),
        "users_evaluated": int(len(precisions)),
    }


def catalog_coverage(recommendations: dict[int, list[int]], catalog_size: int) -> float:
    if catalog_size == 0:
        return 0.0
    rec_items = set()
    for items in recommendations.values():
        rec_items.update(items)
    return float(len(rec_items) / catalog_size)


def recommendation_diversity(
    recommendations: dict[int, list[int]],
    movie_to_index: dict[int, int],
    item_factors: np.ndarray,
    sample_users: int = 500,
) -> float:
    users = list(recommendations.keys())[:sample_users]
    diversities: list[float] = []
    for uid in users:
        items = [movie_to_index[m] for m in recommendations[uid] if m in movie_to_index]
        if len(items) < 2:
            continue
        vecs = item_factors[items]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        denom = norms @ norms.T
        dots = vecs @ vecs.T
        sim = np.divide(dots, denom, out=np.zeros_like(dots, dtype=np.float32), where=denom > 0)
        iu = np.triu_indices_from(sim, k=1)
        avg_sim = float(sim[iu].mean()) if len(iu[0]) else 0.0
        diversities.append(1.0 - avg_sim)
    return float(np.mean(diversities)) if diversities else 0.0


def popularity_bias(
    recommendations: dict[int, list[int]],
    train_df: pd.DataFrame,
    top_fraction: float = 0.1,
) -> float:
    pop = train_df.groupby("movieId")["userId"].size().sort_values(ascending=False)
    cutoff = max(1, int(len(pop) * top_fraction))
    head = set(pop.head(cutoff).index.tolist())
    all_recs = [m for v in recommendations.values() for m in v]
    if not all_recs:
        return 0.0
    in_head = sum(1 for m in all_recs if m in head)
    return float(in_head / len(all_recs))

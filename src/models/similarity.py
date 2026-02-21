from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix


def _row_normalize(matrix: csr_matrix) -> csr_matrix:
    matrix = matrix.tocsr().astype(np.float32)
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    norms[norms == 0] = 1.0
    inv = 1.0 / norms
    return matrix.multiply(inv[:, None]).tocsr()


def top_k_similar_users(user_idx: int, matrix: csr_matrix, k: int = 40) -> tuple[np.ndarray, np.ndarray]:
    normed = _row_normalize(matrix)
    scores = (normed @ normed.getrow(user_idx).T).toarray().ravel()
    scores[user_idx] = -1.0
    k = min(k, len(scores) - 1)
    if k <= 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return idx.astype(np.int32), scores[idx].astype(np.float32)


def top_k_similar_items(item_idx: int, matrix: csr_matrix, k: int = 40) -> tuple[np.ndarray, np.ndarray]:
    item_matrix = matrix.T.tocsr()
    normed = _row_normalize(item_matrix)
    scores = (normed @ normed.getrow(item_idx).T).toarray().ravel()
    scores[item_idx] = -1.0
    k = min(k, len(scores) - 1)
    if k <= 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return idx.astype(np.int32), scores[idx].astype(np.float32)


def user_based_recommendations(
    user_id: int,
    matrix: csr_matrix,
    user_to_index: dict[int, int],
    movie_ids: np.ndarray,
    k_neighbors: int = 40,
    top_n: int = 10,
) -> list[tuple[int, float]]:
    if user_id not in user_to_index:
        return []
    user_idx = user_to_index[user_id]
    neighbors, weights = top_k_similar_users(user_idx, matrix, k=k_neighbors)
    item_count = matrix.shape[1]
    score_sum = np.zeros(item_count, dtype=np.float32)
    weight_sum = np.zeros(item_count, dtype=np.float32)

    for neighbor_idx, w in zip(neighbors, weights):
        if w <= 0:
            continue
        row = matrix.getrow(int(neighbor_idx))
        score_sum[row.indices] += w * row.data
        weight_sum[row.indices] += abs(w)

    preds = np.divide(
        score_sum,
        weight_sum,
        out=np.zeros_like(score_sum, dtype=np.float32),
        where=weight_sum > 0,
    )
    rated = matrix.getrow(user_idx).indices
    preds[rated] = -np.inf
    top_n = min(top_n, item_count)
    if top_n <= 0:
        return []
    rec_idx = np.argpartition(preds, -top_n)[-top_n:]
    rec_idx = rec_idx[np.argsort(preds[rec_idx])[::-1]]
    return [(int(movie_ids[i]), float(preds[i])) for i in rec_idx if np.isfinite(preds[i])]


def item_based_recommendations(
    user_id: int,
    matrix: csr_matrix,
    user_to_index: dict[int, int],
    movie_ids: np.ndarray,
    k_similar_items: int = 60,
    min_liked_rating: float = 4.0,
    top_n: int = 10,
) -> list[tuple[int, float]]:
    if user_id not in user_to_index:
        return []
    user_idx = user_to_index[user_id]
    row = matrix.getrow(user_idx)
    liked_items = row.indices[row.data >= min_liked_rating]
    liked_ratings = row.data[row.data >= min_liked_rating]
    if len(liked_items) == 0:
        return []

    score_sum = np.zeros(matrix.shape[1], dtype=np.float32)
    weight_sum = np.zeros(matrix.shape[1], dtype=np.float32)

    for item_idx, rating in zip(liked_items, liked_ratings):
        sims_idx, sims = top_k_similar_items(int(item_idx), matrix, k=k_similar_items)
        positive = sims > 0
        sims_idx = sims_idx[positive]
        sims = sims[positive]
        score_sum[sims_idx] += sims * rating
        weight_sum[sims_idx] += np.abs(sims)

    preds = np.divide(score_sum, weight_sum, out=np.zeros_like(score_sum), where=weight_sum > 0)
    preds[row.indices] = -np.inf
    top_n = min(top_n, matrix.shape[1])
    if top_n <= 0:
        return []
    rec_idx = np.argpartition(preds, -top_n)[-top_n:]
    rec_idx = rec_idx[np.argsort(preds[rec_idx])[::-1]]
    return [(int(movie_ids[i]), float(preds[i])) for i in rec_idx if np.isfinite(preds[i])]


def recommendations_to_dict(recs: Iterable[tuple[int, float]]) -> dict[int, float]:
    return {int(movie_id): float(score) for movie_id, score in recs}

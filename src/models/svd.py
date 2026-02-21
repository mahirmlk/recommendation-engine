from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


@dataclass
class SVDArtifacts:
    user_factors: np.ndarray
    item_factors: np.ndarray
    singular_values: np.ndarray


def train_svd(
    matrix: csr_matrix,
    n_factors: int = 100,
    n_iter: int = 10,
    random_state: int = 42,
) -> SVDArtifacts:
    max_rank = min(matrix.shape) - 1
    if max_rank < 2:
        raise ValueError("interaction matrix is too small for SVD")
    k = max(2, min(n_factors, max_rank))
    u, s, vt = svds(matrix.astype(np.float32), k=k, return_singular_vectors=True, maxiter=max(1000, n_iter * 100))
    order = np.argsort(s)[::-1]
    s = s[order].astype(np.float32)
    u = u[:, order].astype(np.float32)
    vt = vt[order, :].astype(np.float32)

    sigma_sqrt = np.sqrt(s)
    user_factors = u * sigma_sqrt[None, :]
    item_factors = (vt.T * sigma_sqrt[None, :]).astype(np.float32)
    return SVDArtifacts(user_factors=user_factors, item_factors=item_factors, singular_values=s)


def predict_pairs(
    artifacts: SVDArtifacts,
    user_indices: np.ndarray,
    item_indices: np.ndarray,
) -> np.ndarray:
    u = artifacts.user_factors[user_indices]
    v = artifacts.item_factors[item_indices]
    preds = np.sum(u * v, axis=1)
    return preds.astype(np.float32)


def recommend_from_factors(
    user_id: int,
    matrix: csr_matrix,
    user_to_index: dict[int, int],
    movie_ids: np.ndarray,
    artifacts: SVDArtifacts,
    top_n: int = 10,
) -> list[tuple[int, float]]:
    if user_id not in user_to_index:
        return []
    uidx = user_to_index[user_id]
    scores = artifacts.item_factors @ artifacts.user_factors[uidx]
    rated = matrix.getrow(uidx).indices
    scores = scores.astype(np.float32)
    scores[rated] = -np.inf
    top_n = min(top_n, len(scores))
    idx = np.argpartition(scores, -top_n)[-top_n:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return [(int(movie_ids[i]), float(scores[i])) for i in idx if np.isfinite(scores[i])]


def tune_svd_factors(
    matrix: csr_matrix,
    test_df: pd.DataFrame,
    user_to_index: dict[int, int],
    movie_to_index: dict[int, int],
    factors: list[int],
    sample_size: int = 200000,
) -> pd.DataFrame:
    valid = test_df[
        test_df["userId"].isin(user_to_index.keys()) & test_df["movieId"].isin(movie_to_index.keys())
    ].copy()
    if valid.empty:
        return pd.DataFrame(columns=["factors", "rmse", "mae", "n_eval"])
    if len(valid) > sample_size:
        valid = valid.sample(sample_size, random_state=42)
    uidx = valid["userId"].map(user_to_index).to_numpy(np.int32)
    midx = valid["movieId"].map(movie_to_index).to_numpy(np.int32)
    y_true = valid["rating"].to_numpy(np.float32)

    rows: list[dict[str, float | int]] = []
    for f in factors:
        artifacts = train_svd(matrix, n_factors=f)
        preds = predict_pairs(artifacts, uidx, midx)
        errors = preds - y_true
        rmse = float(np.sqrt(np.mean(errors**2)))
        mae = float(np.mean(np.abs(errors)))
        rows.append({"factors": int(f), "rmse": rmse, "mae": mae, "n_eval": int(len(valid))})
    return pd.DataFrame(rows)


def factor_top_movies(
    artifacts: SVDArtifacts,
    movie_ids: np.ndarray,
    movies: pd.DataFrame,
    top_n: int = 10,
) -> dict[int, list[dict[str, float | int | str]]]:
    title_map = movies.set_index("movieId")["title"].to_dict()
    out: dict[int, list[dict[str, float | int | str]]] = {}
    for f in range(artifacts.item_factors.shape[1]):
        loadings = artifacts.item_factors[:, f]
        idx = np.argpartition(loadings, -top_n)[-top_n:]
        idx = idx[np.argsort(loadings[idx])[::-1]]
        out[f] = [
            {
                "movieId": int(movie_ids[i]),
                "title": str(title_map.get(int(movie_ids[i]), f"movie-{int(movie_ids[i])}")),
                "loading": float(loadings[i]),
            }
            for i in idx
        ]
    return out

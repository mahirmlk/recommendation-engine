from __future__ import annotations

import heapq
from collections import defaultdict

import numpy as np
import pandas as pd


def build_journey_edges(
    item_factors: np.ndarray,
    movie_ids: np.ndarray,
    k_neighbors: int = 20,
    min_similarity: float = 0.25,
) -> pd.DataFrame:
    try:
        from sklearn.neighbors import NearestNeighbors
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "scikit-learn is required for build_journey_edges. Install with: pip install scikit-learn"
        ) from e

    k = min(k_neighbors + 1, item_factors.shape[0])
    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=k)
    nn.fit(item_factors)
    distances, indices = nn.kneighbors(item_factors)

    src_col: list[int] = []
    dst_col: list[int] = []
    sim_col: list[float] = []
    weight_col: list[float] = []
    for i in range(indices.shape[0]):
        src = int(movie_ids[i])
        for j, dist in zip(indices[i, 1:], distances[i, 1:]):
            sim = 1.0 - float(dist)
            if sim < min_similarity:
                continue
            dst = int(movie_ids[int(j)])
            src_col.append(src)
            dst_col.append(dst)
            sim_col.append(sim)
            weight_col.append(1.0 - sim)

    return pd.DataFrame(
        {
            "source_movie_id": src_col,
            "target_movie_id": dst_col,
            "similarity": sim_col,
            "weight": weight_col,
        }
    )


def _adjacency_from_edges(edges: pd.DataFrame) -> dict[int, list[tuple[int, float]]]:
    adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for row in edges.itertuples(index=False):
        s = int(row.source_movie_id)
        t = int(row.target_movie_id)
        w = float(row.weight)
        adj[s].append((t, w))
        adj[t].append((s, w))
    return adj


def find_shortest_journey(
    edges: pd.DataFrame,
    source_movie_id: int,
    target_movie_id: int,
) -> tuple[list[int], float]:
    if source_movie_id == target_movie_id:
        return [source_movie_id], 0.0
    adj = _adjacency_from_edges(edges)
    if source_movie_id not in adj or target_movie_id not in adj:
        return [], float("inf")

    dist: dict[int, float] = {source_movie_id: 0.0}
    prev: dict[int, int] = {}
    pq: list[tuple[float, int]] = [(0.0, source_movie_id)]
    seen: set[int] = set()

    while pq:
        cur_dist, node = heapq.heappop(pq)
        if node in seen:
            continue
        seen.add(node)
        if node == target_movie_id:
            break
        for nxt, w in adj[node]:
            cand = cur_dist + w
            if cand < dist.get(nxt, float("inf")):
                dist[nxt] = cand
                prev[nxt] = node
                heapq.heappush(pq, (cand, nxt))

    if target_movie_id not in dist:
        return [], float("inf")

    path = [target_movie_id]
    while path[-1] != source_movie_id:
        path.append(prev[path[-1]])
    path.reverse()
    return path, float(dist[target_movie_id])


def describe_journey(path: list[int], movies: pd.DataFrame) -> list[str]:
    if not path:
        return []
    title_map = movies.set_index("movieId")["title"].to_dict()
    lines: list[str] = []
    for i in range(len(path) - 1):
        a = title_map.get(path[i], str(path[i]))
        b = title_map.get(path[i + 1], str(path[i + 1]))
        lines.append(f"{a} -> {b}")
    return lines

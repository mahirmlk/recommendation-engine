from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def enrich_movies(movies: pd.DataFrame) -> pd.DataFrame:
    out = movies.copy()
    out["genres"] = out["genres"].fillna("(no genres listed)").astype("string")
    out["genre_list"] = out["genres"].str.split("|")
    out["genre_count"] = out["genre_list"].str.len().astype("int16")
    out["primary_genre"] = out["genre_list"].str[0].fillna("(no genres listed)")
    year = out["title"].str.extract(r"\((\d{4})\)\s*$", expand=False)
    out["year"] = pd.to_numeric(year, errors="coerce").astype("Int32")
    out["decade"] = (out["year"] // 10 * 10).astype("Int32")
    return out


def build_movie_genre_matrix(movies: pd.DataFrame) -> pd.DataFrame:
    exploded = movies[["movieId", "genres"]].copy()
    exploded["genres"] = exploded["genres"].fillna("(no genres listed)").str.split("|")
    exploded = exploded.explode("genres").rename(columns={"genres": "genre"})
    genre_matrix = pd.crosstab(exploded["movieId"], exploded["genre"]).astype("int8")
    return genre_matrix


def build_genre_cooccurrence(movies: pd.DataFrame) -> pd.DataFrame:
    genre_matrix = build_movie_genre_matrix(movies)
    co = genre_matrix.T @ genre_matrix
    return co.astype("int32")


def temporal_train_test_split(
    ratings: pd.DataFrame,
    test_ratio: float = 0.2,
    min_user_ratings_for_test: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if ratings.empty:
        return ratings.copy(), ratings.copy()

    ordered = ratings.sort_values(["userId", "timestamp", "movieId"]).reset_index(drop=True)
    group_size = ordered.groupby("userId")["movieId"].transform("size")
    base_test = np.floor(group_size * test_ratio).astype("int32")
    base_test = base_test.clip(1)
    eligible = group_size >= min_user_ratings_for_test
    base_test = np.where(eligible, base_test, 0)
    base_test = np.minimum(base_test, (group_size - 1).clip(lower=0))

    tail_rank = ordered.groupby("userId").cumcount(ascending=False) + 1
    is_test = tail_rank <= base_test
    train = ordered.loc[~is_test].copy()
    test = ordered.loc[is_test].copy()

    if test.empty:
        return train, test

    train_movie_ids = set(train["movieId"].unique().tolist())
    missing_in_train = ~test["movieId"].isin(train_movie_ids)
    if missing_in_train.any():
        moved = test.loc[missing_in_train].copy()
        train = pd.concat([train, moved], ignore_index=True)
        test = test.loc[~missing_in_train].copy()

    return train.reset_index(drop=True), test.reset_index(drop=True)


@dataclass
class MatrixBundle:
    matrix: csr_matrix
    user_ids: np.ndarray
    movie_ids: np.ndarray
    user_to_index: dict[int, int]
    movie_to_index: dict[int, int]


def build_interaction_matrix(ratings: pd.DataFrame) -> MatrixBundle:
    if ratings.empty:
        raise ValueError("ratings is empty")

    user_ids = np.sort(ratings["userId"].unique())
    movie_ids = np.sort(ratings["movieId"].unique())
    user_to_index = {int(uid): idx for idx, uid in enumerate(user_ids)}
    movie_to_index = {int(mid): idx for idx, mid in enumerate(movie_ids)}

    row_idx = ratings["userId"].map(user_to_index).to_numpy(dtype=np.int32)
    col_idx = ratings["movieId"].map(movie_to_index).to_numpy(dtype=np.int32)
    values = ratings["rating"].to_numpy(dtype=np.float32)

    matrix = csr_matrix(
        (values, (row_idx, col_idx)),
        shape=(len(user_ids), len(movie_ids)),
        dtype=np.float32,
    )

    return MatrixBundle(
        matrix=matrix,
        user_ids=user_ids.astype(np.int32),
        movie_ids=movie_ids.astype(np.int32),
        user_to_index=user_to_index,
        movie_to_index=movie_to_index,
    )


def build_high_rating_matrix(
    ratings: pd.DataFrame,
    user_to_index: dict[int, int],
    movie_to_index: dict[int, int],
    threshold: float = 4.0,
) -> csr_matrix:
    strong = ratings.loc[ratings["rating"] >= threshold, ["userId", "movieId"]]
    mapped = strong.copy()
    mapped["u"] = mapped["userId"].map(user_to_index)
    mapped["m"] = mapped["movieId"].map(movie_to_index)
    mapped = mapped.dropna(subset=["u", "m"])
    row_idx = mapped["u"].to_numpy(dtype=np.int32)
    col_idx = mapped["m"].to_numpy(dtype=np.int32)
    values = np.ones(len(mapped), dtype=np.int8)
    return csr_matrix(
        (values, (row_idx, col_idx)),
        shape=(len(user_to_index), len(movie_to_index)),
        dtype=np.int8,
    )


def cap_dataset(
    ratings: pd.DataFrame,
    max_users: int | None = None,
    max_movies: int | None = None,
) -> pd.DataFrame:
    out = ratings
    if max_users is not None:
        top_users = (
            out.groupby("userId")["movieId"]
            .size()
            .sort_values(ascending=False)
            .head(max_users)
            .index
        )
        out = out[out["userId"].isin(top_users)]
    if max_movies is not None:
        top_movies = (
            out.groupby("movieId")["userId"]
            .size()
            .sort_values(ascending=False)
            .head(max_movies)
            .index
        )
        out = out[out["movieId"].isin(top_movies)]
    return out.reset_index(drop=True)


def to_serializable(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (np.generic,)):
            out[k] = v.item()
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out

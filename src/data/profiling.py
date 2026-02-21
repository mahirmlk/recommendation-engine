from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _rating_distribution(ratings: pd.DataFrame) -> dict[str, int]:
    dist = ratings["rating"].value_counts().sort_index()
    return {str(k): int(v) for k, v in dist.items()}


def _genre_popularity(ratings: pd.DataFrame, movies: pd.DataFrame, top_n: int = 20) -> list[dict[str, Any]]:
    joined = ratings.merge(movies[["movieId", "genres"]], on="movieId", how="left")
    exploded = joined.assign(genre=joined["genres"].fillna("(no genres listed)").str.split("|")).explode("genre")
    agg = (
        exploded.groupby("genre")
        .agg(ratings_count=("rating", "size"), avg_rating=("rating", "mean"))
        .sort_values("ratings_count", ascending=False)
        .head(top_n)
        .reset_index()
    )
    return [
        {
            "genre": str(row["genre"]),
            "ratings_count": int(row["ratings_count"]),
            "avg_rating": float(row["avg_rating"]),
        }
        for _, row in agg.iterrows()
    ]


def profile_dataset(ratings: pd.DataFrame, movies: pd.DataFrame) -> dict[str, Any]:
    n_ratings = int(len(ratings))
    n_users = int(ratings["userId"].nunique())
    n_movies_rated = int(ratings["movieId"].nunique())
    n_movies_catalog = int(movies["movieId"].nunique())
    possible = max(1, n_users * n_movies_rated)
    density = float(n_ratings / possible)

    top_users = (
        ratings.groupby("userId")["movieId"]
        .size()
        .sort_values(ascending=False)
        .head(10)
        .rename("rating_count")
        .reset_index()
    )
    top_movies = (
        ratings.groupby("movieId")["userId"]
        .size()
        .sort_values(ascending=False)
        .head(10)
        .rename("rating_count")
        .reset_index()
        .merge(movies[["movieId", "title"]], on="movieId", how="left")
    )

    missing = ratings.isna().sum().to_dict()
    rating_min = float(ratings["rating"].min()) if n_ratings else float("nan")
    rating_max = float(ratings["rating"].max()) if n_ratings else float("nan")
    timestamp_min = int(ratings["timestamp"].min()) if n_ratings else None
    timestamp_max = int(ratings["timestamp"].max()) if n_ratings else None

    return {
        "total_ratings": n_ratings,
        "unique_users": n_users,
        "unique_movies_rated": n_movies_rated,
        "catalog_movies": n_movies_catalog,
        "density": density,
        "rating_scale_observed": [rating_min, rating_max],
        "rating_distribution": _rating_distribution(ratings),
        "most_active_users": top_users.to_dict(orient="records"),
        "most_rated_movies": top_movies.to_dict(orient="records"),
        "genre_popularity": _genre_popularity(ratings, movies),
        "missing_values": {k: int(v) for k, v in missing.items()},
        "timestamp_range_unix": [timestamp_min, timestamp_max],
        "timestamp_range_utc": [
            pd.to_datetime(timestamp_min, unit="s", utc=True).isoformat() if timestamp_min is not None else None,
            pd.to_datetime(timestamp_max, unit="s", utc=True).isoformat() if timestamp_max is not None else None,
        ],
        "left_skew_indicator_mean_vs_median": {
            "mean_rating": float(ratings["rating"].mean()),
            "median_rating": float(np.median(ratings["rating"].to_numpy())),
        },
    }

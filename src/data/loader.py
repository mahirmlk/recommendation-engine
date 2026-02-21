from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class DatasetBundle:
    ratings: pd.DataFrame
    movies: pd.DataFrame
    users: Optional[pd.DataFrame] = None
    links: Optional[pd.DataFrame] = None
    tags: Optional[pd.DataFrame] = None


def _read_ml_csv_ratings(path: Path, max_rows: Optional[int]) -> pd.DataFrame:
    ratings = pd.read_csv(
        path,
        nrows=max_rows,
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
    )
    ratings["rating_date"] = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)
    return ratings


def _read_ml_csv_movies(path: Path) -> pd.DataFrame:
    movies = pd.read_csv(path, dtype={"movieId": "int32", "title": "string", "genres": "string"})
    movies["genres"] = movies["genres"].fillna("(no genres listed)")
    return movies


def _read_ml_1m_ratings(path: Path, max_rows: Optional[int]) -> pd.DataFrame:
    ratings = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
        nrows=max_rows,
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
    )
    ratings["rating_date"] = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)
    return ratings


def _read_ml_1m_movies(path: Path) -> pd.DataFrame:
    movies = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["movieId", "title", "genres"],
        dtype={"movieId": "int32", "title": "string", "genres": "string"},
        encoding="latin-1",
    )
    movies["genres"] = movies["genres"].fillna("(no genres listed)")
    return movies


def _read_ml_100k_ratings(path: Path, max_rows: Optional[int]) -> pd.DataFrame:
    ratings = pd.read_csv(
        path,
        sep="\t",
        names=["userId", "movieId", "rating", "timestamp"],
        nrows=max_rows,
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
    )
    ratings["rating_date"] = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)
    return ratings


def _read_ml_100k_movies(path: Path) -> pd.DataFrame:
    genre_cols = [
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]
    cols = [
        "movieId",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
        *genre_cols,
    ]
    raw = pd.read_csv(path, sep="|", names=cols, encoding="latin-1")
    genre_block = raw[genre_cols].astype("int8")
    genres = genre_block.apply(lambda row: "|".join([g for g in genre_cols if row[g] == 1]), axis=1)
    movies = raw[["movieId", "title"]].copy()
    movies["genres"] = genres.mask(genres.eq(""), "(no genres listed)").astype("string")
    movies["movieId"] = movies["movieId"].astype("int32")
    movies["title"] = movies["title"].astype("string")
    return movies


def _maybe_read_users(raw_dir: Path) -> Optional[pd.DataFrame]:
    users_csv = raw_dir / "users.csv"
    users_dat = raw_dir / "users.dat"
    u_user = raw_dir / "u.user"
    if users_csv.exists():
        return pd.read_csv(users_csv)
    if users_dat.exists():
        return pd.read_csv(
            users_dat,
            sep="::",
            engine="python",
            names=["userId", "gender", "age", "occupation", "zip_code"],
        )
    if u_user.exists():
        return pd.read_csv(
            u_user,
            sep="|",
            names=["userId", "age", "gender", "occupation", "zip_code"],
        )
    return None


def _maybe_read_links(raw_dir: Path) -> Optional[pd.DataFrame]:
    links_csv = raw_dir / "links.csv"
    return pd.read_csv(links_csv) if links_csv.exists() else None


def _maybe_read_tags(raw_dir: Path) -> Optional[pd.DataFrame]:
    tags_csv = raw_dir / "tags.csv"
    if not tags_csv.exists():
        return None
    tags = pd.read_csv(tags_csv)
    if "timestamp" in tags.columns:
        tags["tag_date"] = pd.to_datetime(tags["timestamp"], unit="s", utc=True, errors="coerce")
    return tags


def load_dataset(raw_dir: str | Path, max_rows: Optional[int] = None) -> DatasetBundle:
    raw_path = Path(raw_dir)
    ratings_csv = raw_path / "ratings.csv"
    movies_csv = raw_path / "movies.csv"
    ratings_dat = raw_path / "ratings.dat"
    movies_dat = raw_path / "movies.dat"
    u_data = raw_path / "u.data"
    u_item = raw_path / "u.item"

    if ratings_csv.exists() and movies_csv.exists():
        ratings = _read_ml_csv_ratings(ratings_csv, max_rows)
        movies = _read_ml_csv_movies(movies_csv)
    elif ratings_dat.exists() and movies_dat.exists():
        ratings = _read_ml_1m_ratings(ratings_dat, max_rows)
        movies = _read_ml_1m_movies(movies_dat)
    elif u_data.exists() and u_item.exists():
        ratings = _read_ml_100k_ratings(u_data, max_rows)
        movies = _read_ml_100k_movies(u_item)
    else:
        raise FileNotFoundError(
            "Could not detect MovieLens schema in data/raw. Expected ratings+movies files in CSV, DAT, or u.* format."
        )

    users = _maybe_read_users(raw_path)
    links = _maybe_read_links(raw_path)
    tags = _maybe_read_tags(raw_path)
    return DatasetBundle(ratings=ratings, movies=movies, users=users, links=links, tags=tags)

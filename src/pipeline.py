from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import load_dataset
from src.data.preprocessing import (
    build_high_rating_matrix,
    build_interaction_matrix,
    cap_dataset,
    enrich_movies,
    temporal_train_test_split,
    to_serializable,
)
from src.data.profiling import profile_dataset
from src.models.evaluation import (
    catalog_coverage,
    popularity_bias,
    precision_recall_at_k,
    prediction_error_metrics,
    recommendation_diversity,
)
from src.models.journey import build_journey_edges
from src.models.similarity import item_based_recommendations, user_based_recommendations
from src.models.svd import factor_top_movies, predict_pairs, recommend_from_factors, train_svd, tune_svd_factors
from src.visualization.interactive_plots import (
    build_plotly_liked_also_liked_network,
    build_plotly_movie_embedding_3d,
)
from src.visualization.static_plots import (
    compute_embedding_frame,
    plot_genre_clustermap,
    plot_rating_box_by_genre,
    plot_scree,
    plot_similarity_network_matplotlib,
    plot_tsne_2d,
)


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def build_cooccurrence_edges(
    train_df: pd.DataFrame,
    high_rating_matrix,
    movie_to_index: dict[int, int],
    top_movies: int = 1500,
    min_support: int = 25,
) -> pd.DataFrame:
    popular = (
        train_df.groupby("movieId")["userId"]
        .size()
        .sort_values(ascending=False)
        .head(top_movies)
        .index.astype(int)
        .tolist()
    )
    cols = [movie_to_index[m] for m in popular if m in movie_to_index]
    mids = [m for m in popular if m in movie_to_index]
    if not cols:
        return pd.DataFrame(columns=["source_movie_id", "target_movie_id", "support", "similarity", "weight"])

    sub = high_rating_matrix[:, cols].astype(np.int32)
    support_diag = np.asarray(sub.sum(axis=0)).ravel().astype(np.float32)
    co = (sub.T @ sub).tocoo()

    src: list[int] = []
    dst: list[int] = []
    sup: list[int] = []
    sim: list[float] = []
    wgt: list[float] = []
    for r, c, d in zip(co.row, co.col, co.data):
        if r >= c:
            continue
        if int(d) < min_support:
            continue
        denom = np.sqrt(max(1.0, support_diag[r] * support_diag[c]))
        s = float(d / denom)
        src.append(int(mids[r]))
        dst.append(int(mids[c]))
        sup.append(int(d))
        sim.append(s)
        wgt.append(1.0 - s)
    return pd.DataFrame(
        {
            "source_movie_id": src,
            "target_movie_id": dst,
            "support": sup,
            "similarity": sim,
            "weight": wgt,
        }
    ).sort_values(["support", "similarity"], ascending=False)


def build_recommendation_set(
    users: list[int],
    matrix,
    bundle,
    artifacts,
    top_n: int,
) -> dict[int, list[int]]:
    recs: dict[int, list[int]] = {}
    for uid in users:
        rec = recommend_from_factors(
            user_id=uid,
            matrix=matrix,
            user_to_index=bundle.user_to_index,
            movie_ids=bundle.movie_ids,
            artifacts=artifacts,
            top_n=top_n,
        )
        recs[int(uid)] = [int(mid) for mid, _ in rec]
    return recs


def run_pipeline(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    embeddings_dir = output_dir / "embeddings"
    models_dir = output_dir / "models"
    for d in [processed_dir, figures_dir, embeddings_dir, models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(data_dir, max_rows=args.max_ratings)
    movies = enrich_movies(dataset.movies)
    ratings = dataset.ratings.copy()
    if args.max_users or args.max_movies:
        ratings = cap_dataset(ratings, max_users=args.max_users, max_movies=args.max_movies)

    rated_movie_ids = set(ratings["movieId"].unique().tolist())
    movies = movies[movies["movieId"].isin(rated_movie_ids)].drop_duplicates("movieId").reset_index(drop=True)

    profile = profile_dataset(ratings, movies)
    save_json(models_dir / "profile_summary.json", profile)
    if args.command == "profile":
        return

    train_df, test_df = temporal_train_test_split(
        ratings,
        test_ratio=args.test_ratio,
        min_user_ratings_for_test=args.min_user_ratings,
    )
    train_df.to_parquet(processed_dir / "ratings_train.parquet", index=False)
    test_df.to_parquet(processed_dir / "ratings_test.parquet", index=False)
    movies.to_parquet(processed_dir / "movies_enriched.parquet", index=False)

    matrix_bundle = build_interaction_matrix(train_df)
    high_rating = build_high_rating_matrix(
        train_df,
        user_to_index=matrix_bundle.user_to_index,
        movie_to_index=matrix_bundle.movie_to_index,
        threshold=args.like_threshold,
    )

    artifacts = train_svd(
        matrix_bundle.matrix,
        n_factors=args.svd_factors,
        n_iter=args.svd_iterations,
    )
    np.savez_compressed(
        models_dir / "svd_artifacts.npz",
        user_ids=matrix_bundle.user_ids,
        movie_ids=matrix_bundle.movie_ids,
        user_factors=artifacts.user_factors,
        item_factors=artifacts.item_factors,
        singular_values=artifacts.singular_values,
    )

    if args.command in {"train", "all", "visualize"} and args.run_tuning:
        grid = [int(x) for x in args.factor_grid.split(",") if x.strip()]
        tuning = tune_svd_factors(
            matrix_bundle.matrix,
            test_df,
            matrix_bundle.user_to_index,
            matrix_bundle.movie_to_index,
            factors=grid,
            sample_size=args.tuning_sample,
        )
        tuning.to_csv(models_dir / "svd_tuning.csv", index=False)

    valid_test = test_df[
        test_df["userId"].isin(matrix_bundle.user_to_index.keys())
        & test_df["movieId"].isin(matrix_bundle.movie_to_index.keys())
    ].copy()
    if len(valid_test) > args.eval_pairs:
        valid_test = valid_test.sample(args.eval_pairs, random_state=42)
    uidx = valid_test["userId"].map(matrix_bundle.user_to_index).to_numpy(np.int32)
    midx = valid_test["movieId"].map(matrix_bundle.movie_to_index).to_numpy(np.int32)
    y_true = valid_test["rating"].to_numpy(np.float32)
    y_pred = predict_pairs(artifacts, uidx, midx) if len(valid_test) else np.array([], dtype=np.float32)
    pred_metrics = prediction_error_metrics(y_true, y_pred) if len(valid_test) else {"rmse": 0.0, "mae": 0.0}

    eval_users = (
        test_df.groupby("userId")["movieId"]
        .size()
        .sort_values(ascending=False)
        .head(args.eval_users)
        .index.astype(int)
        .tolist()
    )
    recommendations = build_recommendation_set(
        users=eval_users,
        matrix=matrix_bundle.matrix,
        bundle=matrix_bundle,
        artifacts=artifacts,
        top_n=args.top_n,
    )
    rank_metrics = precision_recall_at_k(
        recommendations=recommendations,
        test_df=test_df,
        k=args.top_n,
        like_threshold=args.like_threshold,
    )
    coverage = catalog_coverage(recommendations, catalog_size=len(matrix_bundle.movie_ids))
    diversity = recommendation_diversity(
        recommendations=recommendations,
        movie_to_index=matrix_bundle.movie_to_index,
        item_factors=artifacts.item_factors,
    )
    pop_bias = popularity_bias(recommendations, train_df=train_df, top_fraction=0.1)

    sample_user = int(train_df["userId"].iloc[0])
    baseline_user_cf = user_based_recommendations(
        user_id=sample_user,
        matrix=matrix_bundle.matrix,
        user_to_index=matrix_bundle.user_to_index,
        movie_ids=matrix_bundle.movie_ids,
        top_n=args.top_n,
    )
    baseline_item_cf = item_based_recommendations(
        user_id=sample_user,
        matrix=matrix_bundle.matrix,
        user_to_index=matrix_bundle.user_to_index,
        movie_ids=matrix_bundle.movie_ids,
        top_n=args.top_n,
    )

    top_factors = factor_top_movies(
        artifacts=artifacts,
        movie_ids=matrix_bundle.movie_ids,
        movies=movies,
        top_n=10,
    )

    evaluation = {
        "prediction_metrics": pred_metrics,
        "ranking_metrics": rank_metrics,
        "coverage": coverage,
        "diversity": diversity,
        "popularity_bias": pop_bias,
        "evaluated_pairs": int(len(valid_test)),
        "evaluated_users": int(len(eval_users)),
        "sample_user": sample_user,
        "sample_user_recommendations_user_cf": baseline_user_cf,
        "sample_user_recommendations_item_cf": baseline_item_cf,
    }
    save_json(models_dir / "evaluation_summary.json", to_serializable(evaluation))
    save_json(models_dir / "factor_top_movies.json", to_serializable(top_factors))

    popularity = (
        train_df.groupby("movieId")["userId"]
        .size()
        .rename("rating_count")
        .reset_index()
    )
    journey_edges = build_journey_edges(
        item_factors=artifacts.item_factors,
        movie_ids=matrix_bundle.movie_ids,
        k_neighbors=args.journey_neighbors,
        min_similarity=args.journey_min_similarity,
    )
    journey_edges.to_csv(models_dir / "journey_edges.csv", index=False)

    cooccurrence_edges = build_cooccurrence_edges(
        train_df=train_df,
        high_rating_matrix=high_rating,
        movie_to_index=matrix_bundle.movie_to_index,
        top_movies=args.cooccurrence_top_movies,
        min_support=args.cooccurrence_min_support,
    )
    cooccurrence_edges.to_csv(models_dir / "cooccurrence_edges.csv", index=False)

    if args.command in {"visualize", "all", "train"}:
        plot_scree(artifacts.singular_values, figures_dir / "scree_plot.png")
        plot_genre_clustermap(movies, figures_dir / "genre_corr_clustermap.png")
        box_sample = ratings.sample(min(len(ratings), args.plot_rating_sample), random_state=42)
        plot_rating_box_by_genre(box_sample, movies, figures_dir / "rating_boxplot_by_genre.png")

        emb_2d = compute_embedding_frame(
            item_factors=artifacts.item_factors,
            movie_ids=matrix_bundle.movie_ids,
            movies=movies,
            popularity=popularity,
            method="tsne",
            n_components=2,
            max_points=args.embedding_points,
            perplexity=args.tsne_perplexity,
        )
        emb_2d.to_csv(embeddings_dir / "movie_embedding_2d.csv", index=False)
        plot_tsne_2d(emb_2d, figures_dir / "movie_embedding_tsne_2d.png")

        emb_3d = compute_embedding_frame(
            item_factors=artifacts.item_factors,
            movie_ids=matrix_bundle.movie_ids,
            movies=movies,
            popularity=popularity,
            method="pca",
            n_components=3,
            max_points=args.embedding_points,
            perplexity=args.tsne_perplexity,
        )
        emb_3d.to_csv(embeddings_dir / "movie_embedding_3d.csv", index=False)
        build_plotly_movie_embedding_3d(emb_3d, embeddings_dir / "movie_embedding_3d.html")

        plot_similarity_network_matplotlib(
            cooccurrence_edges,
            movies,
            figures_dir / "movie_similarity_network.png",
            max_nodes=120,
        )
        build_plotly_liked_also_liked_network(cooccurrence_edges, movies, embeddings_dir / "liked_also_liked_network.html")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recommendation engine pipeline")
    parser.add_argument("command", choices=["profile", "train", "visualize", "all"], help="Pipeline stage")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-ratings", type=int, default=None)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--max-movies", type=int, default=None)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--min-user-ratings", type=int, default=5)
    parser.add_argument("--like-threshold", type=float, default=4.0)
    parser.add_argument("--svd-factors", type=int, default=100)
    parser.add_argument("--svd-iterations", type=int, default=10)
    parser.add_argument("--factor-grid", type=str, default="10,50,100,200")
    parser.add_argument("--run-tuning", action="store_true")
    parser.add_argument("--tuning-sample", type=int, default=150000)
    parser.add_argument("--eval-pairs", type=int, default=200000)
    parser.add_argument("--eval-users", type=int, default=500)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--journey-neighbors", type=int, default=20)
    parser.add_argument("--journey-min-similarity", type=float, default=0.25)
    parser.add_argument("--cooccurrence-top-movies", type=int, default=1200)
    parser.add_argument("--cooccurrence-min-support", type=int, default=20)
    parser.add_argument("--plot-rating-sample", type=int, default=500000)
    parser.add_argument("--embedding-points", type=int, default=3000)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    return parser


if __name__ == "__main__":
    # Ensure project root is on sys.path when the script is run directly
    # (e.g. `python src/pipeline.py`) rather than as a module (`python -m src.pipeline`).
    import sys as _sys
    from pathlib import Path as _Path
    _root = str(_Path(__file__).resolve().parents[1])
    if _root not in _sys.path:
        _sys.path.insert(0, _root)

    parser = build_arg_parser()
    run_pipeline(parser.parse_args())

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.data.preprocessing import build_movie_genre_matrix


def plot_scree(singular_values: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    explained = (singular_values**2) / np.sum(singular_values**2)
    cumulative = np.cumsum(explained)
    x = np.arange(1, len(singular_values) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(x, explained, marker="o", linewidth=2, label="Explained Variance Ratio")
    ax1.set_xlabel("Latent Factor")
    ax1.set_ylabel("Explained Variance Ratio")
    ax2 = ax1.twinx()
    ax2.plot(x, cumulative, color="tab:orange", linewidth=2, label="Cumulative Variance")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax1.set_title("Scree Plot")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_genre_clustermap(movies: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    genre_matrix = build_movie_genre_matrix(movies)
    corr = genre_matrix.corr().fillna(0.0)
    g = sns.clustermap(corr, cmap="vlag", center=0.0, figsize=(12, 12))
    g.fig.suptitle("Genre Correlation Clustermap", y=1.02)
    g.savefig(output_path, dpi=180)
    plt.close(g.fig)


def plot_rating_box_by_genre(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    output_path: str | Path,
    top_n_genres: int = 12,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged = ratings.merge(movies[["movieId", "genres"]], on="movieId", how="left")
    exploded = merged.assign(genre=merged["genres"].fillna("(no genres listed)").str.split("|")).explode("genre")
    top = exploded["genre"].value_counts().head(top_n_genres).index
    subset = exploded[exploded["genre"].isin(top)]
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=subset, x="genre", y="rating", hue="genre", ax=ax, palette="Set2", legend=False)
    ax.set_title("Rating Distribution by Genre")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Rating")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def compute_embedding_frame(
    item_factors: np.ndarray,
    movie_ids: np.ndarray,
    movies: pd.DataFrame,
    popularity: pd.DataFrame | None = None,
    method: str = "tsne",
    n_components: int = 2,
    max_points: int = 3000,
    perplexity: float = 30.0,
) -> pd.DataFrame:
    df = movies[["movieId", "title", "primary_genre", "year"]].copy()
    df = df[df["movieId"].isin(movie_ids)].copy()
    if popularity is not None:
        df = df.merge(popularity, on="movieId", how="left")
    else:
        df["rating_count"] = 1
    df["rating_count"] = df["rating_count"].fillna(0).astype("int32")
    df = df.sort_values("rating_count", ascending=False).head(max_points).reset_index(drop=True)
    index_lookup = {int(m): i for i, m in enumerate(movie_ids)}
    vec_idx = df["movieId"].map(index_lookup).to_numpy(np.int32)
    x = item_factors[vec_idx]

    if method.lower() == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
        emb = reducer.fit_transform(x)
    else:
        if x.shape[1] > 50:
            x = PCA(n_components=50, random_state=42).fit_transform(x)
        reducer = TSNE(
            n_components=n_components,
            perplexity=min(perplexity, max(5.0, (len(df) - 1) / 3)),
            learning_rate="auto",
            init="pca",
            random_state=42,
        )
        emb = reducer.fit_transform(x)

    for i in range(n_components):
        df[f"dim_{i+1}"] = emb[:, i]
    return df


def plot_tsne_2d(embedding_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=embedding_df,
        x="dim_1",
        y="dim_2",
        hue="primary_genre",
        size="rating_count",
        sizes=(15, 120),
        alpha=0.75,
        legend=False,
        ax=ax,
    )
    ax.set_title("Movie Embeddings in 2D")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_similarity_network_matplotlib(
    edges: pd.DataFrame,
    movies: pd.DataFrame,
    output_path: str | Path,
    max_nodes: int = 120,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if edges.empty:
        return
    pruned = edges.nlargest(max_nodes * 4, "similarity").copy()
    node_scores = pd.concat(
        [
            pruned[["source_movie_id", "similarity"]].rename(columns={"source_movie_id": "movieId"}),
            pruned[["target_movie_id", "similarity"]].rename(columns={"target_movie_id": "movieId"}),
        ]
    ).groupby("movieId")["similarity"].sum().sort_values(ascending=False)
    keep_nodes = set(node_scores.head(max_nodes).index.tolist())
    pruned = pruned[
        pruned["source_movie_id"].isin(keep_nodes) & pruned["target_movie_id"].isin(keep_nodes)
    ].copy()

    g = nx.Graph()
    for row in pruned.itertuples(index=False):
        g.add_edge(int(row.source_movie_id), int(row.target_movie_id), weight=float(row.similarity))
    title = movies.set_index("movieId")["title"].to_dict()
    labels = {n: title.get(n, str(n))[:24] for n in g.nodes}
    pos = nx.spring_layout(g, seed=42, k=0.35)
    fig, ax = plt.subplots(figsize=(16, 10))
    weights = [g[u][v]["weight"] for u, v in g.edges]
    nx.draw_networkx_edges(g, pos, alpha=0.35, width=[w * 2 for w in weights], ax=ax)
    nx.draw_networkx_nodes(g, pos, node_size=70, alpha=0.85, ax=ax)
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=7, ax=ax)
    ax.set_title('Movie Similarity Clusters ("Users Who Liked X Also Liked Y")')
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

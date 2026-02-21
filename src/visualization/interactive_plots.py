from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ─── Vibrant colour palette for genres ────────────────────────────────────────
_GENRE_PALETTE = [
    "#667eea", "#764ba2", "#f093fb", "#f5576c", "#4facfe",
    "#00f2fe", "#43e97b", "#38f9d7", "#fa709a", "#fee140",
    "#a18cd1", "#fbc2eb", "#84fab0", "#8fd3f4", "#ffecd2",
    "#fcb69f", "#a1c4fd", "#c2e9fb", "#d4fc79", "#96e6a1",
]


def _genre_color_map(genres: list[str]) -> dict[str, str]:
    return {g: _GENRE_PALETTE[i % len(_GENRE_PALETTE)] for i, g in enumerate(sorted(set(genres)))}


# ─── 3D Embedding plot ────────────────────────────────────────────────────────
def build_plotly_movie_embedding_3d(
    embedding_df: pd.DataFrame,
    output_html: str | Path | None = None,
) -> go.Figure:
    fig = px.scatter_3d(
        embedding_df,
        x="dim_1",
        y="dim_2",
        z="dim_3",
        color="primary_genre",
        size="rating_count",
        size_max=18,
        hover_data=["title", "year", "rating_count"],
        template="plotly_dark",
        opacity=0.85,
        color_discrete_sequence=_GENRE_PALETTE,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(backgroundcolor="rgba(10,8,30,0.8)", gridcolor="#2a2a4a", showbackground=True),
            yaxis=dict(backgroundcolor="rgba(10,8,30,0.8)", gridcolor="#2a2a4a", showbackground=True),
            zaxis=dict(backgroundcolor="rgba(10,8,30,0.8)", gridcolor="#2a2a4a", showbackground=True),
        ),
        legend=dict(
            title_text="Genre",
            bgcolor="rgba(15,12,41,0.75)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            font=dict(color="#ccd6f6"),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(color="#ccd6f6"),
        title=dict(text="3D Movie Embedding Explorer", font=dict(color="#ccd6f6", size=16)),
        height=550,
    )
    if output_html is not None:
        out = Path(output_html)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out, include_plotlyjs="cdn")
    return fig


# ─── Legacy network (kept for compatibility) ──────────────────────────────────
def build_plotly_liked_also_liked_network(
    edges: pd.DataFrame,
    movies: pd.DataFrame,
    output_html: str | Path | None = None,
    max_nodes: int = 200,
) -> go.Figure:
    return build_plotly_preference_network(edges, movies, output_html=output_html, max_nodes=max_nodes)


# ─── Preference / similarity network ─────────────────────────────────────────
def build_plotly_preference_network(
    edges: pd.DataFrame,
    movies: pd.DataFrame,
    output_html: str | Path | None = None,
    max_nodes: int = 160,
    dark_mode: bool = True,
) -> go.Figure:
    """
    Build an interactive movie-similarity network using a circular layout
    grouped by genre, with colour-coded nodes and weighted edges.
    Falls back gracefully to an empty figure if no data is available.
    """
    if edges.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No network data available",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    sim_col = "similarity" if "similarity" in edges.columns else "weight"

    # ── Select top-N nodes by total similarity score ──────────────────────────
    node_weight = pd.concat(
        [
            edges[["source_movie_id", sim_col]].rename(columns={"source_movie_id": "movieId"}),
            edges[["target_movie_id", sim_col]].rename(columns={"target_movie_id": "movieId"}),
        ]
    ).groupby("movieId")[sim_col].sum().sort_values(ascending=False)
    keep = set(node_weight.head(max_nodes).index.tolist())

    e = edges[edges["source_movie_id"].isin(keep) & edges["target_movie_id"].isin(keep)].copy()
    if e.empty:
        fig = go.Figure()
        return fig

    # ── Build node table with metadata ───────────────────────────────────────
    movie_meta = movies.set_index("movieId")[["title", "primary_genre"]].to_dict("index")
    node_ids = sorted(keep)
    n = len(node_ids)

    node_df = pd.DataFrame({"movieId": node_ids})
    node_df["title"] = node_df["movieId"].map(lambda mid: movie_meta.get(mid, {}).get("title", str(mid)))
    node_df["genre"] = node_df["movieId"].map(lambda mid: movie_meta.get(mid, {}).get("primary_genre", "Other"))
    node_df["score"] = node_df["movieId"].map(lambda mid: float(node_weight.get(mid, 0.0)))

    # ── Circular layout grouped by genre ─────────────────────────────────────
    genres_sorted = node_df.groupby("genre").size().sort_values(ascending=False).index.tolist()
    genre_offset: dict[str, float] = {}
    cumulative_angle = 0.0
    total = n
    for g in genres_sorted:
        cnt = (node_df["genre"] == g).sum()
        genre_offset[g] = cumulative_angle
        cumulative_angle += (cnt / total) * 2 * math.pi

    xs_node: list[float] = []
    ys_node: list[float] = []
    genre_counts_so_far: dict[str, int] = {g: 0 for g in genres_sorted}
    for _, row in node_df.iterrows():
        g = row["genre"]
        cnt_g = (node_df["genre"] == g).sum()
        i = genre_counts_so_far[g]
        base_angle = genre_offset[g]
        angle = base_angle + (i / max(cnt_g, 1)) * (cnt_g / total) * 2 * math.pi
        # Slightly jitter radius so overlapping nodes spread
        r = 1.0 + 0.15 * math.sin(i * 0.7)
        xs_node.append(r * math.cos(angle))
        ys_node.append(r * math.sin(angle))
        genre_counts_so_far[g] += 1

    node_df["x"] = xs_node
    node_df["y"] = ys_node
    idx_map = {mid: i for i, mid in enumerate(node_ids)}

    # ── Colour map ────────────────────────────────────────────────────────────
    color_map = _genre_color_map(node_df["genre"].tolist())

    # ── Edge traces (grouped by similarity quartile for alpha variation) ──────
    sim_vals = e[sim_col].values
    if len(sim_vals) > 0:
        q33, q66 = np.percentile(sim_vals, [33, 66])
    else:
        q33, q66 = 0.33, 0.66

    edge_groups: dict[str, tuple[list, list, str]] = {
        "strong": ([], [], "rgba(167,139,250,0.55)"),
        "medium": ([], [], "rgba(102,126,234,0.30)"),
        "weak":   ([], [], "rgba(100,100,150,0.15)"),
    }
    for row in e.itertuples(index=False):
        si = idx_map.get(int(row.source_movie_id))
        ti = idx_map.get(int(row.target_movie_id))
        if si is None or ti is None:
            continue
        s_val = float(getattr(row, sim_col))
        key = "strong" if s_val >= q66 else ("medium" if s_val >= q33 else "weak")
        edge_groups[key][0].extend([node_df.loc[si, "x"], node_df.loc[ti, "x"], None])
        edge_groups[key][1].extend([node_df.loc[si, "y"], node_df.loc[ti, "y"], None])

    traces: list[go.BaseTraceType] = []
    widths = {"strong": 1.5, "medium": 0.8, "weak": 0.4}
    for key, (xs, ys, color) in edge_groups.items():
        if xs:
            traces.append(
                go.Scatter(
                    x=xs, y=ys,
                    mode="lines",
                    line=dict(width=widths[key], color=color),
                    hoverinfo="none",
                    showlegend=False,
                    name=f"{key} edges",
                )
            )

    # ── Node traces per genre (so we get a legend) ────────────────────────────
    for genre in genres_sorted:
        g_df = node_df[node_df["genre"] == genre].reset_index(drop=True)
        sizes = 8 + (g_df["score"] / (node_df["score"].max() + 1e-9) * 14).clip(0, 14)
        traces.append(
            go.Scatter(
                x=g_df["x"],
                y=g_df["y"],
                mode="markers",
                name=genre,
                text=g_df["title"],
                customdata=g_df["score"].values,
                hovertemplate="<b>%{text}</b><br>Score: %{customdata:.2f}<extra></extra>",
                marker=dict(
                    size=sizes,
                    color=color_map[genre],
                    line=dict(width=0.8, color="rgba(255,255,255,0.25)"),
                    opacity=0.9,
                ),
                showlegend=True,
            )
        )

    bg = "rgba(10,8,30,1)" if dark_mode else "white"
    font_color = "#ccd6f6" if dark_mode else "#111"
    legend_bg = "rgba(15,12,41,0.85)" if dark_mode else "rgba(255,255,255,0.85)"

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text="🌐 Movie Preference Network",
            font=dict(size=16, color=font_color),
            x=0.02,
        ),
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        xaxis=dict(visible=False, range=[-1.5, 1.5]),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1, range=[-1.5, 1.5]),
        showlegend=True,
        legend=dict(
            title_text="Genre",
            bgcolor=legend_bg,
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            font=dict(color=font_color, size=11),
            orientation="v",
            x=1.01,
            y=0.98,
        ),
        margin=dict(l=10, r=140, t=50, b=10),
        font=dict(color=font_color),
        height=600,
        hovermode="closest",
    )
    if output_html is not None:
        out = Path(output_html)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out, include_plotlyjs="cdn")
    return fig

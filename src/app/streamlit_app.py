from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
from pathlib import Path
import sys
import json

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Local imports ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.journey import find_shortest_journey
from src.visualization.interactive_plots import (
    build_plotly_liked_also_liked_network,
    build_plotly_movie_embedding_3d,
    build_plotly_preference_network,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED  = ROOT / "data" / "processed"
MODELS     = ROOT / "outputs" / "models"
EMBEDDINGS = ROOT / "outputs" / "embeddings"


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Recommendation Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — light glassmorphism
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 15px;
    color: #1a1a2e;
}

/* ── Animated pastel gradient background ── */
.stApp {
    background: linear-gradient(135deg, #e8f4fd 0%, #f0eaff 35%, #e8f8f5 65%, #fdf0f8 100%);
    background-size: 400% 400%;
    animation: gradientShift 18s ease infinite;
    min-height: 100vh;
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.55) !important;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-right: 1px solid rgba(255, 255, 255, 0.4);
    box-shadow: 2px 0 20px rgba(0, 0, 0, 0.06);
}
[data-testid="stSidebar"] * { color: #1a1a2e !important; }
[data-testid="stSidebar"] hr { border-color: rgba(0,0,0,0.08); }

/* ── Glass card ── */
.glass-card {
    background: rgba(255, 255, 255, 0.55);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.45);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
    animation: fadeSlideUp 0.45s ease both;
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Metric tile ── */
.metric-tile {
    background: rgba(255, 255, 255, 0.60);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.5);
    border-radius: 14px;
    padding: 1.1rem 1rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.07);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    animation: fadeSlideUp 0.5s ease both;
}
.metric-tile:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.10);
}
.metric-value {
    font-size: 1.9rem;
    font-weight: 600;
    color: #5b5bd6;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 500;
    color: #6e7191;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 0.25rem;
}

/* ── Hero ── */
.hero-title {
    font-size: 2.6rem;
    font-weight: 300;
    color: #1a1a2e;
    letter-spacing: -0.02em;
    line-height: 1.2;
    margin-bottom: 0.4rem;
}
.hero-sub {
    font-size: 1rem;
    color: #6e7191;
    font-weight: 400;
    line-height: 1.6;
    max-width: 540px;
}

/* ── Section label ── */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8b8ba7;
    margin-bottom: 0.6rem;
}

/* ── Section heading ── */
.section-heading {
    font-size: 1.15rem;
    font-weight: 500;
    color: #1a1a2e;
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(0,0,0,0.07);
}

/* ── Recommendation item card ── */
.rec-card {
    background: rgba(255, 255, 255, 0.60);
    border: 1px solid rgba(255, 255, 255, 0.5);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 3px 14px rgba(0, 0, 0, 0.06);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    animation: fadeSlideUp 0.4s ease both;
}
.rec-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.10);
}
.rec-title {
    font-size: 0.95rem;
    font-weight: 500;
    color: #1a1a2e;
    line-height: 1.4;
}
.rec-meta {
    font-size: 0.78rem;
    color: #8b8ba7;
    margin-top: 0.2rem;
}
.genre-tag {
    display: inline-block;
    background: rgba(91, 91, 214, 0.10);
    color: #5b5bd6;
    border: 1px solid rgba(91, 91, 214, 0.20);
    border-radius: 6px;
    padding: 1px 8px;
    font-size: 0.72rem;
    font-weight: 500;
    margin-left: 0.4rem;
    vertical-align: middle;
}

/* ── Similarity bar ── */
.sim-track {
    background: rgba(0,0,0,0.07);
    border-radius: 99px;
    height: 5px;
    margin-top: 0.5rem;
    overflow: hidden;
}
.sim-fill {
    height: 5px;
    border-radius: 99px;
    background: linear-gradient(90deg, #a78bfa, #5b5bd6);
}
.sim-value {
    font-size: 0.72rem;
    color: #8b8ba7;
    margin-top: 0.25rem;
}

/* ── Badge ── */
.badge { border-radius: 6px; padding: 2px 8px; font-size: 0.72rem; font-weight: 500; }
.badge-high   { background: #dcfce7; color: #16a34a; border: 1px solid #bbf7d0; }
.badge-medium { background: #fef9c3; color: #a16207; border: 1px solid #fde68a; }
.badge-low    { background: #fee2e2; color: #dc2626; border: 1px solid #fecaca; }

/* ── Journey path ── */
.journey-path {
    background: rgba(91, 91, 214, 0.06);
    border: 1px solid rgba(91, 91, 214, 0.18);
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    font-size: 0.92rem;
    color: #1a1a2e;
    line-height: 1.7;
    word-break: break-word;
}
.journey-arrow { color: #a78bfa; font-weight: 600; margin: 0 0.3rem; }

/* ── Step card ── */
.step-card {
    background: rgba(255,255,255,0.65);
    border: 1px solid rgba(255,255,255,0.5);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    text-align: center;
    font-size: 0.78rem;
    color: #1a1a2e;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    line-height: 1.4;
}

/* ── Source pill ── */
.source-pill {
    display: inline-block;
    background: rgba(167,139,250,0.12);
    color: #7c6fcd;
    border: 1px solid rgba(167,139,250,0.25);
    border-radius: 99px;
    padding: 2px 12px;
    font-size: 0.75rem;
    font-weight: 500;
    margin-bottom: 1rem;
}

/* ── Primary button override ── */
.stButton > button {
    background: #5b5bd6;
    color: #fff;
    border: none;
    border-radius: 10px;
    font-weight: 500;
    font-size: 0.9rem;
    padding: 0.5rem 1.5rem;
    transition: opacity 0.2s, transform 0.15s;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

/* ── Streamlit label text ── */
label, .stSelectbox label, .stSlider label, .stMultiSelect label {
    color: #6e7191 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}

/* ── Nav radio ── */
[data-testid="stSidebar"] .stRadio > div { gap: 0.1rem; }
[data-testid="stSidebar"] .stRadio label {
    padding: 0.45rem 0.75rem !important;
    border-radius: 8px;
    transition: background 0.15s;
    font-size: 0.88rem !important;
}
[data-testid="stSidebar"] .stRadio label:hover { background: rgba(91,91,214,0.08); }

/* ── Plotly container ── */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }

/* ── Hide Streamlit branding ── */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_movies() -> pd.DataFrame:
    path = PROCESSED / "movies_enriched.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["movieId", "title", "primary_genre"])
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_journey_edges() -> pd.DataFrame:
    path = MODELS / "journey_edges.csv"
    if not path.exists():
        return pd.DataFrame(columns=["source_movie_id", "target_movie_id", "similarity", "weight"])
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_cooccurrence_edges() -> pd.DataFrame:
    path = MODELS / "cooccurrence_edges.csv"
    if not path.exists():
        return pd.DataFrame(columns=["source_movie_id", "target_movie_id", "support", "similarity", "weight"])
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_item_factors() -> tuple[np.ndarray, np.ndarray]:
    path = MODELS / "svd_artifacts.npz"
    if not path.exists():
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int32)
    z = np.load(path)
    return z["item_factors"].astype(np.float32), z["movie_ids"].astype(np.int32)


@st.cache_data(show_spinner=False)
def load_embedding_3d() -> pd.DataFrame:
    path = EMBEDDINGS / "movie_embedding_3d.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_evaluation() -> dict:
    path = MODELS / "evaluation_summary.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def similar_movies(
    source_movie_id: int,
    item_factors: np.ndarray,
    movie_ids: np.ndarray,
    movies: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    idx_map = {int(mid): i for i, mid in enumerate(movie_ids)}
    if source_movie_id not in idx_map:
        return pd.DataFrame(columns=["movieId", "title", "similarity"])
    idx = idx_map[source_movie_id]
    source = item_factors[idx].astype(np.float32)
    dots = item_factors @ source
    source_norm = float(np.linalg.norm(source))
    matrix_norms = np.linalg.norm(item_factors, axis=1)
    denom = source_norm * matrix_norms
    sims = np.divide(
        dots, denom,
        out=np.zeros_like(dots, dtype=np.float32),
        where=denom > 0,
    ).astype(np.float32)
    sims[idx] = -1
    top = np.argpartition(sims, -top_n)[-top_n:]
    top = top[np.argsort(sims[top])[::-1]]
    out = pd.DataFrame({"movieId": movie_ids[top], "similarity": sims[top]})
    return out.merge(movies[["movieId", "title", "primary_genre"]], on="movieId", how="left")


def liked_also_liked_from_edges(
    source_movie_id: int,
    edges: pd.DataFrame,
    movies: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame(columns=["movieId", "title", "support", "similarity", "confidence"])
    has_support = "support" in edges.columns
    pick_cols_tgt = ["target_movie_id", "support", "similarity"] if has_support else ["target_movie_id", "similarity"]
    pick_cols_src = ["source_movie_id", "support", "similarity"] if has_support else ["source_movie_id", "similarity"]
    left  = edges[edges["source_movie_id"] == source_movie_id][pick_cols_tgt].rename(columns={"target_movie_id": "movieId"})
    right = edges[edges["target_movie_id"] == source_movie_id][pick_cols_src].rename(columns={"source_movie_id": "movieId"})
    out = pd.concat([left, right], ignore_index=True)
    if out.empty:
        return pd.DataFrame(columns=["movieId", "title", "support", "similarity", "confidence"])
    sort_cols = ["support", "similarity"] if has_support else ["similarity"]
    # Deduplicate: a movie may appear via both forward and reverse edges — keep best score
    out = out.sort_values(sort_cols, ascending=False).drop_duplicates(subset=["movieId"]).head(top_n)
    out = out.merge(movies[["movieId", "title", "primary_genre"]], on="movieId", how="left")
    if has_support:
        out["confidence"] = np.select(
            [out["support"] >= 100, out["support"] >= 40],
            ["high", "medium"],
            default="low",
        )
    return out


def sim_bar_html(value: float) -> str:
    pct = int(min(max(value, 0.0), 1.0) * 100)
    return (
        f'<div class="sim-track"><div class="sim-fill" style="width:{pct}%"></div></div>'
        f'<div class="sim-value">{value:.3f}</div>'
    )


def badge_html(conf: str) -> str:
    cls = {"high": "badge-high", "medium": "badge-medium"}.get(conf.lower(), "badge-low")
    return f'<span class="badge {cls}">{conf}</span>'


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div style="padding: 0.5rem 0 0.25rem;"><span style="font-size:1.05rem;font-weight:600;color:#1a1a2e;">Recommendation Explorer</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.78rem;color:#8b8ba7;margin-bottom:1.2rem;line-height:1.5;">SVD-powered collaborative filtering for movies</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="section-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "",
        ["Overview", "Movie Recommender", "Movie Journey", "Preference Network", "3D Explorer"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown('<div class="section-label">Settings</div>', unsafe_allow_html=True)
    top_n            = st.slider("Top-N recommendations", 5, 20, 10)
    max_network_nodes = st.slider("Max network nodes", 40, 200, 120)

    st.divider()
    st.markdown('<div style="font-size:0.72rem;color:#8b8ba7;line-height:1.6;">Built with SVD · t-SNE · Network Analysis</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
movies             = load_movies()
journey_edges      = load_journey_edges()
cooccurrence_edges = load_cooccurrence_edges()
item_factors, movie_ids = load_item_factors()
embedding_3d       = load_embedding_3d()
evaluation         = load_evaluation()

ready        = not movies.empty and item_factors.size > 0
network_edges = cooccurrence_edges if not cooccurrence_edges.empty else journey_edges

if not ready:
    st.markdown(
        '<div class="glass-card" style="text-align:center;padding:3rem;">'
        '<div style="font-size:1.15rem;font-weight:500;color:#dc2626;margin-bottom:0.75rem;">Artifacts not found</div>'
        '<code style="background:rgba(0,0,0,0.06);padding:0.4rem 1rem;border-radius:8px;font-size:0.85rem;">'
        'python -m src.pipeline all</code></div>',
        unsafe_allow_html=True,
    )
    st.stop()

options      = movies.sort_values("title")[["movieId", "title"]]
movie_lookup = {f"{row.title} [{int(row.movieId)}]": int(row.movieId) for row in options.itertuples(index=False)}
movie_labels = list(movie_lookup.keys())


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":

    # Hero
    st.markdown(
        '<div class="hero-title">Recommendation Engine</div>'
        '<div class="hero-sub">Explore movie similarities, latent factor embeddings, and collaborative filtering insights.</div>'
        '<br>',
        unsafe_allow_html=True,
    )

    # Metric tiles
    n_movies  = len(movies)
    n_genres  = movies["primary_genre"].nunique() if "primary_genre" in movies.columns else 0
    n_journey = len(journey_edges)
    n_net     = len(network_edges)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, f"{n_movies:,}", "Movies"),
        (c2, f"{n_genres}", "Genres"),
        (c3, f"{n_journey:,}", "Journey Edges"),
        (c4, f"{n_net:,}", "Network Edges"),
    ]:
        col.markdown(
            f'<div class="metric-tile"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Model performance
    if evaluation:
        pred = evaluation.get("prediction_metrics", {})
        rank = evaluation.get("ranking_metrics", {})
        cov  = evaluation.get("coverage", 0)
        div_ = evaluation.get("diversity", 0)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Model Performance</div>', unsafe_allow_html=True)
        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
        for col, val, label in [
            (mc1, f"{pred.get('rmse', 0):.3f}", "RMSE"),
            (mc2, f"{pred.get('mae', 0):.3f}", "MAE"),
            (mc3, f"{rank.get('precision_at_k', 0):.3f}", "Precision@K"),
            (mc4, f"{rank.get('recall_at_k', 0):.3f}", "Recall@K"),
            (mc5, f"{cov:.1%}", "Coverage"),
            (mc6, f"{div_:.3f}", "Diversity"),
        ]:
            col.metric(label, val)
        st.markdown('</div>', unsafe_allow_html=True)

    # Genre distribution
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Genre Distribution</div>', unsafe_allow_html=True)
    genre_counts = movies["primary_genre"].fillna("Unknown").value_counts().reset_index()
    genre_counts.columns = ["Genre", "Count"]
    fig_genre = px.bar(
        genre_counts, x="Count", y="Genre", orientation="h",
        color="Count",
        color_continuous_scale=[[0, "#e0e7ff"], [1, "#5b5bd6"]],
        template="none",
    )
    fig_genre.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=4, b=0),
        font=dict(family="Inter, system-ui", color="#1a1a2e", size=12),
        yaxis=dict(categoryorder="total ascending", gridcolor="rgba(0,0,0,0.04)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0.04)"),
        height=400,
    )
    fig_genre.update_traces(marker_line_width=0)
    st.plotly_chart(fig_genre, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — MOVIE RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Movie Recommender":

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Movie Recommender</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.85rem;color:#6e7191;margin-bottom:1rem;line-height:1.6;">Select a movie to discover similar titles based on collaborative filtering signals.</div>', unsafe_allow_html=True)

    selected_label = st.selectbox("Select a movie", movie_labels, index=0, label_visibility="visible")
    selected_id    = movie_lookup[selected_label]

    # Source priority: cooccurrence -> journey -> SVD
    edge_df = liked_also_liked_from_edges(selected_id, cooccurrence_edges, movies, top_n=top_n)
    if edge_df.empty:
        edge_df = liked_also_liked_from_edges(selected_id, journey_edges, movies, top_n=top_n)

    if edge_df.empty:
        edge_df      = similar_movies(selected_id, item_factors, movie_ids, movies, top_n=top_n)
        source_label = "SVD Cosine Similarity"
        show_conf    = False
    elif "support" in edge_df.columns:
        source_label = "Co-occurrence Network"
        show_conf    = True
    else:
        source_label = "Journey Graph Similarity"
        show_conf    = False

    st.markdown(f'<div class="source-pill">Source: {source_label}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Recommendation cards
    if not edge_df.empty:
        sims = edge_df["similarity"].values if "similarity" in edge_df.columns else np.ones(len(edge_df))
        for i, (_, row) in enumerate(edge_df.iterrows()):
            title  = row.get("title", "Unknown")
            genre  = row.get("primary_genre", "")
            sim_v  = float(sims[i]) if i < len(sims) else 0.5
            conf   = str(row.get("confidence", "")) if show_conf else ""

            genre_html = f'<span class="genre-tag">{genre}</span>' if genre else ""
            badge_str  = badge_html(conf) if conf else ""

            st.markdown(
                f'<div class="rec-card">'
                f'  <div style="display:flex;justify-content:space-between;align-items:flex-start;">'
                f'    <div><span class="rec-title">{i+1}. {title}</span>{genre_html}</div>'
                f'    <div>{badge_str}</div>'
                f'  </div>'
                f'  {sim_bar_html(sim_v)}'
                f'</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — MOVIE JOURNEY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Movie Journey":

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Movie Journey</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.85rem;color:#6e7191;margin-bottom:1.2rem;line-height:1.6;">Find a chain of related movies connecting two titles through the similarity graph.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    start_label = c1.selectbox("Start movie", movie_labels, index=0, key="start")
    end_label   = c2.selectbox("End movie", movie_labels, index=min(5, len(movie_labels) - 1), key="end")
    start_id    = movie_lookup[start_label]
    end_id      = movie_lookup[end_label]

    find_btn = st.button("Find Journey", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if find_btn:
        with st.spinner("Computing shortest path..."):
            path, dist = find_shortest_journey(journey_edges, start_id, end_id)

        if path:
            title_map  = movies.set_index("movieId")["title"].to_dict()
            path_parts = [
                f'<span style="color:#5b5bd6;font-weight:500;">{title_map.get(mid, str(mid))}</span>'
                for mid in path
            ]
            path_html = '<span class="journey-arrow">→</span>'.join(path_parts)

            st.markdown(
                f'<div class="glass-card">'
                f'  <div class="journey-path">{path_html}</div>'
                f'  <div style="margin-top:0.6rem;font-size:0.8rem;color:#8b8ba7;">'
                f'    Path length: <span style="color:#5b5bd6;font-weight:500;">{len(path)}</span> movies'
                f'    &nbsp;&middot;&nbsp; Cost: <span style="color:#5b5bd6;font-weight:500;">{dist:.4f}</span>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Step cards
            if len(path) >= 2:
                cols = st.columns(min(len(path), 6))
                for i, mid in enumerate(path[:6]):
                    label = "Start" if i == 0 else ("End" if i == len(path) - 1 else f"Step {i}")
                    cols[i].markdown(
                        f'<div class="step-card">'
                        f'  <div style="font-size:0.68rem;color:#8b8ba7;margin-bottom:0.25rem;">{label}</div>'
                        f'  {title_map.get(mid, str(mid))[:30]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.warning("No valid path found. Try a different pair of movies.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — PREFERENCE NETWORK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Preference Network":

    source_note = "co-occurrence data" if not cooccurrence_edges.empty else "journey graph (co-occurrence data unavailable)"
    st.markdown(
        f'<div class="glass-card">'
        f'  <div class="section-heading">Preference Network</div>'
        f'  <div style="font-size:0.85rem;color:#6e7191;line-height:1.6;">'
        f'    Movie similarity network built from <span style="color:#5b5bd6;font-weight:500;">{source_note}</span>.'
        f'    Nodes are movies; edges connect similar titles grouped by genre.'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if network_edges.empty:
        st.warning("No network data available. Run the pipeline to generate edges.")
    else:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        with st.spinner("Building network..."):
            net_fig = build_plotly_preference_network(
                network_edges, movies,
                max_nodes=max_network_nodes,
                dark_mode=False,
            )
        st.plotly_chart(net_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — 3D EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "3D Explorer":

    st.markdown(
        '<div class="glass-card">'
        '  <div class="section-heading">3D Embedding Explorer</div>'
        '  <div style="font-size:0.85rem;color:#6e7191;line-height:1.6;">'
        '    SVD latent factors projected to 3D via PCA. Each node is a movie — hover to explore.'
        '    Clusters reveal latent genre similarity.'
        '  </div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if not embedding_3d.empty:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        genres          = sorted(embedding_3d["primary_genre"].dropna().unique().tolist())
        selected_genres = st.multiselect("Filter genres", genres, default=genres[:8] if len(genres) > 8 else genres)
        filtered_emb    = embedding_3d[embedding_3d["primary_genre"].isin(selected_genres)] if selected_genres else embedding_3d

        fig3d = px.scatter_3d(
            filtered_emb,
            x="dim_1", y="dim_2", z="dim_3",
            color="primary_genre",
            size="rating_count",
            size_max=14,
            hover_data=["title", "year", "rating_count"],
            template="none",
            opacity=0.85,
            color_discrete_sequence=[
                "#5b5bd6", "#a78bfa", "#22c55e", "#f59e0b", "#ef4444",
                "#06b6d4", "#ec4899", "#84cc16", "#8b5cf6", "#f97316",
                "#0ea5e9", "#14b8a6", "#e11d48", "#d97706", "#4f46e5",
            ],
        )
        fig3d.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            scene=dict(
                bgcolor="rgba(248,250,252,0.7)",
                xaxis=dict(backgroundcolor="rgba(240,246,255,0.8)", gridcolor="rgba(91,91,214,0.12)", showbackground=True, title=""),
                yaxis=dict(backgroundcolor="rgba(240,246,255,0.8)", gridcolor="rgba(91,91,214,0.12)", showbackground=True, title=""),
                zaxis=dict(backgroundcolor="rgba(240,246,255,0.8)", gridcolor="rgba(91,91,214,0.12)", showbackground=True, title=""),
            ),
            legend=dict(
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="rgba(91,91,214,0.2)",
                borderwidth=1,
                font=dict(color="#1a1a2e", size=11, family="Inter, system-ui"),
                title_text="Genre",
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            font=dict(color="#1a1a2e", family="Inter, system-ui"),
            height=560,
        )
        st.plotly_chart(fig3d, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("3D embedding file not found. Run the pipeline first.")

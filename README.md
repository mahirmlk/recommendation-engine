# Recommendation Engine with Visual Exploration

Collaborative filtering system for MovieLens with:
- User-based and item-based nearest-neighbor recommendations
- Matrix factorization with truncated SVD
- Evaluation metrics (RMSE, MAE, Precision@K, Recall@K, coverage, diversity)
- Static visual analysis (Seaborn and Matplotlib)
- Interactive exploration (Plotly + Streamlit)
- Movie journey pathfinding between two films in latent preference space

## Project Structure

```
recommendation-engine/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_cf.ipynb
│   ├── 04_matrix_factorization.ipynb
│   ├── 05_visualization.ipynb
│   └── 06_interactive_dashboard.ipynb
├── src/
│   ├── data/
│   ├── models/
│   ├── visualization/
│   └── app/
├── outputs/
│   ├── figures/
│   ├── embeddings/
│   └── models/
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Full Pipeline

```bash
python -m src.pipeline all --data-dir data/raw --output-dir outputs --processed-dir data/processed
```

Optional performance controls for very large ratings files:

```bash
python -m src.pipeline all --data-dir data/raw --max-ratings 5000000 --max-users 30000 --max-movies 10000
```

## Run Streamlit App

```bash
streamlit run src/app/streamlit_app.py
```

## Outputs

- `data/processed/`: cleaned and split tables
- `outputs/models/`: SVD artifacts, evaluation, journey graph edges
- `outputs/figures/`: scree plot, clustermap, boxplots, t-SNE, network graph
- `outputs/embeddings/`: 2D/3D embeddings for exploration

## Notes

- Loader supports common MovieLens schemas: `ratings.csv/movies.csv`, `ratings.dat/movies.dat`, and `u.data/u.item`.
- Temporal split is user-wise, using earlier ratings for train and later ratings for test.
- Sparse matrices are used end-to-end to avoid dense memory blowups.

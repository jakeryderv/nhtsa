# nhtsa

Topic discovery in NHTSA vehicle-safety complaints: TF-IDF baselines and
sentence-embedding clusters, scored against NHTSA's component labels.

## Setup

```sh
uv sync
uv run nhtsa download   # complaint flat file into data/raw/
uv run nhtsa build-db   # curated DuckDB database into data/processed/
```

## Notebooks

Open with `uv run jupyter lab`, or re-run one in place:

```sh
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_tfidf_topics.ipynb
```

| Notebook | Contents |
|---|---|
| `01_overview` | Volume, sources, components, vehicles, severity, text style |
| `02_text_cleaning` | Checks on text cleaning and the component label mapping |
| `03_tfidf_topics` | NMF topics and LSA + k-means on TF-IDF |
| `04_embedding_clusters` | k-means and UMAP + HDBSCAN on sentence embeddings |

Notebooks 03 and 04 share a fixed 100k-complaint sample. Notebook 04 encodes it
once (a few minutes on a GPU) and caches the result in
`data/processed/embeddings/`.

## Code

`src/nhtsa/`: `download` and `load` (raw data), `text` and `labels` (cleaning,
component labels), `dataset` (modeling tables), `database` (curated DuckDB database for query
tools, described by `sources/complaints.yaml`), `topics` (TF-IDF clusterings and
cluster words), `embed` (cached embeddings), `evaluate` (scores), `notebook`
(display helpers).

```sh
uv run pytest
uv run ruff check . && uv run ty check
```

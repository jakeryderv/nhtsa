"""Sentence embeddings for complaint text, cached on disk.

Encoding the 100k sample takes minutes on a GPU, so it is done once per model
and sample. The cache records a fingerprint of the rows it was built from and
is rebuilt when the sample or its cleaned text changes.
"""

import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl

from nhtsa.dataset import sample_complaints
from nhtsa.paths import PROCESSED_DIR

EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"
# 512-token limit, which covers all but ~0.1% of complaints in full.
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"

Encoder = Callable[[list[str]], np.ndarray]


def embed_sample(
    model: str = DEFAULT_MODEL, n: int = 100_000, seed: int = 0
) -> np.ndarray:
    """Unit-length embeddings of sample_complaints(n, seed)["TEXT"], in row order."""
    sample = sample_complaints(n, seed)
    path = EMBEDDINGS_DIR / f"{model.replace('/', '__')}_{n}_{seed}.npz"
    return cached_embeddings(
        sample["ODINO"], sample["TEXT"], path, lambda texts: encode(texts, model)
    )


def cached_embeddings(
    ids: pl.Series, texts: pl.Series, path: Path, encoder: Encoder
) -> np.ndarray:
    """Embeddings for texts, read from path if it was built from these ids and texts."""
    key = fingerprint(ids, texts)
    if path.exists():
        with np.load(path) as cache:
            if str(cache["key"]) == key:
                return cache["embeddings"]
    embeddings = np.asarray(encoder(texts.to_list()), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, key=key, embeddings=embeddings)
    return embeddings


def fingerprint(ids: pl.Series, texts: pl.Series) -> str:
    """Hash of the ids and texts, in order."""
    digest = hashlib.sha256()
    for value in pl.DataFrame({"id": ids, "text": texts}).hash_rows(seed=0):
        digest.update(value.to_bytes(8, "little"))
    return digest.hexdigest()


def encode(texts: list[str], model: str = DEFAULT_MODEL) -> np.ndarray:
    """Encode texts with a sentence-transformers model (on the GPU if there is one)."""
    from sentence_transformers import SentenceTransformer  # slow import

    return SentenceTransformer(model).encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True
    )

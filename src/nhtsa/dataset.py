"""Complaint tables ready for modeling: cleaned text, flags, and labels."""

import polars as pl

from nhtsa.labels import component_labels
from nhtsa.load import load_complaints
from nhtsa.text import clean_text, is_hotline_template, is_recall_parts_unavailable


def cleaned_complaints() -> pl.DataFrame:
    """One row per complaint with TEXT (cleaned CDESCR), flags, and LABELS.

    TEXT is empty for placeholder complaints ("no summary listed").
    """
    raw = pl.col("CDESCR")
    return load_complaints().select(
        "ODINO",
        "CDESCR",
        "COMPONENTS",
        YEAR=pl.col("LDATE").dt.year(),
        TEXT=clean_text(raw),
        HOTLINE=is_hotline_template(raw),
        PARTS_UNAVAILABLE=is_recall_parts_unavailable(raw),
        LABELS=component_labels(pl.col("COMPONENTS")),
    )


def sample_complaints(n: int = 100_000, seed: int = 0) -> pl.DataFrame:
    """A fixed random sample of complaints with non-empty TEXT.

    The same n and seed give the same rows, so notebooks can share cached
    features (e.g. embeddings) for the sample.
    """
    return (
        cleaned_complaints()
        .filter(pl.col("TEXT") != "")
        .sample(n, seed=seed)
        .sort("ODINO")
    )

"""Score clusterings of complaints against component labels and era.

Cluster ids are integers; negative ids mean "unassigned" (e.g. HDBSCAN noise)
and are left out of both scores.
"""

import numpy as np
import polars as pl
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Windows on either side of NHTSA's 2021 system change.
ERA_BEFORE = (2017, 2020)
ERA_AFTER = (2022, 2025)


def single_label(labels: pl.Series) -> pl.Series:
    """Each complaint's label if it has exactly one known label, else null."""
    return labels.to_frame("labels").select(
        pl.when(
            (pl.col("labels").list.len() == 1)
            & (pl.col("labels").list.first() != "unknown")
        ).then(pl.col("labels").list.first())
    )["labels"]


def label_agreement(clusters: np.ndarray, labels: pl.Series) -> dict[str, float]:
    """NMI and ARI between clusters and component labels.

    Uses complaints with exactly one known label that were assigned a cluster;
    `coverage` is the share of complaints that qualified.
    """
    single_series = single_label(labels)
    single = single_series.to_numpy()
    mask = single_series.is_not_null().to_numpy() & (clusters >= 0)
    return {
        "nmi": float(normalized_mutual_info_score(single[mask], clusters[mask])),
        "ari": float(adjusted_rand_score(single[mask], clusters[mask])),
        "coverage": float(mask.mean()),
    }


def era_accuracy(clusters: np.ndarray, years: np.ndarray, seed: int = 0) -> float:
    """How well cluster membership predicts era (ERA_BEFORE vs ERA_AFTER).

    Balances the two eras, then predicts each cluster's majority era. 0.5 means
    clusters carry no era signal; 1.0 means each cluster is one era only.
    """
    df = pl.DataFrame({"cluster": clusters, "year": years}).filter(
        pl.col("cluster") >= 0
    )
    df = df.with_columns(
        after=pl.when(pl.col("year").is_between(*ERA_BEFORE))
        .then(False)
        .when(pl.col("year").is_between(*ERA_AFTER))
        .then(True)
    ).drop_nulls("after")
    n = df.group_by("after").len()["len"].min()
    balanced = df.filter(pl.int_range(pl.len()).shuffle(seed=seed).over("after") < n)
    per_cluster = balanced.group_by("cluster").agg(
        majority=pl.max_horizontal(pl.col("after").sum(), (~pl.col("after")).sum())
    )
    return float(per_cluster["majority"].sum() / balanced.height)


def scores(
    clusters: np.ndarray, labels: pl.Series, years: np.ndarray
) -> dict[str, float]:
    """label_agreement plus era_accuracy and the share of unassigned complaints."""
    return {
        **label_agreement(clusters, labels),
        "era_accuracy": era_accuracy(clusters, years),
        "unassigned": float((clusters < 0).mean()),
    }

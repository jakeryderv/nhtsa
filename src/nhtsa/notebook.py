"""Display helpers shared by the notebooks."""

import html

import altair as alt
import numpy as np
import polars as pl
from IPython.display import HTML

from nhtsa.evaluate import single_label


def stats(values: dict[str, str]) -> HTML:
    """A row of headline numbers: {label: formatted value}."""
    cells = "".join(
        f'<div style="margin-right: 2.5em"><div style="font-size: 1.5em">{html.escape(v)}'
        f'</div><div style="opacity: 0.7">{html.escape(k)}</div></div>'
        for k, v in values.items()
    )
    return HTML(f'<div style="display: flex; flex-wrap: wrap">{cells}</div>')


def show_texts(header: str, *texts: str) -> HTML:
    """A bold header line over one or more texts, side by side."""
    columns = "".join(
        f'<div style="flex: 1; white-space: pre-wrap">{html.escape(t)}</div>'
        for t in texts
    )
    return HTML(
        f'<p style="margin-top: 1em"><b>{html.escape(header)}</b></p>'
        f'<div style="display: flex; gap: 2em">{columns}</div>'
    )


def cluster_names(words: list[list[str]]) -> list[str]:
    """Display names like "03 brake, pedal, brakes" from each cluster's top words."""
    return [f"{k:02d} {', '.join(w[:3])}" for k, w in enumerate(words)]


def cluster_table(
    clusters: np.ndarray, labels: pl.Series, words: list[list[str]]
) -> pl.DataFrame:
    """One row per cluster: size, most common component label and its share, top words.

    The label and share are over the cluster's single-label complaints. High
    share = the cluster lines up with a component; low share = a cross-cutting
    theme (recalls, dealer experience, …).
    """
    assigned = _assigned(clusters, label=single_label(labels))
    top_label = (
        assigned.drop_nulls("label")
        .group_by("cluster", "label")
        .len()
        .with_columns(share=pl.col("len") / pl.col("len").sum().over("cluster"))
        .sort("len", descending=True)
        .unique("cluster", keep="first")
        .select("cluster", top_label="label", label_share="share")
    )
    names = pl.DataFrame(
        {
            "cluster": range(len(words)),
            "name": cluster_names(words),
            "top_words": [", ".join(w) for w in words],
        },
        schema_overrides={"cluster": assigned.schema["cluster"]},
    )
    return (
        assigned.group_by("cluster")
        .agg(complaints=pl.len())
        .join(top_label, on="cluster", how="left")
        .join(names, on="cluster", how="left")
        .sort("cluster")
        .select("name", "complaints", "top_label", "label_share", "top_words")
    )


def label_share_chart(
    clusters: np.ndarray, labels: pl.Series, names: list[str]
) -> alt.Chart:
    """Heatmap: the mix of component labels within each cluster."""
    counts = _shares(
        _assigned(clusters, label=single_label(labels)).drop_nulls("label"),
        by="label",
        within="cluster",
        names=names,
    )
    return _heatmap(counts, "Component labels within each cluster", 560, names).encode(
        x=alt.X("label:N", title=None, axis=alt.Axis(labelAngle=-45)),
        tooltip=["name", "label", alt.Tooltip("share:Q", format=".0%"), "len"],
    )


def year_share_chart(
    clusters: np.ndarray, years: pl.Series, names: list[str]
) -> alt.Chart:
    """Heatmap: each cluster's share of the complaints received in each year."""
    counts = _shares(
        _assigned(clusters, year=years), by="year", within="year", names=names
    )
    return _heatmap(counts, "Cluster share by year received", 640, names).encode(
        x=alt.X("year:O", title=None),
        tooltip=["name", "year", alt.Tooltip("share:Q", format=".1%")],
    )


def _assigned(clusters: np.ndarray, **columns: pl.Series) -> pl.DataFrame:
    return pl.DataFrame({"cluster": clusters, **columns}).filter(pl.col("cluster") >= 0)


def _shares(df: pl.DataFrame, by: str, within: str, names: list[str]) -> pl.DataFrame:
    return (
        df.group_by("cluster", by)
        .len()
        .with_columns(
            share=pl.col("len") / pl.col("len").sum().over(within),
            name=pl.col("cluster").replace_strict(
                dict(enumerate(names)), return_dtype=pl.String
            ),
        )
    )


def _heatmap(
    counts: pl.DataFrame, title: str, width: int, names: list[str]
) -> alt.Chart:
    """Clusters down the side, shaded by share; the caller adds the x encoding."""
    chart = alt.Chart(
        counts, mark="rect", title=title, width=width, height=max(200, 20 * len(names))
    )
    return chart.encode(
        y=alt.Y("name:N", title=None, sort=names),
        color=alt.Color("share:Q", scale=alt.Scale(scheme="blues"), title="share"),
    )

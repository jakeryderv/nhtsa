import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import html

    import altair as alt
    import marimo as mo
    import polars as pl

    from nhtsa.load import load_complaints, load_rows

    return alt, html, load_complaints, load_rows, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NHTSA complaints: overview

    One row per complaint (`ODINO`) from `load_complaints()`; field definitions are in
    `data/raw/CMPL.txt`. Caveats from NHTSA's notes in that file:

    - Since a May 2021 system change, previously blank Y/N fields read **N** and blank
      numbers read **0**, so N/0 on older complaints can mean "not recorded".
    - Complaint text (`CDESCR`) is capped at 2,048 characters.
    - The current year is partial.
    """)
    return


@app.cell
def _(load_complaints, load_rows, pl):
    rows = load_rows()
    complaints = load_complaints().with_columns(pl.col("LDATE").dt.year().alias("YEAR"))
    return complaints, rows


@app.cell(hide_code=True)
def _(complaints, mo, pl, rows):
    _multi = complaints.select(
        (pl.col("COMPONENTS").list.len() > 1).mean().alias("components"),
        (pl.col("PRODUCTS").list.len() > 1).mean().alias("products"),
    ).row(0, named=True)
    mo.hstack(
        [
            mo.stat(f"{complaints.height:,}", label="complaints"),
            mo.stat(f"{rows.height:,}", label="flat-file rows"),
            mo.stat(
                f"{complaints['LDATE'].min()} → {complaints['LDATE'].max()}",
                label="received",
            ),
            mo.stat(f"{_multi['components']:.1%}", label="with >1 component"),
            mo.stat(f"{_multi['products']:.1%}", label="with >1 vehicle/product"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Volume and source
    """)
    return


@app.cell
def _(alt, complaints, pl):
    # Codes from CMPL.txt field 21; the long tail is lumped into OTHER.
    SOURCES = {
        "IVOQ": "NHTSA web site",
        "EVOQ": "hotline",
        "VOQ": "owner questionnaire",
        "MIVQ": "mobile app",
        "MAVQ": "mobile app",
        "LETR": "consumer letter",
    }
    _by_year = (
        complaints.with_columns(
            pl.col("CMPL_TYPE").replace_strict(SOURCES, default="other").alias("source")
        )
        .group_by("YEAR", "source")
        .len()
    )
    alt.Chart(_by_year).mark_bar().encode(
        x=alt.X("YEAR:O", title=None),
        y=alt.Y("len:Q", title="complaints"),
        color=alt.Color("source:N", sort="-y"),
        tooltip=["YEAR", "source", alt.Tooltip("len:Q", format=",")],
    ).properties(title="Complaints received per year, by source", width=720)
    return


@app.cell
def _(alt, complaints):
    # Codes from CMPL.txt field 46.
    _types = {"V": "vehicle", "T": "tires", "E": "equipment", "C": "child restraint"}
    _by_type = (
        complaints["PROD_TYPE"].replace_strict(_types, default="unknown").value_counts()
    )
    alt.Chart(_by_type).mark_bar().encode(
        x=alt.X("count:Q", title="complaints"),
        y=alt.Y("PROD_TYPE:N", sort="-x", title=None),
        tooltip=[alt.Tooltip("count:Q", format=",")],
    ).properties(title="Product type", width=500)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Components

    `COMPDESC` is hierarchical (`GROUP:SUBSYSTEM:PART`). The top-level group is a
    candidate label set for evaluating clusters later, but it needs cleanup first:

    - Detail shrinks over time: ~85% of 1995–99 components have subsystems, ~10%
      after 2010.
    - Overlapping groups coexist, e.g. `ENGINE` vs `ENGINE AND ENGINE COOLING`,
      `SERVICE BRAKES` vs `SERVICE BRAKES, HYDRAULIC`.
    """)
    return


@app.cell
def _(complaints, pl):
    components = (
        complaints.select("ODINO", "COMPONENTS")
        .explode("COMPONENTS")
        .with_columns(pl.col("COMPONENTS").str.split(":").list.first().alias("GROUP"))
    )
    return (components,)


@app.cell
def _(alt, components, mo):
    # A complaint with two parts in the same group counts once.
    _groups = components.unique(["ODINO", "GROUP"]).group_by("GROUP").len()
    mo.vstack(
        [
            mo.md(
                f"{components['GROUP'].n_unique()} top-level groups, "
                f"{components['COMPONENTS'].n_unique():,} distinct full components."
            ),
            alt.Chart(_groups.sort("len", descending=True).head(25))
            .mark_bar()
            .encode(
                x=alt.X("len:Q", title="complaints"),
                y=alt.Y("GROUP:N", sort="-x", title=None),
                tooltip=["GROUP", alt.Tooltip("len:Q", format=",")],
            )
            .properties(title="Top 25 component groups", width=560),
        ]
    )
    return


@app.cell
def _(alt, complaints, pl):
    _per_complaint = complaints.select(
        pl.col("COMPONENTS").list.len().clip(upper_bound=6).alias("n")
    )["n"].value_counts()
    alt.Chart(_per_complaint).mark_bar().encode(
        x=alt.X("n:O", title="components per complaint (6 = 6+)"),
        y=alt.Y("count:Q", title="complaints"),
        tooltip=[alt.Tooltip("count:Q", format=",")],
    ).properties(width=360)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Vehicles
    """)
    return


@app.cell
def _(complaints):
    products = (
        complaints.select("ODINO", "PRODUCTS").explode("PRODUCTS").unnest("PRODUCTS")
    )
    return (products,)


@app.cell
def _(alt, mo, pl, products):
    _makes = (
        products.unique(["ODINO", "MAKETXT"])["MAKETXT"]
        .value_counts(sort=True)
        .head(20)
    )
    _years = (
        products.select(pl.col("YEARTXT").cast(pl.Int32, strict=False))
        .filter(pl.col("YEARTXT").is_between(1975, 2027))["YEARTXT"]
        .value_counts()
    )
    mo.hstack(
        [
            alt.Chart(_makes)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="complaints"),
                y=alt.Y("MAKETXT:N", sort="-x", title=None),
                tooltip=[alt.Tooltip("count:Q", format=",")],
            )
            .properties(title="Top 20 makes", width=320),
            alt.Chart(_years)
            .mark_bar()
            .encode(
                x=alt.X("YEARTXT:O", title="model year", axis=alt.Axis(labelAngle=-90)),
                y=alt.Y("count:Q", title="complaints"),
                tooltip=["YEARTXT", alt.Tooltip("count:Q", format=",")],
            )
            .properties(title="Model year", width=420),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Severity

    Share of complaints reporting each outcome, by year received. Read pre-2021
    values with the blank → N/0 caveat in mind. `MEDICAL_ATTN` was added in April
    2014, so it's shown from 2015 on.
    """)
    return


@app.cell
def _(alt, complaints, pl):
    _severity = (
        complaints.group_by("YEAR")
        .agg(
            crash=pl.col("CRASH").mean(),
            fire=pl.col("FIRE").mean(),
            injury=(pl.col("INJURED") > 0).mean(),
            medical=pl.col("MEDICAL_ATTN").filter(pl.col("YEAR") >= 2015).mean(),
            death=(pl.col("DEATHS") > 0).mean(),
        )
        .unpivot(index="YEAR", variable_name="outcome", value_name="share")
    )
    alt.Chart(_severity).mark_line(point=True).encode(
        x=alt.X("YEAR:O", title=None),
        y=alt.Y("share:Q", axis=alt.Axis(format="%")),
        color="outcome:N",
        tooltip=["YEAR", "outcome", alt.Tooltip("share:Q", format=".2%")],
    ).properties(width=720)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Complaint text

    Things that will matter for TF-IDF and embeddings:

    - Everything received through 2020 is all caps; from 2022 almost nothing is
      (the 2021 system change). Unnormalized, casing alone reveals the era.
    - The hotline template ("TL* The contact owns a …") starts in 2007.
    - Older texts often end in analyst initials like `*TR` or `*AK`.
    """)
    return


@app.cell
def _(complaints, pl):
    text = complaints.select(
        "ODINO",
        "YEAR",
        words=pl.col("CDESCR").str.count_matches(r"\S+"),
        truncated=pl.col("CDESCR").str.len_chars() >= 2048,
        all_caps=pl.col("CDESCR").str.contains(r"[A-Z]")
        & ~pl.col("CDESCR").str.contains(r"[a-z]"),
        template=pl.col("CDESCR").str.contains(r"(?i)^(TL\*\s*)?THE CONTACT"),
    )
    return (text,)


@app.cell
def _(alt, mo, pl, text):
    _hist = (
        text.group_by((pl.col("words").clip(upper_bound=400) // 10 * 10).alias("start"))
        .agg(count=pl.len())
        .with_columns(end=pl.col("start") + 10)
    )
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(f"{text['words'].median():.0f}", label="median words"),
                    mo.stat(
                        f"{text['truncated'].mean():.2%}", label="hit 2,048-char cap"
                    ),
                    mo.stat(f"{text['all_caps'].mean():.1%}", label="all caps"),
                    mo.stat(f"{text['template'].mean():.1%}", label="hotline template"),
                ]
            ),
            alt.Chart(_hist)
            .mark_bar()
            .encode(
                x=alt.X("start:Q", bin="binned", title="words (400 = 400+)"),
                x2="end:Q",
                y=alt.Y("count:Q", title="complaints"),
            )
            .properties(width=720, height=200),
        ]
    )
    return


@app.cell
def _(alt, pl, text):
    _style = (
        text.group_by("YEAR")
        .agg(pl.col("all_caps").mean(), pl.col("template").mean())
        .unpivot(index="YEAR", variable_name="style", value_name="share")
    )
    alt.Chart(_style).mark_line(point=True).encode(
        x=alt.X("YEAR:O", title=None),
        y=alt.Y("share:Q", axis=alt.Axis(format="%")),
        color="style:N",
        tooltip=["YEAR", "style", alt.Tooltip("share:Q", format=".1%")],
    ).properties(title="Text style by year received", width=720)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Read some complaints
    """)
    return


@app.cell
def _(components, mo):
    _top = components["GROUP"].value_counts(sort=True)["GROUP"].drop_nulls().head(30)
    group_picker = mo.ui.dropdown(
        options=_top.to_list(), value=_top[0], label="Component group"
    )
    resample = mo.ui.button(value=0, on_click=lambda n: n + 1, label="New sample")
    mo.hstack([group_picker, resample], justify="start")
    return group_picker, resample


@app.cell
def _(complaints, components, group_picker, html, mo, pl, resample):
    def _wrapped(text):
        return mo.Html(f'<div style="white-space: pre-wrap">{html.escape(text)}</div>')

    _ids = components.filter(pl.col("GROUP") == group_picker.value)["ODINO"].unique()
    _sample = complaints.filter(pl.col("ODINO").is_in(_ids.implode())).sample(
        5, seed=resample.value
    )
    mo.vstack(
        [
            mo.vstack(
                [
                    mo.md(
                        f"**{_r['ODINO']}** · received {_r['LDATE']} · "
                        f"{', '.join(_r['COMPONENTS'])}"
                    ),
                    _wrapped(_r["CDESCR"]),
                ]
            )
            for _r in _sample.iter_rows(named=True)
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Missing values (flat-file rows)
    """)
    return


@app.cell
def _(alt, pl, rows):
    _missing = rows.select(pl.all().is_null().mean()).unpivot(
        variable_name="column", value_name="null_share"
    )
    alt.Chart(_missing).mark_bar().encode(
        x=alt.X("null_share:Q", axis=alt.Axis(format="%"), title="null"),
        y=alt.Y("column:N", sort="-x", title=None),
        tooltip=["column", alt.Tooltip("null_share:Q", format=".1%")],
    ).properties(width=480)
    return


if __name__ == "__main__":
    app.run()

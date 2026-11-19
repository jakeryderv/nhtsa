import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import html

    import altair as alt
    import marimo as mo
    import polars as pl
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    from nhtsa.dataset import cleaned_complaints
    from nhtsa.labels import COMPONENT_LABELS, component_label

    return (
        COMPONENT_LABELS,
        LogisticRegression,
        TfidfVectorizer,
        alt,
        cleaned_complaints,
        component_label,
        html,
        mo,
        pl,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Text cleaning and component labels

    Checks `nhtsa.text.clean_text` and `nhtsa.labels.component_labels` on the full
    data, via `nhtsa.dataset.cleaned_complaints()`. Cleaning adds a `TEXT` column;
    the original `CDESCR` is untouched.
    """)
    return


@app.cell
def _(cleaned_complaints, pl):
    # CHANGED: changed beyond what lowercasing and whitespace cleanup alone would do.
    complaints = cleaned_complaints().with_columns(
        CHANGED=pl.col("TEXT")
        != pl.col("CDESCR")
        .str.to_lowercase()
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )
    return (complaints,)


@app.cell(hide_code=True)
def _(complaints, mo, pl):
    _chars = complaints.select(
        1
        - pl.col("TEXT").str.len_chars().sum() / pl.col("CDESCR").str.len_chars().sum()
    ).item()
    mo.hstack(
        [
            mo.stat(f"{complaints['CHANGED'].mean():.1%}", label="changed by cleaning"),
            mo.stat(f"{_chars:.1%}", label="characters removed"),
            mo.stat(f"{complaints['HOTLINE'].mean():.1%}", label="hotline template"),
            mo.stat(
                f"{complaints['PARTS_UNAVAILABLE'].mean():.1%}",
                label="recall parts unavailable",
            ),
            mo.stat(
                f"{(complaints['TEXT'] == '').sum():,}", label="empty after cleaning"
            ),
        ]
    )
    return


@app.cell
def _(alt, complaints, pl):
    _by_year = (
        complaints.group_by("YEAR")
        .agg(
            pl.col("CHANGED").mean().alias("changed by cleaning"),
            pl.col("HOTLINE").mean().alias("hotline template"),
            pl.col("PARTS_UNAVAILABLE").mean().alias("recall parts unavailable"),
        )
        .unpivot(index="YEAR", variable_name="flag", value_name="share")
    )
    alt.Chart(_by_year).mark_line(point=True).encode(
        x=alt.X("YEAR:O", title=None),
        y=alt.Y("share:Q", axis=alt.Axis(format="%")),
        color="flag:N",
        tooltip=["YEAR", "flag", alt.Tooltip("share:Q", format=".1%")],
    ).properties(title="Share of complaints by year received", width=720)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Before and after
    """)
    return


@app.cell
def _(mo):
    sample_filter = mo.ui.dropdown(
        options=[
            "changed by cleaning",
            "hotline template",
            "recall parts unavailable",
            "any",
        ],
        value="changed by cleaning",
        label="Show",
    )
    resample = mo.ui.button(value=0, on_click=lambda n: n + 1, label="New sample")
    mo.hstack([sample_filter, resample], justify="start")
    return resample, sample_filter


@app.cell
def _(complaints, html, mo, pl, resample, sample_filter):
    def _wrapped(text):
        return mo.Html(f'<div style="white-space: pre-wrap">{html.escape(text)}</div>')

    _filters = {
        "changed by cleaning": pl.col("CHANGED"),
        "hotline template": pl.col("HOTLINE"),
        "recall parts unavailable": pl.col("PARTS_UNAVAILABLE"),
        "any": pl.lit(True),
    }
    _sample = complaints.filter(_filters[sample_filter.value]).sample(
        4, seed=resample.value
    )
    mo.vstack(
        [
            mo.vstack(
                [
                    mo.md(
                        f"**{_r['ODINO']}** · {_r['YEAR']} · {', '.join(_r['LABELS'])}"
                    ),
                    mo.hstack(
                        [_wrapped(_r["CDESCR"]), _wrapped(_r["TEXT"])],
                        widths="equal",
                        align="start",
                    ),
                ]
            )
            for _r in _sample.iter_rows(named=True)
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What's left

    Leftover `*` tokens and the most common openings, for spotting boilerplate the
    patterns miss. "the contact stated…" is kept on purpose: the rest of a hotline
    complaint describes the actual problem.
    """)
    return


@app.cell
def _(complaints, mo, pl):
    _stars = (
        complaints.select(pl.col("TEXT").str.extract_all(r"\*\S{1,8}").alias("token"))
        .explode("token")
        .drop_nulls()["token"]
        .value_counts(sort=True)
        .head(15)
    )
    _openings = (
        complaints.select(
            pl.col("TEXT").str.extract(r"^(\S+ \S+ \S+ \S+ \S+)").alias("opening")
        )
        .drop_nulls()["opening"]
        .value_counts(sort=True)
        .head(15)
    )
    mo.hstack([mo.ui.table(_stars), mo.ui.table(_openings)], widths="equal")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Does the text still reveal its era?

    Complaints from just before NHTSA's 2021 system change (2017–2020) vs just after
    (2022–2025), 25k each. A TF-IDF + logistic regression model tries to tell them
    apart. Some signal is real (new vehicle tech, new recalls), so accuracy won't hit
    50%, but formatting artifacts should account for much less after cleaning. The
    top words show what still separates the eras; expect model years, recalls of
    the period (e.g. Takata), street addresses in older complaints (newer ones are
    redacted), and shifts in the hotline's wording.
    """)
    return


@app.cell
def _(complaints, pl):
    era_sample = pl.concat(
        [
            complaints.filter(
                pl.col("YEAR").is_between(2017, 2020), pl.col("TEXT") != ""
            )
            .sample(25_000, seed=0)
            .with_columns(AFTER=pl.lit(False)),
            complaints.filter(
                pl.col("YEAR").is_between(2022, 2025), pl.col("TEXT") != ""
            )
            .sample(25_000, seed=0)
            .with_columns(AFTER=pl.lit(True)),
        ]
    )
    return (era_sample,)


@app.cell
def _(LogisticRegression, TfidfVectorizer, era_sample, mo, pl, train_test_split):
    def _fit(texts, lowercase):
        vectorizer = TfidfVectorizer(lowercase=lowercase, min_df=5, sublinear_tf=True)
        x = vectorizer.fit_transform(texts)
        x_train, x_test, y_train, y_test = train_test_split(
            x, era_sample["AFTER"].to_numpy(), test_size=0.25, random_state=0
        )
        model = LogisticRegression(max_iter=1000).fit(x_train, y_train)
        return model.score(x_test, y_test), vectorizer, model

    _raw_acc, _, _ = _fit(era_sample["CDESCR"].to_list(), lowercase=False)
    _lower_acc, _, _ = _fit(era_sample["CDESCR"].to_list(), lowercase=True)
    _clean_acc, era_vectorizer, era_model = _fit(
        era_sample["TEXT"].to_list(), lowercase=False
    )
    mo.ui.table(
        pl.DataFrame(
            {
                "text": ["raw (case kept)", "raw, lowercased", "cleaned (TEXT)"],
                "era accuracy": [_raw_acc, _lower_acc, _clean_acc],
            }
        ),
        format_mapping={"era accuracy": "{:.1%}"},
        selection=None,
    )
    return era_model, era_vectorizer


@app.cell
def _(alt, era_model, era_vectorizer, pl):
    _weights = pl.DataFrame(
        {
            "word": era_vectorizer.get_feature_names_out(),
            "weight": era_model.coef_[0],
        }
    )
    _top = pl.concat(
        [
            _weights.sort("weight").head(20).with_columns(era=pl.lit("2017–2020")),
            _weights.sort("weight").tail(20).with_columns(era=pl.lit("2022–2025")),
        ]
    )
    alt.Chart(_top).mark_bar().encode(
        x=alt.X("weight:Q", title="weight toward 2022–2025"),
        y=alt.Y("word:N", sort="x", title=None),
        color=alt.Color("era:N", title="predicts"),
        tooltip=["word", alt.Tooltip("weight:Q", format=".2f")],
    ).properties(title="Words that most separate the eras (cleaned text)", width=480)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Component labels

    NHTSA's 54 top-level component groups mapped to consistent labels
    (`nhtsa.labels.COMPONENT_LABELS`). Unmapped groups would show as null.
    """)
    return


@app.cell
def _(COMPONENT_LABELS, complaints, component_label, mo, pl):
    _mapping = (
        complaints.select("ODINO", "COMPONENTS")
        .explode("COMPONENTS")
        .select(
            "ODINO",
            group=pl.col("COMPONENTS").str.split(":").list.first(),
            label=component_label(pl.col("COMPONENTS")),
        )
        .unique(["ODINO", "group"])
        .group_by("label", "group")
        .agg(complaints=pl.len())
        .sort("label", "complaints", descending=[False, True])
    )
    _unmapped = _mapping.filter(
        pl.col("label").is_null(), pl.col("group").is_not_null()
    )
    mo.vstack(
        [
            mo.md(
                f"{len(set(COMPONENT_LABELS.values()))} labels from "
                f"{len(COMPONENT_LABELS)} groups; **{_unmapped.height} unmapped "
                "groups** in the data."
            ),
            mo.ui.table(_mapping, page_size=15, selection=None),
        ]
    )
    return


@app.cell
def _(alt, complaints, pl):
    _counts = complaints.select(pl.col("LABELS").explode())["LABELS"].value_counts()
    _per_complaint = complaints.select(
        pl.col("LABELS").list.len().clip(upper_bound=4).alias("labels")
    )["labels"].value_counts()
    (
        alt.Chart(_counts)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="complaints"),
            y=alt.Y("LABELS:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("count:Q", format=",")],
        )
        .properties(title="Complaints per label", width=420)
        | alt.Chart(_per_complaint)
        .mark_bar()
        .encode(
            x=alt.X("labels:O", title="labels per complaint (4 = 4+)"),
            y=alt.Y("count:Q", title="complaints"),
            tooltip=[alt.Tooltip("count:Q", format=",")],
        )
        .properties(width=200)
    )
    return


if __name__ == "__main__":
    app.run()

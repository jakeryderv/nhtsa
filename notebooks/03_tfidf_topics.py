import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import html

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import NMF, TruncatedSVD
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
    from sklearn.preprocessing import normalize

    from nhtsa.dataset import sample_complaints
    from nhtsa.evaluate import scores, single_label

    return (
        ENGLISH_STOP_WORDS,
        MiniBatchKMeans,
        NMF,
        TfidfVectorizer,
        TruncatedSVD,
        alt,
        html,
        mo,
        normalize,
        np,
        pl,
        sample_complaints,
        scores,
        single_label,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TF-IDF topics

    Classic baselines on the shared 100k-complaint sample (`sample_complaints()`,
    cleaned `TEXT`):

    - **NMF topics** on TF-IDF: interpretable word lists; each complaint is assigned
      its strongest topic.
    - **LSA + k-means**: TF-IDF reduced to 100 dimensions, then k-means.

    Both are scored with `nhtsa.evaluate.scores`: agreement with component labels
    (NMI/ARI on single-label complaints, excluding `unknown`) and `era_accuracy`
    (how well clusters predict 2017–20 vs 2022–25; 0.5 = no era signal).
    """)
    return


@app.cell
def _(mo, sample_complaints):
    sample = sample_complaints()
    mo.md(
        f"{sample.height:,} complaints, received "
        f"{sample['YEAR'].min()}–{sample['YEAR'].max()}."
    )
    return (sample,)


@app.cell
def _(ENGLISH_STOP_WORDS, TfidfVectorizer, mo, sample):
    # "contact"/"stated" come from the hotline template; "redacted" is a placeholder.
    STOP_WORDS = sorted(ENGLISH_STOP_WORDS | {"contact", "stated", "redacted"})
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b[a-z][a-z]+\b",  # letters only: no model years or mileage
        stop_words=STOP_WORDS,
        ngram_range=(1, 2),
        min_df=10,
        max_df=0.5,
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(sample["TEXT"].to_list())
    vocab = vectorizer.get_feature_names_out()
    mo.md(f"TF-IDF: {tfidf.shape[1]:,} terms (words and word pairs).")
    return tfidf, vocab


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## NMF topics
    """)
    return


@app.cell
def _(NMF, np, tfidf):
    N_TOPICS = 25  # about the number of component labels
    nmf = NMF(N_TOPICS, init="nndsvda", random_state=0, max_iter=300)
    doc_topics = nmf.fit_transform(tfidf)
    # Strongest topic per complaint; -1 if the complaint has no weight on any topic.
    nmf_clusters = np.where(doc_topics.max(axis=1) > 0, doc_topics.argmax(axis=1), -1)
    return N_TOPICS, doc_topics, nmf, nmf_clusters


@app.cell
def _(nmf, nmf_clusters, pl, sample, single_label, vocab):
    def _top_words(k, n):
        return [vocab[i] for i in nmf.components_[k].argsort()[::-1][:n]]

    topic_names = [
        f"{k:02d} {', '.join(_top_words(k, 3))}" for k in range(len(nmf.components_))
    ]
    _assigned = pl.DataFrame(
        {"cluster": nmf_clusters, "label": single_label(sample["LABELS"])}
    ).filter(pl.col("cluster") >= 0)
    _top_label = (
        _assigned.drop_nulls("label")
        .group_by("cluster", "label")
        .len()
        .with_columns(share=pl.col("len") / pl.col("len").sum().over("cluster"))
        .sort("len", descending=True)
        .unique("cluster", keep="first")
    )
    topic_table = (
        _assigned.group_by("cluster")
        .len()
        .join(_top_label.select("cluster", "label", "share"), on="cluster", how="left")
        .with_columns(
            topic=pl.col("cluster").map_elements(
                lambda k: topic_names[k], return_dtype=pl.String
            ),
            words=pl.col("cluster").map_elements(
                lambda k: ", ".join(_top_words(k, 10)), return_dtype=pl.String
            ),
        )
        .sort("cluster")
        .select(
            "topic",
            complaints="len",
            top_label="label",
            label_share="share",
            top_words="words",
        )
    )
    return topic_names, topic_table


@app.cell
def _(mo, topic_table):
    mo.ui.table(
        topic_table,
        format_mapping={"label_share": "{:.0%}"},
        page_size=25,
        selection=None,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `top_label` / `label_share`: the most common component label among the topic's
    single-label complaints, and its share. High share = the topic lines up with a
    component; low share = a cross-cutting theme (recalls, dealer experience, …).
    """)
    return


@app.cell
def _(alt, nmf_clusters, pl, sample, single_label, topic_names):
    _counts = (
        pl.DataFrame({"cluster": nmf_clusters, "label": single_label(sample["LABELS"])})
        .filter(pl.col("cluster") >= 0)
        .drop_nulls("label")
        .group_by("cluster", "label")
        .len()
        .with_columns(
            share=pl.col("len") / pl.col("len").sum().over("cluster"),
            topic=pl.col("cluster").map_elements(
                lambda k: topic_names[k], return_dtype=pl.String
            ),
        )
    )
    alt.Chart(_counts).mark_rect().encode(
        x=alt.X("label:N", title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("topic:N", title=None, sort=topic_names),
        color=alt.Color("share:Q", scale=alt.Scale(scheme="blues"), title="share"),
        tooltip=["topic", "label", alt.Tooltip("share:Q", format=".0%"), "len"],
    ).properties(title="Component labels within each topic", width=560, height=520)
    return


@app.cell
def _(alt, nmf_clusters, pl, sample, topic_names):
    _by_year = (
        pl.DataFrame({"cluster": nmf_clusters, "YEAR": sample["YEAR"]})
        .filter(pl.col("cluster") >= 0)
        .group_by("YEAR", "cluster")
        .len()
        .with_columns(
            share=pl.col("len") / pl.col("len").sum().over("YEAR"),
            topic=pl.col("cluster").map_elements(
                lambda k: topic_names[k], return_dtype=pl.String
            ),
        )
    )
    alt.Chart(_by_year).mark_rect().encode(
        x=alt.X("YEAR:O", title=None),
        y=alt.Y("topic:N", title=None, sort=topic_names),
        color=alt.Color("share:Q", scale=alt.Scale(scheme="blues"), title="share"),
        tooltip=["topic", "YEAR", alt.Tooltip("share:Q", format=".1%")],
    ).properties(title="Topic share by year received", width=640, height=520)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LSA + k-means
    """)
    return


@app.cell
def _(MiniBatchKMeans, N_TOPICS, TruncatedSVD, normalize, tfidf):
    lsa = normalize(TruncatedSVD(100, random_state=0).fit_transform(tfidf))
    kmeans_clusters = MiniBatchKMeans(N_TOPICS, random_state=0, n_init=3).fit_predict(
        lsa
    )
    return (kmeans_clusters,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scores

    The random row is a reference: what the scores look like with no structure.
    """)
    return


@app.cell
def _(N_TOPICS, kmeans_clusters, mo, nmf_clusters, np, pl, sample, scores):
    _methods = {
        "NMF topics": nmf_clusters,
        "LSA + k-means": kmeans_clusters,
        "random (reference)": np.random.default_rng(0).integers(
            0, N_TOPICS, sample.height
        ),
    }
    _rows = [
        {"method": name, **scores(c, sample["LABELS"], sample["YEAR"].to_numpy())}
        for name, c in _methods.items()
    ]
    mo.ui.table(
        pl.DataFrame(_rows),
        format_mapping={
            k: "{:.3f}"
            for k in ["nmi", "ari", "coverage", "era_accuracy", "unassigned"]
        },
        selection=None,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Read a topic

    The complaints that weigh most on the chosen topic.
    """)
    return


@app.cell
def _(mo, topic_names):
    topic_picker = mo.ui.dropdown(
        options=topic_names, value=topic_names[0], label="Topic"
    )
    topic_picker
    return (topic_picker,)


@app.cell
def _(doc_topics, html, mo, sample, topic_names, topic_picker):
    def _wrapped(text):
        return mo.Html(f'<div style="white-space: pre-wrap">{html.escape(text)}</div>')

    _k = topic_names.index(topic_picker.value)
    _top = doc_topics[:, _k].argsort()[::-1][:6]
    mo.vstack(
        [
            mo.vstack(
                [
                    mo.md(
                        f"**{sample['ODINO'][int(_i)]}** · {sample['YEAR'][int(_i)]} · "
                        f"{', '.join(sample['LABELS'][int(_i)])}"
                    ),
                    _wrapped(sample["TEXT"][int(_i)]),
                ]
            )
            for _i in _top
        ]
    )
    return


if __name__ == "__main__":
    app.run()

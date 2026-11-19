"""TF-IDF features, the baseline clusterings built on them, and cluster words.

Cluster ids follow nhtsa.evaluate: integers, negative for "unassigned".
"""

import numpy as np
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.preprocessing import normalize

# "contact"/"stated" come from the hotline template; "redacted" is a placeholder.
STOP_WORDS = sorted(ENGLISH_STOP_WORDS | {"contact", "stated", "redacted"})
VOCABULARY = {
    "token_pattern": r"(?u)\b[a-z][a-z]+\b",  # letters only: no model years or mileage
    "stop_words": STOP_WORDS,
    "ngram_range": (1, 2),
    "min_df": 10,
    "max_df": 0.5,
}


def tfidf_features(texts: list[str]) -> tuple[sparse.csr_matrix, np.ndarray]:
    """TF-IDF matrix (complaints x terms) and its terms (words and word pairs)."""
    vectorizer = TfidfVectorizer(**VOCABULARY, sublinear_tf=True)
    return vectorizer.fit_transform(texts), vectorizer.get_feature_names_out()


def nmf_topics(
    tfidf: sparse.csr_matrix, n_topics: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """NMF on TF-IDF: (complaints x topics) weights and (topics x terms) weights."""
    nmf = NMF(n_topics, init="nndsvda", random_state=seed, max_iter=300)
    return nmf.fit_transform(tfidf), nmf.components_


def strongest_topic(doc_topics: np.ndarray) -> np.ndarray:
    """Each complaint's strongest topic; -1 if it has no weight on any topic."""
    return np.where(doc_topics.max(axis=1) > 0, doc_topics.argmax(axis=1), -1)


def lsa_kmeans(tfidf: sparse.csr_matrix, n_clusters: int, seed: int = 0) -> np.ndarray:
    """k-means clusters on TF-IDF reduced to 100 dimensions (LSA)."""
    lsa = normalize(TruncatedSVD(100, random_state=seed).fit_transform(tfidf))
    return MiniBatchKMeans(n_clusters, random_state=seed, n_init=3).fit_predict(lsa)


def cluster_term_weights(
    texts: list[str], clusters: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Class-based TF-IDF: (clusters x terms) weights and the terms.

    Each cluster's complaints are treated as one document, so a term scores high
    when it is frequent in the cluster and rare overall. Rows are in cluster-id
    order (0, 1, …); unassigned complaints are left out.
    """
    vectorizer = CountVectorizer(**VOCABULARY)
    counts = vectorizer.fit_transform(texts)
    assigned = np.flatnonzero(clusters >= 0)
    n_clusters = int(clusters.max()) + 1
    membership = sparse.csr_matrix(
        (np.ones(len(assigned)), (clusters[assigned], assigned)),
        shape=(n_clusters, len(texts)),
    )
    per_cluster = (membership @ counts).toarray()
    tf = per_cluster / np.maximum(per_cluster.sum(axis=1, keepdims=True), 1)
    term_totals = per_cluster.sum(axis=0)
    idf = np.log1p(per_cluster.sum() / n_clusters / np.maximum(term_totals, 1))
    return tf * idf, vectorizer.get_feature_names_out()


def top_terms(weights: np.ndarray, terms: np.ndarray, n: int = 10) -> list[list[str]]:
    """The n highest-weighted terms in each row of a (clusters x terms) matrix."""
    return [[str(terms[i]) for i in row.argsort()[::-1][:n]] for row in weights]

import numpy as np

from nhtsa.topics import cluster_term_weights, strongest_topic, top_terms


def test_strongest_topic_marks_zero_rows_unassigned():
    doc_topics = np.array([[0.1, 0.9], [0.0, 0.0], [0.5, 0.2]])
    assert strongest_topic(doc_topics).tolist() == [1, -1, 0]


def test_cluster_terms_pick_words_distinctive_to_each_cluster():
    # "vehicle" is everywhere; min_df=10 needs each term in at least 10 texts.
    texts = (
        ["vehicle brake pedal"] * 10
        + ["vehicle engine stalled"] * 10
        + ["vehicle noise"] * 10
    )
    clusters = np.array([0] * 10 + [1] * 10 + [-1] * 10)
    weights, terms = cluster_term_weights(texts, clusters)
    assert weights.shape[0] == 2  # unassigned complaints form no cluster
    top = top_terms(weights, terms, n=2)
    assert all("brake" in t or "pedal" in t for t in top[0])
    assert all("engine" in t or "stalled" in t for t in top[1])

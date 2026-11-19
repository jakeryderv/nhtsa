import numpy as np
import polars as pl

from nhtsa.embed import cached_embeddings


class CountingEncoder:
    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        return np.array([[len(t), 1.0] for t in texts])


def test_cached_embeddings_encodes_once(tmp_path):
    ids, texts = pl.Series([1, 2]), pl.Series(["brakes failed", "engine stalled"])
    encoder = CountingEncoder()
    first = cached_embeddings(ids, texts, tmp_path / "cache.npz", encoder)
    second = cached_embeddings(ids, texts, tmp_path / "cache.npz", encoder)
    assert encoder.calls == 1
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, [[13, 1], [14, 1]])


def test_cached_embeddings_rebuilds_when_rows_change(tmp_path):
    path = tmp_path / "cache.npz"
    encoder = CountingEncoder()
    cached_embeddings(pl.Series([1, 2]), pl.Series(["a", "bb"]), path, encoder)
    # Same ids, different text (e.g. after a change to clean_text).
    changed = cached_embeddings(
        pl.Series([1, 2]), pl.Series(["a", "bbb"]), path, encoder
    )
    np.testing.assert_array_equal(changed, [[1, 1], [3, 1]])
    # Same texts, different rows.
    cached_embeddings(pl.Series([1, 3]), pl.Series(["a", "bbb"]), path, encoder)
    # Same rows, different order.
    cached_embeddings(pl.Series([3, 1]), pl.Series(["bbb", "a"]), path, encoder)
    assert encoder.calls == 4

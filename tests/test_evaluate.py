import numpy as np
import polars as pl
import pytest

from nhtsa.evaluate import era_accuracy, label_agreement, single_label


def test_single_label_keeps_only_one_known_label():
    labels = pl.Series([["brakes"], ["brakes", "engine"], ["unknown"], []])
    assert single_label(labels).to_list() == ["brakes", None, None, None]


def test_label_agreement_perfect_and_ignores_unassigned():
    labels = pl.Series([["brakes"], ["brakes"], ["engine"], ["engine"], ["tires"]])
    clusters = np.array([0, 0, 1, 1, -1])
    result = label_agreement(clusters, labels)
    assert result["nmi"] == pytest.approx(1.0)
    assert result["ari"] == pytest.approx(1.0)
    assert result["coverage"] == pytest.approx(0.8)


def test_era_accuracy_extremes():
    years = np.array([2018] * 50 + [2023] * 50)
    by_era = np.array([0] * 50 + [1] * 50)
    assert era_accuracy(by_era, years) == pytest.approx(1.0)
    mixed = np.tile([0, 1], 50)  # each cluster half before, half after
    assert era_accuracy(mixed, years) == pytest.approx(0.5)


def test_era_accuracy_balances_eras_and_ignores_other_years():
    # 90 before vs 10 after: without balancing, one cluster would score 0.9.
    years = np.array([2018] * 90 + [2023] * 10 + [2005] * 30)
    clusters = np.zeros(130, dtype=int)
    assert era_accuracy(clusters, years) == pytest.approx(0.5)

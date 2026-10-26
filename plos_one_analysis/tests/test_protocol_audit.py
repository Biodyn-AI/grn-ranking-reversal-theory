"""Focused checks for estimands, cluster weights, and metric-key matching."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protocol_audit import (  # noqa: E402
    aggregate_tissue_candidates,
    cluster_dependence,
    match_metric_pairs,
    metric_pair_rows,
    endpoint_status,
    row_weighted_cluster_interval,
)


def test_unequal_clusters_preserve_transition_row_estimand():
    comparisons = pd.DataFrame(
        {"cluster": ["large"] * 9 + ["small"], "reversal": [False] * 9 + [True]}
    )
    summary = row_weighted_cluster_interval(comparisons, "reversal", ["cluster"], 200, 42)
    assert summary["rate"] == pytest.approx(0.1)
    assert summary["equal_cluster_mean"] == pytest.approx(0.5)
    assert summary["n_clusters"] == 2
    assert summary["interval_low"] <= summary["rate"] <= summary["interval_high"]
    reordered = row_weighted_cluster_interval(
        comparisons.iloc[::-1], "reversal", ["cluster"], 200, 42
    )
    assert reordered == summary


def test_candidate_aggregation_changes_unit_and_event():
    comparisons = pd.DataFrame(
        {
            "tissue_from": ["immune"] * 3,
            "tissue_to": ["kidney"] * 3,
            "method_a": ["a"] * 3,
            "method_b": ["b"] * 3,
            "method_pair": ["a__b"] * 3,
            "candidate_set": ["all_pairs", "tf_sources", "tf_sources_targets"],
            "reversal": [False, True, False],
        }
    )
    aggregated = aggregate_tissue_candidates(comparisons)
    assert len(aggregated) == 1
    assert bool(aggregated.iloc[0].reversal_any_candidate)
    assert not bool(aggregated.iloc[0].reversal_all_candidates)
    assert aggregated.iloc[0].n_candidate_sets == 3
    assert comparisons.reversal.mean() == pytest.approx(1 / 3)


def test_exact_metric_matching_excludes_ties_and_missing_values():
    tissue_scores = pd.DataFrame(
        {
            "gene_set": ["x", "x", "y", "y"],
            "prediction_method": ["a", "b", "a", "b"],
            "aupr_median": [0.4, 0.2, 0.1, 0.3],
            "auroc_median": [0.6, 0.6, 0.5, 0.7],
            "f1_median": [0.2, 0.1, np.nan, 0.3],
        }
    )
    pairs = metric_pair_rows(tissue_scores)
    assert len(pairs) == 3
    assert pairs.set_index("metric").loc["AUPR", "status"] == "eligible"
    assert pairs.set_index("metric").loc["AUROC", "status"] == "tie_from"
    assert pairs.set_index("metric").loc["F1", "status"] == "missing_score"
    matched, exclusions = match_metric_pairs(pairs)
    assert matched.empty
    assert len(exclusions) == 1
    assert "AUROC:tie_from" in exclusions.iloc[0].excluded_reasons
    assert "F1:missing_score" in exclusions.iloc[0].excluded_reasons


def test_pair_clusters_sharing_methods_are_not_disjoint():
    comparisons = pd.DataFrame(
        {"method_a": ["a", "a", "b"], "method_b": ["b", "c", "c"]}
    )
    summary = cluster_dependence(comparisons)
    assert summary["n_method_pair_clusters"] == 3
    assert summary["n_cluster_pairs_sharing_method"] == 3
    assert summary["fraction_cluster_pairs_sharing_method"] == 1


def test_zero_event_interval_is_a_degenerate_empirical_resample():
    comparisons = pd.DataFrame({"cluster": ["a", "b"], "reversal": [False, False]})
    summary = row_weighted_cluster_interval(comparisons, "reversal", ["cluster"], 20, 42)
    assert summary["rate"] == summary["interval_low"] == summary["interval_high"] == 0


@pytest.mark.parametrize(
    "values,expected",
    [([0.3, 0.2, 0.1, 0.2], "eligible"),
     ([0.3, 0.3, 0.1, 0.2], "tie_from"),
     ([0.3, 0.2, 0.1, 0.1], "tie_to"),
     ([0.3, 0.3, 0.1, 0.1], "tie_both"),
     ([0.3, np.nan, 0.1, 0.2], "missing_score")],
)
def test_endpoint_exclusions_are_explicit(values, expected):
    assert endpoint_status(values) == expected

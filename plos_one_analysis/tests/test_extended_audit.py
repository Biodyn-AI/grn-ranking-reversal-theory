"""Focused checks for scale, informativeness, rounding, dependence and feasibility."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extended_audit import (  # noqa: E402
    four_significant_half_unit,
    held_out_logistic_scores,
    held_out_shift_bound,
    leave_two_methods_out,
    method_level_resampling,
    metric_rounding_simulation,
    normalized_axis_rows,
    reference_endpoints,
    reference_row_audit,
    screening_frame,
    source_full_ranking_aupr,
    tie_aware_zero_recovery_ap,
)
from zero_recovery_replay import flat_index, source_full_ranking_aupr as replay_aupr  # noqa: E402

INPUTS = ROOT / "inputs"
OUTPUTS = ROOT / "outputs"


def test_tie_aware_ap_of_zero_recovery_list_is_base_rate():
    ap, base_rate = tie_aware_zero_recovery_ap(20000, 30, 700, seed=3)
    assert ap == pytest.approx(base_rate, abs=1e-15)


def test_full_ranking_aupr_changes_with_list_length_but_tie_aware_ap_does_not():
    rng = np.random.default_rng(5)
    n, positives = 50000, 40
    labels = np.zeros(n, dtype=np.int8)
    labels[rng.choice(n, positives, replace=False)] = 1
    negatives = np.flatnonzero(labels == 0)
    values, tie_aware = [], []
    for length in (10, 3000):
        scores = np.zeros(n, dtype=np.float32)
        scores[rng.choice(negatives, length, replace=False)] = rng.uniform(0.01, 1, length)
        values.append(source_full_ranking_aupr(scores, labels))
        tie_aware.append(average_precision_score(labels, scores))
        assert replay_aupr(scores, labels) == source_full_ranking_aupr(scores, labels)
    assert values[0] != values[1]
    assert tie_aware[0] == pytest.approx(tie_aware[1], abs=1e-15)


def test_archived_reference_endpoints_and_rows():
    endpoints = reference_endpoints(pd.read_csv(INPUTS / "score_eval_grn_baselines_immune.csv"))
    zero = endpoints.groupby("reference")["zero_recovery"].sum()
    assert zero["hpn_dream"] == 9 and zero["beeline_gsd"] == 9
    assert zero["dorothea_trrust_union_immune"] == 1
    rows = reference_row_audit(pd.read_csv(INPUTS / "reference_shift_pairwise_rows.csv"), endpoints)
    assert len(rows) == 106 and int(rows["reversal"].sum()) == 34
    assert (rows["n_informative_endpoints"] == 2).sum() == 0
    assert rows["tied_under_tie_aware_ap"].all()
    assert rows.loc[rows["reversal"], "min_normalized_margin"].max() < 1e-3


def test_recovered_edge_counts_must_agree():
    table = pd.DataFrame({"method": ["a"], "reference": ["r"], "precision": [0.5], "recall": [0.5],
                          "aupr": [0.1], "pred_edges": [4], "true_edges": [8], "candidate_edges": [100]})
    with pytest.raises(ValueError):
        reference_endpoints(table)


def test_four_significant_half_units():
    assert four_significant_half_unit(0.5016) == pytest.approx(5e-5)
    assert four_significant_half_unit(0.5) == pytest.approx(5e-5)
    assert four_significant_half_unit(0.009266) == pytest.approx(5e-7)
    assert four_significant_half_unit(0.0) == 0.0


def test_rounding_ties_reverse_about_half_the_time():
    table = pd.DataFrame({"gene_set": ["x", "x", "y", "y"], "prediction_method": ["a", "b", "a", "b"],
                          "aupr_median": [0.2, 0.1, 0.2, 0.1], "auroc_median": [0.5, 0.5, 0.5, 0.5],
                          "f1_median": [0.3, 0.1, 0.1, 0.3]})
    matched = pd.DataFrame({"tissue_from": ["x"], "tissue_to": ["y"], "method_a": ["a"], "method_b": ["b"]})
    summary, _ = metric_rounding_simulation(table, matched, 4000, 1)
    by_metric = summary[summary["scope"] == "all_keys"].set_index("metric")
    assert by_metric.loc["AUPR", "mean_reversals"] == 0
    assert by_metric.loc["F1", "mean_reversals"] == 1
    assert by_metric.loc["AUROC", "mean_reversals"] == pytest.approx(0.5, abs=0.05)


def test_method_level_resampling_on_a_constant_grid():
    pairs = [("a", "b"), ("a", "c"), ("b", "c")]
    comparisons = pd.DataFrame({"method_a": [p[0] for p in pairs] * 2,
                                "method_b": [p[1] for p in pairs] * 2,
                                "reversal": [True, False, False, True, False, False]})
    result = method_level_resampling(comparisons, 3000, 2)
    assert result["rate"] == pytest.approx(1 / 3)
    assert result["n_methods"] == 3 and result["pair_clusters"] == 3
    assert 0 <= result["range_low"] <= result["range_high"] <= 1
    dropped = leave_two_methods_out(comparisons)
    assert len(dropped) == 3 and dropped["n"].eq(0).all()


def test_held_out_logistic_scores_reproduce_archived_predictions():
    candidate = pd.read_csv(INPUTS / "candidate_shift_pairwise_rows.csv")
    frame = screening_frame(candidate)
    scores = held_out_logistic_scores(frame, frame["reversal"].to_numpy().astype(int))
    archived = pd.read_csv(OUTPUTS / "protocol_classifier_predictions.csv")
    keys = ["gene_set", "shift_from", "shift_to", "method_pair"]
    merged = frame.assign(score=scores).merge(archived, on=keys)
    assert len(merged) == 135
    assert np.allclose(merged["score"], merged["learned_probability"], atol=1e-12)


def test_certified_rows_cannot_reverse_when_the_bound_holds():
    bound = held_out_shift_bound(pd.read_csv(INPUTS / "candidate_shift_pairwise_rows.csv"))
    totals = bound[bound["held_tissue"] == "all"]
    assert (totals["n_test"] == 135).all()
    assert (totals["reversals_total"] == 22).all()
    # |delta_from| > B and |delta_to - delta_from| <= B imply equal signs.
    delta_from, bound_value = 0.3, 0.2
    for shift in np.linspace(-bound_value, bound_value, 9):
        assert np.sign(delta_from) == np.sign(delta_from + shift)


def test_normalized_tissue_margins_use_tissue_invariant_base_rates():
    frames = normalized_axis_rows(INPUTS)
    tissue = frames["tissue"]
    assert (tissue["base_rate_from"] == tissue["base_rate_to"]).all()
    assert len(frames["candidate"]) == 135 and len(frames["reference"]) == 106


def test_flat_index_matches_masked_row_major_order():
    n = 6
    mask = ~np.eye(n, dtype=bool)
    positions = np.arange(n * n).reshape(n, n)[mask]
    lookup = {int(value): i for i, value in enumerate(positions)}
    for source in range(n):
        for target in range(n):
            if source != target:
                expected = lookup[source * n + target]
                assert flat_index(np.array([source]), np.array([target]), n)[0] == expected


def test_bootstrap_width_inputs_match_the_archived_source_text():
    widths = pd.read_csv(INPUTS / "source_bootstrap_ci_widths.csv")
    text = (INPUTS / "source_workshop_tables.md").read_text()
    assert ("Median AUPR CI widths rise sharply as candidate sets shrink: 1.02e-04 (all pairs), "
            "2.16e-03 (TF-source), and 7.86e-03 (TF-source-target).") in text
    assert widths.set_index("candidate_set")["median_aupr_ci_width"].to_dict() == {
        "all_pairs": 1.02e-4, "tf_sources": 2.16e-3, "tf_sources_targets": 7.86e-3}

"""Focused checks for algebraic constraints and shared-cell perturbations."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "algebra_null.py"
SPEC = importlib.util.spec_from_file_location("algebra_null", MODULE_PATH)
algebra_null = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(algebra_null)


def test_increasing_base_rates_force_gap_dominance_on_reversal():
    accounting = algebra_null.decompose(
        [0.1, 0.1, 0.1], [0.2, 0.1, 0.9], [1.0, -3.0, 2.0], [-2.0, 1.0, -0.001]
    )
    assert accounting["reversal"].all()
    assert accounting["gap_dominant"].all()
    assert np.isinf(accounting.loc[1, "magnitude_ratio"])
    assert accounting["closure_error"].max() < 1e-12


def test_decreasing_base_rates_allow_either_magnitude_ordering():
    accounting = algebra_null.decompose([0.8, 0.8], [0.1, 0.1], [1.0, 1.0], [-0.1, -8.0])
    assert accounting["reversal"].all()
    assert accounting["gap_dominant"].tolist() == [False, True]
    # r = b2/b1 and t = |delta2|/|delta1| give the exact condition 2r+t>1.
    rate_ratio = accounting["b2"] / accounting["b1"]
    margin_ratio = accounting["delta2"].abs() / accounting["delta1"].abs()
    assert np.array_equal(2 * rate_ratio + margin_ratio > 1, accounting["gap_dominant"])


def test_fixed_nonzero_gap_cannot_reverse_at_positive_base_rates():
    accounting = algebra_null.decompose([0.001, 0.9, 0.2], [0.9, 0.001, 0.2], [1, -2, 4], [1, -2, 4])
    assert not accounting["reversal"].any()
    assert np.array_equal(accounting["gap_term"], np.zeros(3))


def test_invalid_base_rates_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        algebra_null.decompose([0.0], [0.1], [1], [-1])
    with pytest.raises(ValueError, match="finite"):
        algebra_null.decompose([0.1], [0.2], [np.nan], [-1])


def test_endpoint_swap_preserves_reversal_but_changes_accounting():
    original = algebra_null.decompose([0.1], [0.9], [1.0], [-1.0])
    backward = algebra_null.decompose([0.9], [0.1], [-1.0], [1.0])
    assert original["reversal"].equals(backward["reversal"])
    assert original.loc[0, "gap_dominant"]
    assert not backward.loc[0, "gap_dominant"]


def toy_grid():
    score_rows = pd.DataFrame(
        [("t", c, m, v) for c, values in [("c1", [0.1, 0.2, 0.3]), ("c2", [0.3, 0.2, 0.1]), ("c3", [0.15, 0.2, 0.25])]
         for m, v in zip(["a", "b", "c"], values)],
        columns=["gene_set", "candidate_set", "prediction_method", "aupr_median"],
    )
    comparisons = pd.DataFrame(
        [("t", c1, c2, a, b) for c1, c2 in [("c1", "c2"), ("c1", "c3"), ("c2", "c3")]
         for a, b in [("a", "b"), ("a", "c"), ("b", "c")]],
        columns=["gene_set", "shift_from", "shift_to", "method_a", "method_b"],
    )
    return algebra_null.build_score_grid(score_rows, comparisons)


def test_comparison_status_uses_common_scores_for_shared_cells():
    grid = toy_grid()
    eligible, reversal = algebra_null.comparison_status(grid["scores"][None, :, :], grid["comparisons"])
    assert eligible.all()
    assert reversal.sum() == 6
    changed_scores = grid["scores"].copy()
    # Making c2 equal to c1 must remove reversals for both transitions touching c2.
    changed_scores[1] = changed_scores[0]
    _, changed_reversal = algebra_null.comparison_status(changed_scores[None, :, :], grid["comparisons"])
    assert changed_reversal.sum() == 0


def test_cell_permutation_preserves_each_cells_score_multiset_and_seed():
    scores = toy_grid()["scores"]
    first = algebra_null.permute_scores(scores, n_draws=25, seed=42)
    second = algebra_null.permute_scores(scores, n_draws=25, seed=42)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.sort(first, axis=2), np.broadcast_to(np.sort(scores, axis=1), first.shape))
    assert np.any(first[:, 0, :] != first[:, 1, :])


def test_zero_noise_retains_all_observed_reversals_and_has_no_new_ones():
    draws = algebra_null.noise_sensitivity(toy_grid(), n_draws=10, seed=42, fractions=[0.0])
    assert (draws["retained_reversals"] == 6).all()
    assert (draws["lost_reversals"] == 0).all()
    assert (draws["new_reversals"] == 0).all()
    assert (draws["eligible_comparisons"] == 9).all()
    assert np.allclose(draws["reversal_rate"], 6 / 9)


def test_noise_change_counts_reconcile_and_report_ties():
    draws = algebra_null.noise_sensitivity(toy_grid(), n_draws=30, seed=42, fractions=[1.0])
    assert (draws["retained_reversals"] + draws["lost_reversals"] == 6).all()
    assert (draws["retained_reversals"] + draws["new_reversals"] == draws["reversals"]).all()
    assert (draws["eligible_comparisons"] + draws["tied_comparisons"] == 9).all()


def test_prevalence_only_change_can_reverse_aupr_with_fixed_score_distributions():
    population, roc = algebra_null.fixed_gaussian_population([3.23e-5, 0.002006, 0.009975], n_tail=2000)
    assert population["aupr_margin_a_minus_b"].gt(0).tolist() == [True, True, False]
    assert population["normalized_gap"].gt(0).tolist() == [True, True, False]
    assert population["auroc_a"].nunique() == population["auroc_b"].nunique() == 1
    assert population["auroc_b"].min() > population["auroc_a"].max()
    assert np.all(np.diff(population["aupr_a"]) > 0)
    assert np.all(np.diff(population["aupr_b"]) > 0)
    # The positive-class distributions have different fixed tails; ROC curves cross.
    false_positive_a = roc.loc[roc["method"] == "A", "false_positive_rate"].to_numpy()
    false_positive_b = roc.loc[roc["method"] == "B", "false_positive_rate"].to_numpy()
    assert (false_positive_a < false_positive_b).any()
    assert (false_positive_a > false_positive_b).any()


def test_population_integration_converges_at_observed_prevalences():
    first, _ = algebra_null.fixed_gaussian_population([3.23e-5, 0.002006, 0.009975], n_tail=2000)
    second, _ = algebra_null.fixed_gaussian_population([3.23e-5, 0.002006, 0.009975], n_tail=4000)
    np.testing.assert_allclose(first[["aupr_a", "aupr_b"]], second[["aupr_a", "aupr_b"]], atol=5e-8, rtol=0)


def test_population_aupr_matches_independent_threshold_integral():
    population, _ = algebra_null.fixed_gaussian_population([3.23e-5, 0.002006, 0.009975])
    validation = algebra_null.validate_population_integrals(population)
    assert len(validation) == 6
    assert validation["absolute_difference"].max() < 5e-8

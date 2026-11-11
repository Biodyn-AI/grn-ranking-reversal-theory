"""Focused checks for scale, informativeness, rounding, dependence and feasibility."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extended_audit import (  # noqa: E402
    DIFFERENCE_WIDTH_MULTIPLE,
    four_significant_half_unit,
    held_out_logistic_scores,
    held_out_shift_bound,
    intraclass_correlation,
    leave_two_methods_out,
    method_level_resampling,
    metric_rounding_simulation,
    metric_tie_resolution,
    noise_yardstick,
    normalized_axis_rows,
    reference_endpoints,
    reference_row_audit,
    release_value_check,
    resolved_metric_comparison,
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


def test_tie_aware_column_is_label_vector_rate():
    check = json.loads((INPUTS / "immune_reference_panel" / "export_check.json").read_text())
    positives = {name: int(v["label_vector_positives"]) for name, v in check["references"].items()}
    endpoints = reference_endpoints(pd.read_csv(INPUTS / "score_eval_grn_baselines_immune.csv"), positives)
    zero = endpoints[endpoints["zero_recovery"]]
    expected = zero["reference"].map(positives) / int(check["n_candidates"])
    assert np.allclose(zero["tie_aware_ap_if_zero_recovery"], expected, rtol=0, atol=0)
    assert (zero["tie_aware_ap_if_zero_recovery"] < zero["base_rate"]).all()
    assert endpoints.loc[~endpoints["zero_recovery"], "tie_aware_ap_if_zero_recovery"].isna().all()
    # The reported rate exceeds the label-vector rate by at most 127/112.
    assert (endpoints["base_rate"] / endpoints["label_vector_base_rate"]).max() == pytest.approx(127 / 112)


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


def _release_check():
    table_a1 = pd.read_csv(INPUTS / "table_a1.csv")
    release = pd.read_csv(INPUTS / "source_release_method_summary_by_tissue.csv")
    return table_a1, release_value_check(table_a1, release)


def test_release_reproduces_every_deterministic_kidney_and_lung_median():
    _, check = _release_check()
    assert len(check) == 54
    deterministic = check[check["deterministic_scorer"] & (check["gene_set"] != "immune")]
    assert len(deterministic) == 24 and deterministic["reproduced"].all()
    # The release belongs to a later protocol version whose immune medians differ;
    # an isolated immune agreement must not make any immune value usable.
    assert check.loc[check["gene_set"] == "immune", "reproduced"].sum() <= 1
    assert not check.loc[check["gene_set"] == "immune", "tissue_validated"].any()
    assert check["usable"].sum() == 24
    assert set(check.loc[check["usable"], "gene_set"]) == {"kidney", "lung"}


def test_release_values_break_the_kidney_lung_pearson_spearman_tie():
    table_a1, check = _release_check()
    ties = metric_tie_resolution(table_a1, check)
    assert len(ties) == 5
    keys = set(zip(ties["tissue_from"], ties["tissue_to"], ties["method_a"], ties["method_b"]))
    assert keys == {("immune", "kidney", "pearson", "spearman"), ("immune", "lung", "pearson", "spearman"),
                    ("kidney", "lung", "pearson", "spearman"),
                    ("immune", "kidney", "random", "scgpt_attention"),
                    ("immune", "lung", "random", "scgpt_attention")}
    resolved = ties[ties["resolved"]]
    assert len(resolved) == 1
    row = resolved.iloc[0]
    assert (row["tissue_from"], row["tissue_to"], row["method_a"]) == ("kidney", "lung", "pearson")
    assert row["release_delta_from"] == pytest.approx(-1.0798e-6, rel=1e-3)
    assert row["release_delta_to"] == pytest.approx(7.0382e-6, rel=1e-3)
    assert bool(row["release_reversal"])


def test_conditional_rounding_leaves_two_or_four_auroc_reversals():
    table_a1, check = _release_check()
    matched = pd.read_csv(OUTPUTS / "protocol_metric_matched_rows.csv")
    fixed = {(r.gene_set, r.prediction_method, r.metric): r.release
             for r in check[check["usable"]].itertuples(index=False)}
    summary, keys = metric_rounding_simulation(table_a1, matched, 2000, 7, fixed=fixed)
    auroc = summary[(summary["metric"] == "AUROC") & (summary["scope"] == "all_keys")].iloc[0]
    assert auroc["p_zero"] == 0 and auroc["p_at_least_one"] == 1
    assert set(int(part.split(":")[0]) for part in auroc["count_distribution"].split(";")) == {2, 4}
    assert auroc["mean_reversals"] == pytest.approx(3.0, abs=0.1)
    per_key = keys[(keys["metric"] == "AUROC") & (keys["method_a"] == "pearson") & (keys["method_b"] == "spearman")]
    probability = per_key.set_index(["tissue_from", "tissue_to"])["reversal_probability"]
    assert probability[("kidney", "lung")] == 1.0
    # Kidney and lung order the pair oppositely, so exactly one immune transition reverses.
    assert probability[("immune", "kidney")] + probability[("immune", "lung")] == pytest.approx(1.0)
    unconditional, _ = metric_rounding_simulation(table_a1, matched, 2000, 7)
    assert unconditional.loc[(unconditional["metric"] == "AUROC") & (unconditional["scope"] == "all_keys"),
                             "p_zero"].iloc[0] > 0


def test_resolved_matched_comparison_adds_one_key():
    table_a1, check = _release_check()
    matched = pd.read_csv(OUTPUTS / "protocol_metric_matched_rows.csv")
    result = resolved_metric_comparison(table_a1, check, matched)
    assert result["n_matched"] == 41 and result["removed_keys"] == []
    assert result["added_keys"] == ["kidney/lung/pearson/spearman"]
    counts = {metric: values["k_matched"] for metric, values in result["per_metric"].items()}
    assert counts == {"AUPR": 2, "AUROC": 1, "F1": 3}


def test_difference_threshold_is_one_over_root_two_of_a_full_width():
    assert DIFFERENCE_WIDTH_MULTIPLE == pytest.approx(1 / np.sqrt(2))
    sigma = 0.37
    width = 2 * 1.959963984540054 * sigma  # full 95% interval width of one score
    rng = np.random.default_rng(11)
    difference = rng.normal(0, sigma, 400000) - rng.normal(0, sigma, 400000)
    assert (np.abs(difference) > DIFFERENCE_WIDTH_MULTIPLE * width).mean() == pytest.approx(0.05, abs=0.002)


def test_noise_yardstick_denominators():
    frames = normalized_axis_rows(INPUTS)
    yard = noise_yardstick(frames, pd.read_csv(INPUTS / "source_bootstrap_ci_widths.csv"))
    both = yard[yard["statistic"] == "rows_with_both_margins_at_least_multiple"].set_index(["axis", "ci_width_multiple"])
    assert (both.loc[("candidate", 2.0), "n"], both.loc[("candidate", 2.0), "k"]) == (40, 0)
    assert (both.loc[("tissue", 2.0), "n"], both.loc[("tissue", 2.0), "k"]) == (39, 0)
    assert (both.loc[("candidate", 1.0), "n"], both.loc[("candidate", 1.0), "k"]) == (58, 4)
    below = yard[yard["statistic"] == "reversals_with_a_margin_below_multiple"].set_index(["axis", "ci_width_multiple"])
    assert (below.loc[("candidate", 1.0), "k"], below.loc[("candidate", 1.0), "n"]) == (18, 22)
    assert (below.loc[("tissue", 1.0), "k"], below.loc[("tissue", 1.0), "n"]) == (24, 26)
    # A count below a smaller multiple can never exceed the count below a larger one.
    for axis in ("candidate", "tissue"):
        assert below.loc[axis]["k"].is_monotonic_increasing


def test_intraclass_correlation_extremes():
    groups = np.repeat(np.arange(5), 4)
    assert intraclass_correlation(np.repeat(np.arange(5.0), 4), groups) == pytest.approx(1.0)
    rng = np.random.default_rng(3)
    values = [intraclass_correlation(rng.normal(size=20), groups) for _ in range(400)]
    assert np.mean(values) == pytest.approx(0.0, abs=0.05)


def test_interval_widths_match_headline_intervals_and_roughly_double():
    widths = pd.read_csv(OUTPUTS / "extended_interval_widths.csv")
    headline = pd.read_csv(OUTPUTS / "protocol_headline_rates.csv").set_index("axis")
    for axis in ("candidate", "tissue", "reference"):
        rows = widths[widths["axis"] == axis].set_index("summary")
        assert rows.loc["pair_resampling", "low"] == pytest.approx(headline.loc[axis, "interval_low"])
        assert rows.loc["pair_resampling", "high"] == pytest.approx(headline.loc[axis, "interval_high"])
        assert rows.loc["pair_resampling", "width_ratio_to_pair"] == pytest.approx(1.0)
        assert 1.9 < rows.loc["method_resampling", "width_ratio_to_pair"] < 2.2


def test_power_variants_against_fixed_score_power():
    variants = pd.read_csv(OUTPUTS / "extended_screening_power_variants.csv")
    fixed = pd.read_csv(OUTPUTS / "extended_screening_power.csv").set_index("planted_auroc")
    assert set(variants["variant"]) == {"pair_shared_fixed_score", "trained_screen", "top_concentrated_fixed_score"}
    null = variants[variants["planted_auroc"] == 0.5]
    assert ((null["power_at_own_null_q95"] >= 0.05) & (null["power_at_own_null_q95"] <= 0.07)).all()
    # The null of every design is centred on chance (no pair-identity leakage).
    assert (null["mean_realized_auroc"].sub(0.5).abs() < 0.03).all()
    binormal = variants[variants["variant"].isin(["pair_shared_fixed_score", "trained_screen"])]
    for auroc in sorted(set(binormal["planted_auroc"]) & set(fixed.index)):
        subset = binormal[binormal["planted_auroc"] == auroc]
        assert (subset["power_at_own_null_q95"] <= fixed.loc[auroc, "power_at_random_q95"]).all()
    # Power at a given AUROC depends on the ROC shape: concentrating the signal at the top raises it.
    top = variants[variants["variant"] == "top_concentrated_fixed_score"].set_index("planted_auroc")
    assert top.loc[0.6, "power_at_own_null_q95"] > fixed.loc[0.6, "power_at_random_q95"]


def test_derived_scale_quantities():
    import json
    derived = json.loads((OUTPUTS / "extended_summary.json").read_text())["derived"]
    length = derived["zero_recovery_aupr_vs_list_length"]
    assert all(length[name]["all_zero_recovery"] and length[name]["n_lists"] == 9 for name in ("hpn_dream", "beeline_gsd"))
    assert length["hpn_dream"]["spearman_rho"] < 0 and length["beeline_gsd"]["spearman_rho"] < 0
    widths = derived["bootstrap_width_in_base_rate_units"]
    base = pd.read_csv(INPUTS / "table_a2.csv").groupby("candidate_set")["base_rate_median"].first()
    source = pd.read_csv(INPUTS / "source_bootstrap_ci_widths.csv").set_index("candidate_set")["median_aupr_ci_width"]
    for candidate, value in widths.items():
        assert value == pytest.approx(source[candidate] / base[candidate])

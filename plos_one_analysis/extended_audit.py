#!/usr/bin/env python3
"""Scale, informativeness, rounding, dependence and feasibility checks.

These analyses use only archived summary inputs. They ask whether observed
reversal margins are distinguishable from evaluation artifacts, how much the
reported fractions depend on shared methods, and what the small screening and
shift-bound designs can and cannot detect. Every random procedure is seeded.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))  # also works under python -I
from protocol_audit import AXIS_FILES, METRICS, row_weighted_cluster_interval  # noqa: E402


NORMALIZED_THRESHOLDS = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5, 1.0)
# 1/sqrt(2): a full 95% width W of one score is about 2 * 1.96 * sigma, and the
# difference of two independent scores has a 95% threshold 1.96 * sqrt(2) * sigma = W / sqrt(2).
DIFFERENCE_WIDTH_MULTIPLE = 1.0 / math.sqrt(2.0)
WIDTH_MULTIPLES = (0.25, 0.5, DIFFERENCE_WIDTH_MULTIPLE, 1.0, 2.0)
POWER_AUROCS = (0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
POWER_VARIANT_AUROCS = (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9)
BOUND_QUANTILES = (0.5, 0.75, 0.9, 1.0)
CANDIDATES = ["all_pairs", "tf_sources", "tf_sources_targets"]
SCREEN_FEATURES = ["abs_delta_from", "base_rate_from", "b_change", "abs_gap_from"]
# Scorers without stochastic fitting; tree ensembles (GENIE3, GRNBoost2) can differ between runs.
DETERMINISTIC_SCORERS = ("pearson", "random", "scgpt_attention", "spearman")
DEPENDENCE_AXES = ("candidate", "tissue", "reference")


# ---------------------------------------------------------------------------
# Reference endpoints: recovered edges and tie-aware average precision
# ---------------------------------------------------------------------------

def reference_endpoints(scores: pd.DataFrame, label_positives: dict[str, int] | None = None) -> pd.DataFrame:
    """Recover predicted reference edges from archived precision and recall.

    Precision times predicted-list length and recall times reference size
    must agree; a disagreement would mean the archived rows are not the
    counts they claim to be. ``base_rate`` is the reported rate (reference
    size over candidates). The reported reference sizes also count
    self-pairs, which the candidate set excludes, so the exact tie-aware
    average precision of a list without recovered edges is the label-vector
    rate (positive candidates over candidates); it is filled in when the
    label-vector positives of the export check are given.
    """
    endpoints = scores[["method", "reference", "precision", "recall", "aupr",
                        "pred_edges", "true_edges", "candidate_edges"]].copy()
    from_precision = (endpoints["precision"] * endpoints["pred_edges"]).round().astype(int)
    from_recall = (endpoints["recall"] * endpoints["true_edges"]).round().astype(int)
    if not from_precision.equals(from_recall):
        raise ValueError("recovered-edge counts from precision and recall disagree")
    endpoints["recovered_edges"] = from_precision
    endpoints["zero_recovery"] = endpoints["recovered_edges"].eq(0)
    endpoints["base_rate"] = endpoints["true_edges"] / endpoints["candidate_edges"]
    endpoints["aupr_over_base_rate"] = endpoints["aupr"] / endpoints["base_rate"]
    # A tie-aware average precision assigns a list without any recovered
    # reference edge exactly the label-vector base rate P/N (see
    # tie_aware_zero_recovery_ap).
    if label_positives is None:
        endpoints["label_vector_base_rate"] = np.nan
    else:
        endpoints["label_vector_base_rate"] = (endpoints["reference"].map(label_positives)
                                               / endpoints["candidate_edges"])
    endpoints["tie_aware_ap_if_zero_recovery"] = np.where(
        endpoints["zero_recovery"], endpoints["label_vector_base_rate"], np.nan)
    return endpoints.sort_values(["reference", "method"]).reset_index(drop=True)


def tie_aware_zero_recovery_ap(n_candidates: int, n_positive: int, n_predicted: int,
                               seed: int = 0) -> tuple[float, float]:
    """Return (tie-aware AP, base rate) for a list with no recovered positive.

    Predicted false positives receive distinct positive scores and all other
    candidates are tied at zero. scikit-learn evaluates the tied block as one
    threshold, so the only nonzero recall step has precision P/N.
    """
    rng = np.random.default_rng(seed)
    labels = np.zeros(n_candidates, dtype=np.int8)
    labels[rng.choice(n_candidates, n_positive, replace=False)] = 1
    negatives = np.flatnonzero(labels == 0)
    scores = np.zeros(n_candidates, dtype=np.float32)
    scores[rng.choice(negatives, n_predicted, replace=False)] = rng.uniform(0.01, 1.0, n_predicted)
    return float(average_precision_score(labels, scores)), n_positive / n_candidates


def source_full_ranking_aupr(scores: np.ndarray, labels: np.ndarray) -> float:
    """Trapezoidal full-ranking AUPR, as in the immune reference-panel source.

    Copied from network_inference/src/evaluation/metrics.py: every candidate
    is ranked, unscored candidates are tied at zero, and their order is set
    by numpy's default argsort.
    """
    order = np.argsort(-scores)
    labels = labels[order].astype(np.int64, copy=False)
    total_pos = int(labels.sum())
    if total_pos == 0:
        return 0.0
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    precisions = tp / (tp + fp)
    recalls = tp / total_pos
    return float(np.trapezoid(precisions, recalls) if hasattr(np, "trapezoid")
                 else np.trapz(precisions, recalls))


def reference_row_audit(rows: pd.DataFrame, endpoints: pd.DataFrame) -> pd.DataFrame:
    """Flag reference comparisons by endpoint informativeness and scale."""
    recovered = endpoints.set_index(["method", "reference"])["recovered_edges"]
    audited = rows.copy()
    for side in ("from", "to"):
        reference = audited["reference_" + side]
        tp_a = [recovered[(m, r)] for m, r in zip(audited["method_a"], reference)]
        tp_b = [recovered[(m, r)] for m, r in zip(audited["method_b"], reference)]
        audited["recovered_a_" + side] = tp_a
        audited["recovered_b_" + side] = tp_b
        audited["informative_" + side] = (np.array(tp_a) > 0) & (np.array(tp_b) > 0)
        audited["shared_zero_recovery_" + side] = (np.array(tp_a) == 0) & (np.array(tp_b) == 0)
        audited["normalized_margin_" + side] = (audited["delta_" + side].abs()
                                                / audited["base_rate_" + side])
    audited["n_informative_endpoints"] = (audited["informative_from"].astype(int)
                                          + audited["informative_to"].astype(int))
    audited["tied_under_tie_aware_ap"] = (audited["shared_zero_recovery_from"]
                                          | audited["shared_zero_recovery_to"])
    audited["min_normalized_margin"] = audited[["normalized_margin_from",
                                                "normalized_margin_to"]].min(axis=1)
    return audited


def reference_summary(audited: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for (first, second), group in audited.groupby(["reference_from", "reference_to"], sort=True):
        for n_informative, subset in group.groupby("n_informative_endpoints"):
            summaries.append({"reference_from": first, "reference_to": second,
                              "n_informative_endpoints": int(n_informative),
                              "n": len(subset), "k": int(subset["reversal"].sum())})
    for n_informative, subset in audited.groupby("n_informative_endpoints"):
        summaries.append({"reference_from": "all", "reference_to": "all",
                          "n_informative_endpoints": int(n_informative),
                          "n": len(subset), "k": int(subset["reversal"].sum())})
    summaries.append({"reference_from": "all", "reference_to": "all",
                      "n_informative_endpoints": -1, "n": len(audited),
                      "k": int(audited["reversal"].sum())})
    return pd.DataFrame(summaries)


# ---------------------------------------------------------------------------
# Common-scale margins
# ---------------------------------------------------------------------------

def normalized_axis_rows(inputs: Path) -> dict[str, pd.DataFrame]:
    """Express each endpoint margin in units of that endpoint's base rate."""
    base_rates = pd.read_csv(inputs / "table_a2.csv")
    if base_rates.groupby("candidate_set")["base_rate_median"].nunique().max() != 1:
        raise ValueError("candidate base rates differ across tissues; tissue normalization undefined")
    by_candidate = base_rates.groupby("candidate_set")["base_rate_median"].first()
    candidate = pd.read_csv(inputs / AXIS_FILES["candidate"])
    tissue = pd.read_csv(inputs / AXIS_FILES["tissue"])
    tissue["base_rate_from"] = tissue["candidate_set"].map(by_candidate)
    tissue["base_rate_to"] = tissue["base_rate_from"]
    reference = pd.read_csv(inputs / AXIS_FILES["reference"])
    frames = {"candidate": candidate, "tissue": tissue, "reference": reference}
    for frame in frames.values():
        frame["normalized_margin_from"] = frame["delta_from"].abs() / frame["base_rate_from"]
        frame["normalized_margin_to"] = frame["delta_to"].abs() / frame["base_rate_to"]
        frame["min_normalized_margin"] = frame[["normalized_margin_from",
                                                "normalized_margin_to"]].min(axis=1)
    return frames


def normalized_margin_sweep(frames: dict[str, pd.DataFrame], n_bootstrap: int,
                            seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    sweep, quantiles = [], []
    for axis, frame in frames.items():
        for threshold in NORMALIZED_THRESHOLDS:
            retained = frame[frame["min_normalized_margin"] >= threshold]
            summary = row_weighted_cluster_interval(retained, "reversal", ["method_pair"],
                                                    n_bootstrap, seed)
            sweep.append({"axis": axis, "threshold_base_rate_units": threshold,
                          "n": summary["n"], "k": summary["k"], "rate": summary["rate"],
                          "range_low": summary["interval_low"],
                          "range_high": summary["interval_high"],
                          "n_clusters": summary["n_clusters"]})
        for reversal, group in frame.groupby("reversal"):
            values = group["min_normalized_margin"]
            quantiles.append({"axis": axis, "reversal": bool(reversal), "n": len(group),
                              "min": values.min(), "q25": values.quantile(0.25),
                              "median": values.median(), "q75": values.quantile(0.75),
                              "max": values.max()})
    return pd.DataFrame(sweep), pd.DataFrame(quantiles)


def noise_yardstick(frames: dict[str, pd.DataFrame], widths: pd.DataFrame) -> pd.DataFrame:
    """Compare candidate/tissue margins with the source's bootstrap CI widths.

    The widths are per-candidate-set medians over methods, tissues and
    references, obtained by resampling predicted edges; they are a coarse
    yardstick, not a per-comparison standard error.
    """
    width = widths.set_index("candidate_set")["median_aupr_ci_width"]
    rows = []
    for axis in ("candidate", "tissue"):
        frame = frames[axis].copy()
        if axis == "candidate":
            frame["width_from"] = frame["shift_from"].map(width)
            frame["width_to"] = frame["shift_to"].map(width)
        else:
            frame["width_from"] = frame["candidate_set"].map(width)
            frame["width_to"] = frame["width_from"]
        frame["min_width_multiple"] = np.minimum(frame["delta_from"].abs() / frame["width_from"],
                                                 frame["delta_to"].abs() / frame["width_to"])
        for multiple in WIDTH_MULTIPLES:
            retained = frame[frame["min_width_multiple"] >= multiple]
            rows.append({"axis": axis, "statistic": "rows_with_both_margins_at_least_multiple",
                         "ci_width_multiple": multiple, "n": len(retained),
                         "k": int(retained["reversal"].sum()),
                         "rate": float(retained["reversal"].mean()) if len(retained) else np.nan})
        reversals = frame[frame["reversal"]]
        for multiple in WIDTH_MULTIPLES:
            below = reversals["min_width_multiple"] < multiple
            rows.append({"axis": axis, "statistic": "reversals_with_a_margin_below_multiple",
                         "ci_width_multiple": multiple, "n": len(reversals), "k": int(below.sum()),
                         "rate": float(below.mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metric comparison: rounding intervals and exact paired tests
# ---------------------------------------------------------------------------

def four_significant_half_unit(value: float) -> float:
    """Half the last-digit unit of a value printed with Python's ``.4g``."""
    if value == 0:
        return 0.0
    return 0.5 * 10 ** (math.floor(math.log10(abs(value))) - 3)


def metric_rounding_simulation(table_a1: pd.DataFrame, matched_keys: pd.DataFrame,
                               n_draws: int, seed: int,
                               fixed: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reversal counts when each printed median is any value that rounds to it.

    Exact zeros are kept fixed. Draws are independent uniform values within
    each rounding interval; this describes rounding ambiguity only. ``fixed``
    maps (tissue, method, metric) to a known unrounded value, which then
    replaces the printed value and is not redrawn.
    """
    tissues = sorted(table_a1["gene_set"].unique())
    methods = sorted(table_a1["prediction_method"].unique())
    keys = [(t1, t2, m1, m2) for t1, t2 in itertools.combinations(tissues, 2)
            for m1, m2 in itertools.combinations(methods, 2)]
    matched = {(r.tissue_from, r.tissue_to, r.method_a, r.method_b)
               for r in matched_keys.itertuples(index=False)}
    in_matched = np.array([key in matched for key in keys])
    lookup = {(t, m): i for i, (t, m) in enumerate(zip(table_a1["gene_set"],
                                                        table_a1["prediction_method"]))}
    index = np.array([[lookup[(t1, m1)], lookup[(t1, m2)], lookup[(t2, m1)], lookup[(t2, m2)]]
                      for t1, t2, m1, m2 in keys])
    rng = np.random.default_rng(seed)
    summaries, per_key = [], []
    for metric, column in METRICS.items():
        values = table_a1[column].to_numpy(dtype=float, copy=True)
        half_units = np.array([four_significant_half_unit(v) for v in values])
        if fixed:
            for i, cell in enumerate(zip(table_a1["gene_set"], table_a1["prediction_method"])):
                if (*cell, metric) in fixed:
                    values[i], half_units[i] = fixed[(*cell, metric)], 0.0
        draws = values + rng.uniform(-1.0, 1.0, size=(n_draws, len(values))) * half_units
        delta_from = draws[:, index[:, 0]] - draws[:, index[:, 1]]
        delta_to = draws[:, index[:, 2]] - draws[:, index[:, 3]]
        reversed_ = delta_from * delta_to < 0
        for scope, mask in [("all_keys", np.ones(len(keys), bool)),
                            ("matched_keys", in_matched)]:
            counts = reversed_[:, mask].sum(axis=1)
            distribution = ";".join(f"{value}:{share:.4f}" for value, share in
                                    zip(*np.unique(counts, return_counts=True)) for share in [share / n_draws])
            summaries.append({"metric": metric, "scope": scope, "n_keys": int(mask.sum()),
                              "mean_reversals": counts.mean(),
                              "q025": np.quantile(counts, 0.025), "median": np.median(counts),
                              "q975": np.quantile(counts, 0.975),
                              "p_zero": float((counts == 0).mean()),
                              "p_at_least_one": float((counts >= 1).mean()),
                              "p_at_least_two": float((counts >= 2).mean()),
                              "count_distribution": distribution})
        probabilities = reversed_.mean(axis=0)
        for key, probability, matched_key in zip(keys, probabilities, in_matched):
            per_key.append({"metric": metric, "tissue_from": key[0], "tissue_to": key[1],
                            "method_a": key[2], "method_b": key[3],
                            "in_matched_set": bool(matched_key),
                            "reversal_probability": float(probability)})
    return pd.DataFrame(summaries), pd.DataFrame(per_key)


def metric_exact_tests(matched: pd.DataFrame, table_a1: pd.DataFrame) -> dict:
    """Exact paired tests and binomial intervals on the matched 40 keys."""
    result = {"n_matched": int(len(matched)), "per_metric": {}, "paired": {}}
    for metric in METRICS:
        k = int(matched[metric].astype(bool).sum())
        interval = stats.binomtest(k, len(matched)).proportion_ci(method="exact")
        result["per_metric"][metric] = {"k": k, "clopper_pearson_low": interval.low,
                                        "clopper_pearson_high": interval.high}
    for first, second in itertools.combinations(METRICS, 2):
        a, b = matched[first].astype(bool), matched[second].astype(bool)
        first_only, second_only = int((a & ~b).sum()), int((~a & b).sum())
        discordant = first_only + second_only
        p_value = stats.binomtest(first_only, discordant, 0.5).pvalue if discordant else 1.0
        result["paired"][first + "_vs_" + second] = {
            "first_only": first_only, "second_only": second_only,
            "exact_mcnemar_two_sided_p": p_value}
    pair = table_a1[table_a1["prediction_method"].isin(["genie3", "grnboost2"])]
    margins = {}
    for metric, column in METRICS.items():
        wide = pair.pivot(index="gene_set", columns="prediction_method", values=column)
        margins[metric] = {tissue: float(wide.loc[tissue, "genie3"] - wide.loc[tissue, "grnboost2"])
                           for tissue in wide.index}
    result["genie3_minus_grnboost2"] = margins
    lung = pair[pair["gene_set"] == "lung"].set_index("prediction_method")["aupr_median"]
    result["lung_genie3_grnboost2_relative_aupr_margin"] = float(
        (lung["genie3"] - lung["grnboost2"]) / lung[["genie3", "grnboost2"]].mean())
    lung_f1 = pair[pair["gene_set"] == "lung"].set_index("prediction_method")["f1_median"]
    result["lung_genie3_grnboost2_relative_f1_margin"] = float(
        (lung_f1["genie3"] - lung_f1["grnboost2"]) / lung_f1[["genie3", "grnboost2"]].mean())
    return result


def release_value_check(table_a1: pd.DataFrame, release: pd.DataFrame) -> pd.DataFrame:
    """Which printed Table A1 medians the later public source release reproduces.

    A release median reproduces a printed value when it prints identically
    with four significant figures, the precision of the archived table. A
    release value is usable, that is, allowed to replace a printed value or
    break a printed tie, only for a deterministic scorer in a tissue where the
    release reproduces every deterministic-scorer median in all three metrics.
    Isolated agreement in a tissue whose other medians differ is treated as
    coincidence, and every other cell is left as printed.
    """
    merged = table_a1.merge(release, on=["gene_set", "prediction_method"],
                            suffixes=("_printed", "_release"), validate="one_to_one")
    if len(merged) != len(table_a1):
        raise ValueError("release table does not cover every Table A1 tissue/method cell")
    rows = []
    for metric, column in METRICS.items():
        for record in merged.to_dict("records"):
            printed, release_value = float(record[column + "_printed"]), float(record[column + "_release"])
            rounded = float(f"{release_value:.4g}")
            rows.append({"gene_set": record["gene_set"], "prediction_method": record["prediction_method"],
                         "metric": metric, "printed": printed, "release": release_value,
                         "release_four_significant": rounded, "reproduced": rounded == printed,
                         "deterministic_scorer": record["prediction_method"] in DETERMINISTIC_SCORERS})
    check = pd.DataFrame(rows)
    validated = check[check["deterministic_scorer"]].groupby("gene_set")["reproduced"].all()
    check["tissue_validated"] = check["gene_set"].map(validated).astype(bool)
    check["usable"] = check["reproduced"] & check["deterministic_scorer"] & check["tissue_validated"]
    return check


def _metric_keys(table_a1: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    tissues = sorted(table_a1["gene_set"].unique())
    methods = sorted(table_a1["prediction_method"].unique())
    return [(t1, t2, m1, m2) for t1, t2 in itertools.combinations(tissues, 2)
            for m1, m2 in itertools.combinations(methods, 2)]


def metric_tie_resolution(table_a1: pd.DataFrame, check: pd.DataFrame) -> pd.DataFrame:
    """Printed AUROC ties, and whether reproduced release values break them.

    An endpoint is resolved when both methods' AUROC there has a usable
    release value (see release_value_check); a key is resolved when both of
    its endpoints are.
    """
    printed = {(r.gene_set, r.prediction_method, r.metric): r.printed for r in check.itertuples(index=False)}
    known = {(r.gene_set, r.prediction_method, r.metric): r.release
             for r in check.itertuples(index=False) if r.usable}
    rows = []
    for t1, t2, m1, m2 in _metric_keys(table_a1):
        printed_from = printed[(t1, m1, "AUROC")] - printed[(t1, m2, "AUROC")]
        printed_to = printed[(t2, m1, "AUROC")] - printed[(t2, m2, "AUROC")]
        if printed_from != 0 and printed_to != 0:
            continue
        unrounded = {}
        for side, tissue in (("from", t1), ("to", t2)):
            a, b = known.get((tissue, m1, "AUROC")), known.get((tissue, m2, "AUROC"))
            unrounded[side] = np.nan if a is None or b is None else a - b
        resolved = bool(np.isfinite(unrounded["from"]) and np.isfinite(unrounded["to"]))
        rows.append({"tissue_from": t1, "tissue_to": t2, "method_a": m1, "method_b": m2,
                     "printed_auroc_a_from": printed[(t1, m1, "AUROC")],
                     "printed_auroc_b_from": printed[(t1, m2, "AUROC")],
                     "printed_auroc_a_to": printed[(t2, m1, "AUROC")],
                     "printed_auroc_b_to": printed[(t2, m2, "AUROC")],
                     "printed_delta_from": printed_from, "printed_delta_to": printed_to,
                     "release_delta_from": unrounded["from"], "release_delta_to": unrounded["to"],
                     "resolved": resolved,
                     "release_reversal": bool(unrounded["from"] * unrounded["to"] < 0) if resolved else None})
    return pd.DataFrame(rows)


def resolved_metric_comparison(table_a1: pd.DataFrame, check: pd.DataFrame,
                               printed_matched: pd.DataFrame) -> dict:
    """Matched AUPR/AUROC/F1 comparison after substituting usable release values.

    Cells with a usable release value (see release_value_check) take that
    unrounded value and every other cell keeps its printed value, so a key
    enters the matched set only if no endpoint is tied under any metric after
    substitution.
    """
    values = table_a1.set_index(["gene_set", "prediction_method"])
    values = values[[column for column in METRICS.values()]].astype(float).copy()
    for r in check[check["usable"]].itertuples(index=False):
        values.loc[(r.gene_set, r.prediction_method), METRICS[r.metric]] = r.release
    records = []
    for t1, t2, m1, m2 in _metric_keys(table_a1):
        record = {"tissue_from": t1, "tissue_to": t2, "method_a": m1, "method_b": m2}
        for metric, column in METRICS.items():
            delta_from = values.loc[(t1, m1), column] - values.loc[(t1, m2), column]
            delta_to = values.loc[(t2, m1), column] - values.loc[(t2, m2), column]
            record[metric + "_eligible"] = bool(delta_from != 0 and delta_to != 0
                                                and np.isfinite(delta_from) and np.isfinite(delta_to))
            record[metric] = bool(delta_from * delta_to < 0)
        records.append(record)
    keys = pd.DataFrame(records)
    eligible = keys[[metric + "_eligible" for metric in METRICS]].all(axis=1)
    matched = keys[eligible]
    before = {(r.tissue_from, r.tissue_to, r.method_a, r.method_b) for r in printed_matched.itertuples(index=False)}
    after = {(r.tissue_from, r.tissue_to, r.method_a, r.method_b) for r in matched.itertuples(index=False)}
    result = {"n_keys": int(len(keys)), "n_matched": int(len(matched)),
              "added_keys": ["/".join(key) for key in sorted(after - before)],
              "removed_keys": ["/".join(key) for key in sorted(before - after)],
              "per_metric": {}, "paired": {}}
    for metric in METRICS:
        usable = keys[keys[metric + "_eligible"]]
        k = int(matched[metric].sum())
        interval = stats.binomtest(k, len(matched)).proportion_ci(method="exact")
        result["per_metric"][metric] = {"k_matched": k, "n_eligible_alone": int(len(usable)),
                                        "k_eligible_alone": int(usable[metric].sum()),
                                        "clopper_pearson_low": interval.low,
                                        "clopper_pearson_high": interval.high}
    for first, second in itertools.combinations(METRICS, 2):
        a, b = matched[first], matched[second]
        first_only, second_only = int((a & ~b).sum()), int((~a & b).sum())
        discordant = first_only + second_only
        result["paired"][first + "_vs_" + second] = {
            "first_only": first_only, "second_only": second_only,
            "exact_mcnemar_two_sided_p": stats.binomtest(first_only, discordant, 0.5).pvalue if discordant else 1.0}
    return result


# ---------------------------------------------------------------------------
# Shared-method dependence
# ---------------------------------------------------------------------------

def method_level_resampling(comparisons: pd.DataFrame, n_bootstrap: int, seed: int) -> dict:
    """Resample whole methods with replacement and keep every induced pair.

    A sampled method appearing c times contributes each of its pair clusters
    with multiplicity c_a * c_b; pairs of a method with itself do not exist.
    This propagates the dependence created when distinct pairs share a
    method, which pair resampling ignores.
    """
    methods = sorted(set(comparisons["method_a"]) | set(comparisons["method_b"]))
    position = {method: i for i, method in enumerate(methods)}
    events = np.zeros((len(methods), len(methods)))
    rows = np.zeros_like(events)
    for a, b, flag in comparisons[["method_a", "method_b", "reversal"]].itertuples(index=False):
        i, j = position[a], position[b]
        events[i, j] += flag
        events[j, i] += flag
        rows[i, j] += 1
        rows[j, i] += 1
    rng = np.random.default_rng(seed)
    rates, degenerate = [], 0
    for _ in range(n_bootstrap):
        counts = np.bincount(rng.integers(0, len(methods), len(methods)), minlength=len(methods))
        weights = np.outer(counts, counts).astype(float)
        np.fill_diagonal(weights, 0.0)
        n = (weights * rows).sum() / 2
        if n == 0:
            degenerate += 1
            continue
        rates.append((weights * events).sum() / 2 / n)
    rates = np.array(rates)
    return {"n_methods": len(methods), "rate": float(comparisons["reversal"].mean()),
            "range_low": float(np.quantile(rates, 0.025)),
            "range_high": float(np.quantile(rates, 0.975)),
            "mean_draw": float(rates.mean()), "degenerate_draws": degenerate,
            "pairs_per_method": len(methods) - 1,
            "pair_clusters": len(methods) * (len(methods) - 1) // 2}


def leave_two_methods_out(comparisons: pd.DataFrame) -> pd.DataFrame:
    methods = sorted(set(comparisons["method_a"]) | set(comparisons["method_b"]))
    rows = []
    for omitted in itertools.combinations(methods, 2):
        retained = comparisons[~comparisons["method_a"].isin(omitted)
                               & ~comparisons["method_b"].isin(omitted)]
        rows.append({"omitted_methods": "+".join(omitted), "n": len(retained),
                     "k": int(retained["reversal"].sum()),
                     "rate": float(retained["reversal"].mean())})
    return pd.DataFrame(rows)


def interval_widths(outputs: Path, method_level: pd.DataFrame, leave_two: pd.DataFrame) -> pd.DataFrame:
    """Width of every dependence summary and its ratio to the pair-resampling width.

    Percentile intervals come from the conditional resampling schemes and
    ranges from the deletion summaries. The ratio to the method-pair
    interval, the interval printed with the primary fractions, shows how much
    each alternative dependence structure widens it.
    """
    clusters = pd.read_csv(outputs / "protocol_cluster_sensitivity.csv")
    leave_one = pd.read_csv(outputs / "protocol_leave_one_method_out.csv")
    leave_tissue = pd.read_csv(outputs / "protocol_leave_one_tissue_out.csv")
    headline = pd.read_csv(outputs / "protocol_headline_rates.csv").set_index("axis")
    schemes = {"unordered_method_pair": "pair_resampling", "transition_cell": "transition_cell_resampling",
               "tissue_three_clusters": "tissue_cluster_resampling",
               "tissue_transition_three_clusters": "tissue_cluster_resampling"}
    rows = []
    for axis in DEPENDENCE_AXES:
        found = clusters[clusters["axis"] == axis]
        pair = found[found["scheme"] == "unordered_method_pair"].iloc[0]
        if not (np.isclose(pair["interval_low"], headline.loc[axis, "interval_low"])
                and np.isclose(pair["interval_high"], headline.loc[axis, "interval_high"])):
            raise ValueError(f"{axis}: pair-resampling interval differs from the headline table")
        pair_width = pair["interval_high"] - pair["interval_low"]
        entries = [(schemes[r.scheme], "percentile interval", r.n_clusters, r.interval_low, r.interval_high)
                   for r in found.itertuples(index=False)]
        method = method_level[method_level["axis"] == axis].iloc[0]
        entries.append(("method_resampling", "percentile interval", method["n_methods"],
                        method["range_low"], method["range_high"]))
        for name, table, unit in (("leave_one_method_out", leave_one, "omitted_method"),
                                  ("leave_two_methods_out", leave_two, "omitted_methods"),
                                  ("leave_one_tissue_out", leave_tissue, "omitted_tissue")):
            subset = table[table["axis"] == axis]
            if len(subset):
                entries.append((name, "deletion range", len(subset), subset["rate"].min(), subset["rate"].max()))
        for summary, kind, units, low, high in entries:
            rows.append({"axis": axis, "summary": summary, "kind": kind, "units": int(units),
                         "k": int(headline.loc[axis, "k"]), "n": int(headline.loc[axis, "n"]),
                         "rate": float(headline.loc[axis, "rate"]), "low": float(low), "high": float(high),
                         "width": float(high - low), "width_ratio_to_pair": float((high - low) / pair_width)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Screening feasibility
# ---------------------------------------------------------------------------

def screening_frame(candidate_rows: pd.DataFrame) -> pd.DataFrame:
    frame = candidate_rows.copy()
    frame["abs_delta_from"] = frame["delta_from"].abs()
    frame["b_change"] = frame["base_rate_to"] - frame["base_rate_from"]
    frame["abs_gap_from"] = frame["gap_from"].abs()
    return frame


def held_out_logistic_from_features(features: np.ndarray, labels: np.ndarray,
                                    tissues: np.ndarray) -> np.ndarray:
    """Balanced logistic scores, each tissue scored by a model fitted on the others."""
    scores = np.zeros(len(labels))
    for tissue in sorted(set(tissues)):
        train, test = tissues != tissue, tissues == tissue
        mean, sd = features[train].mean(axis=0), features[train].std(axis=0) + 1e-12
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        model.fit((features[train] - mean) / sd, labels[train].astype(int))
        scores[test] = model.predict_proba((features[test] - mean) / sd)[:, 1]
    return scores


def held_out_logistic_scores(frame: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """Leave-one-tissue-out logistic scores, identical to protocol_audit."""
    return held_out_logistic_from_features(frame[SCREEN_FEATURES].to_numpy(), labels,
                                           frame["gene_set"].to_numpy())


def intraclass_correlation(values: np.ndarray, groups: np.ndarray) -> float:
    """One-way random-effects ICC(1) for groups of equal size."""
    frame = pd.DataFrame({"value": values, "group": groups})
    sizes = frame.groupby("group").size()
    if sizes.nunique() != 1:
        raise ValueError("ICC(1) here assumes equal group sizes")
    k, n_groups = float(sizes.iloc[0]), len(sizes)
    means = frame.groupby("group")["value"].transform("mean")
    between = k * ((frame.groupby("group")["value"].mean() - frame["value"].mean()) ** 2).sum() / (n_groups - 1)
    within = ((frame["value"] - means) ** 2).sum() / (len(frame) - n_groups)
    return float((between - within) / (between + (k - 1) * within))


def screening_power_variants(candidate_rows: pd.DataFrame, n_draws: int,
                             seed: int) -> tuple[pd.DataFrame, dict]:
    """Detection power for planted signals other than an independent binormal score.

    The fixed-score power in screening_feasibility assumes a binormal score
    that is independent across rows and needs no fitting. Two features of the
    actual design lower it: rows of the same method pair share score
    variation (pair-shared design; the shared fraction of variance is the
    intraclass correlation of the initial-margin heuristic's ranks within
    method pairs), and a screen must learn which feature carries the signal
    (trained design: one informative binormal feature among four, refitted
    on each pair of training tissues). A third design shows that power at a
    given AUROC also depends on the shape of the ROC curve: a fixed score
    that ranks a random fraction 2(AUROC - 0.5) of reversals above every
    other row and leaves the rest at random (top-concentrated design).
    A trained design with pair-shared noise is not used: a noise component
    shared across tissues identifies method pairs, so leave-one-tissue-out
    fitting would learn the recurrence of reversals within pairs even under
    the null. Each design is calibrated by its own null (planted AUROC 0.5);
    power is the probability of reaching that null's 95th percentile of
    pooled average precision.
    """
    frame = screening_frame(candidate_rows)
    labels = frame["reversal"].to_numpy().astype(int)
    tissues = frame["gene_set"].to_numpy()
    pairs = pd.factorize(frame["method_pair"], sort=True)[0]
    rho = intraclass_correlation(stats.rankdata(-frame["abs_delta_from"].to_numpy()), pairs)
    variants = {"pair_shared_fixed_score": ("fixed", rho),
                "trained_screen": ("trained", 0.0),
                "top_concentrated_fixed_score": ("top", 0.0)}
    rng = np.random.default_rng(seed)
    rows = []
    for variant, (kind, shared) in variants.items():
        trained = kind == "trained"
        n_features = len(SCREEN_FEATURES) if trained else 1
        results = {}
        for auroc in POWER_VARIANT_AUROCS:
            separation = math.sqrt(2.0) * stats.norm.ppf(auroc)
            top_fraction = 2.0 * (auroc - 0.5)
            ap, realized = np.empty(n_draws), np.empty(n_draws)
            for draw in range(n_draws):
                features = rng.normal(size=(len(labels), n_features))
                if shared > 0:
                    common = rng.normal(size=(pairs.max() + 1, n_features))[pairs]
                    features = math.sqrt(shared) * common + math.sqrt(1.0 - shared) * features
                if kind == "top":
                    # Lifted reversals rank above every row; expected AUROC is 0.5 + top_fraction / 2.
                    lifted = (labels == 1) & (rng.random(len(labels)) < top_fraction)
                    features[lifted, 0] += 1000.0
                else:
                    features[:, 0] += separation * labels
                score = (held_out_logistic_from_features(features, labels, tissues) if trained
                         else features[:, 0])
                ap[draw] = average_precision_score(labels, score)
                realized[draw] = roc_auc_score(labels, score)
            results[auroc] = (ap, realized)
        threshold = float(np.quantile(results[0.5][0], 0.95))
        for auroc, (ap, realized) in results.items():
            rows.append({"variant": variant, "planted_auroc": auroc, "trained": trained, "signal": "top" if kind == "top" else "binormal",
                         "within_pair_shared_variance": shared, "n_features": n_features,
                         "n_draws": n_draws, "null_q95_ap": threshold, "mean_ap": float(ap.mean()),
                         "mean_realized_auroc": float(realized.mean()),
                         "power_at_own_null_q95": float((ap >= threshold).mean())})
    meta = {"within_pair_icc_initial_margin_ranks": rho, "n_draws": n_draws, "seed": seed,
            "n_method_pairs": int(pairs.max() + 1),
            "label_icc_within_method_pairs": intraclass_correlation(labels.astype(float), pairs)}
    return pd.DataFrame(rows), meta


def screening_feasibility(candidate_rows: pd.DataFrame, n_permutations: int, n_power: int,
                          seed: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Permutation nulls and planted-signal power for the held-out-tissue screens.

    Labels are permuted within each held-out tissue, which preserves fold
    prevalence. Power uses a fixed planted score with binormal AUROC; it is an
    upper bound for a trained screen, whose fitting adds variance.
    """
    frame = screening_frame(candidate_rows)
    labels = frame["reversal"].to_numpy().astype(int)
    tissues = frame["gene_set"].to_numpy()
    heuristic = -frame["abs_delta_from"].to_numpy()
    logistic = held_out_logistic_scores(frame, labels)
    observed = {"logistic": average_precision_score(labels, logistic),
                "heuristic": average_precision_score(labels, heuristic)}
    rng = np.random.default_rng(seed)
    null = {"logistic": [], "heuristic": []}
    for _ in range(n_permutations):
        permuted = labels.copy()
        for tissue in sorted(set(tissues)):
            mask = tissues == tissue
            permuted[mask] = rng.permutation(permuted[mask])
        null["logistic"].append(average_precision_score(permuted, held_out_logistic_scores(frame, permuted)))
        null["heuristic"].append(average_precision_score(permuted, heuristic))
    prevalence = float(labels.mean())
    summary = {"n_rows": int(len(labels)), "n_positive": int(labels.sum()), "prevalence": prevalence,
               "n_permutations": n_permutations, "seed": seed}
    for name in ("logistic", "heuristic"):
        draws = np.array(null[name])
        summary[name] = {
            "observed_ap": float(observed[name]),
            "ap_over_prevalence": float(observed[name] / prevalence),
            "observed_auroc": float(roc_auc_score(labels, logistic if name == "logistic" else heuristic)),
            "null_mean_ap": float(draws.mean()), "null_q05": float(np.quantile(draws, 0.05)),
            "null_q95": float(np.quantile(draws, 0.95)),
            "plus_one_p_value": float((1 + (draws >= observed[name]).sum()) / (1 + len(draws))),
        }
    random_ap = np.array([average_precision_score(labels, rng.normal(size=len(labels)))
                          for _ in range(n_permutations)])
    threshold = float(np.quantile(random_ap, 0.95))
    summary["random_ranking_ap"] = {"mean": float(random_ap.mean()), "q95": threshold}
    power_rows = []
    for auroc in POWER_AUROCS:
        separation = math.sqrt(2.0) * stats.norm.ppf(auroc)
        draws = np.array([average_precision_score(labels, rng.normal(size=len(labels)) + separation * labels)
                          for _ in range(n_power)])
        power_rows.append({"planted_auroc": auroc, "mean_ap": float(draws.mean()),
                           "median_ap": float(np.median(draws)),
                           "power_at_random_q95": float((draws >= threshold).mean())})
    null_draws = pd.DataFrame({"permutation": np.arange(n_permutations),
                               "logistic_ap": null["logistic"], "heuristic_ap": null["heuristic"]})
    return summary, pd.DataFrame(power_rows), null_draws


# ---------------------------------------------------------------------------
# Held-out shift bound
# ---------------------------------------------------------------------------

def held_out_shift_bound(candidate_rows: pd.DataFrame) -> pd.DataFrame:
    """Estimate |delta_to - delta_from| bounds from two tissues, test on the third.

    A row is certified when its initial margin exceeds the bound. A valid
    bound would make every certified row non-reversing.
    """
    frame = candidate_rows.copy()
    frame["shift"] = (frame["delta_to"] - frame["delta_from"]).abs()
    schemes = [("transition_quantile_" + str(q), ["shift_from", "shift_to"], q) for q in BOUND_QUANTILES]
    schemes.append(("pair_transition_max", ["shift_from", "shift_to", "method_pair"], 1.0))
    rows = []
    for scheme, group_columns, quantile in schemes:
        for tissue in sorted(frame["gene_set"].unique()):
            train = frame[frame["gene_set"] != tissue]
            test = frame[frame["gene_set"] == tissue]
            bound = train.groupby(group_columns)["shift"].quantile(quantile).rename("bound")
            scored = test.join(bound, on=group_columns)
            certified = scored["delta_from"].abs() > scored["bound"]
            rows.append({"scheme": scheme, "held_tissue": tissue, "n_test": len(scored),
                         "bound_holds": int((scored["shift"] <= scored["bound"]).sum()),
                         "certified": int(certified.sum()),
                         "reversals_certified": int((certified & scored["reversal"]).sum()),
                         "reversals_total": int(scored["reversal"].sum())})
    table = pd.DataFrame(rows)
    totals = table.groupby("scheme", sort=False)[["n_test", "bound_holds", "certified",
                                                  "reversals_certified", "reversals_total"]].sum().reset_index()
    totals["held_tissue"] = "all"
    totals["coverage"] = totals["bound_holds"] / totals["n_test"]
    table["coverage"] = table["bound_holds"] / table["n_test"]
    return pd.concat([table, totals], ignore_index=True)


# ---------------------------------------------------------------------------
# Small derived quantities quoted in the text
# ---------------------------------------------------------------------------

def derived_quantities(inputs: Path, endpoints: pd.DataFrame) -> dict:
    """Chance recovery, random-score scale, and method-resampling design facts."""
    check = json.loads((inputs / "immune_reference_panel" / "export_check.json").read_text())
    n_candidates = int(check["n_candidates"])
    positives = {name: int(values["label_vector_positives"]) for name, values in check["references"].items()}
    longest = int(endpoints["pred_edges"].max())
    random_length = int(endpoints.loc[endpoints["method"] == "random", "pred_edges"].iloc[0])
    base = pd.read_csv(inputs / "table_a2.csv")
    medians = pd.read_csv(inputs / "table_a3.csv")
    random_scores = medians[medians["prediction_method"] == "random"].merge(
        base[["gene_set", "candidate_set", "base_rate_median"]], on=["gene_set", "candidate_set"])
    ratio = random_scores["aupr_median"] / random_scores["base_rate_median"]
    draws = list(itertools.product(range(6), repeat=6))
    # At an endpoint that recovers nothing, AUPR can depend only on list length and tie order.
    length_dependence = {}
    for reference in ("hpn_dream", "beeline_gsd"):
        lists = endpoints[endpoints["reference"] == reference]
        rho = stats.spearmanr(lists["pred_edges"], lists["aupr"])
        length_dependence[reference] = {"spearman_rho": float(rho.statistic), "p_value": float(rho.pvalue),
                                        "n_lists": int(len(lists)),
                                        "all_zero_recovery": bool(lists["zero_recovery"].all())}
    widths = pd.read_csv(inputs / "source_bootstrap_ci_widths.csv").set_index("candidate_set")["median_aupr_ci_width"]
    rates = base.groupby("candidate_set")["base_rate_median"].first()
    return {
        "longest_list_pairs": longest,
        "expected_chance_recovery_longest_list": {name: longest * positives[name] / n_candidates
                                                  for name in ("hpn_dream", "beeline_gsd")},
        "expected_chance_recovery_random_list_dorothea": random_length * positives["dorothea_trrust_union_immune"] / n_candidates,
        "random_score_median_over_median_base_rate": {"min": float(ratio.min()), "max": float(ratio.max())},
        "probability_six_method_draw_has_at_most_three_distinct": sum(len(set(d)) <= 3 for d in draws) / len(draws),
        "zero_recovery_aupr_vs_list_length": length_dependence,
        "bootstrap_width_in_base_rate_units": {candidate: float(widths[candidate] / rates[candidate])
                                               for candidate in CANDIDATES},
    }


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--input-dir", type=Path, default=here / "inputs")
    parser.add_argument("--output-dir", type=Path, default=here / "outputs")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--n-rounding", type=int, default=10000)
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument("--n-power", type=int, default=4000)
    parser.add_argument("--n-power-variants", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    inputs, outputs = arguments.input_dir, arguments.output_dir
    outputs.mkdir(parents=True, exist_ok=True)

    export_check = json.loads((inputs / "immune_reference_panel" / "export_check.json").read_text())
    label_positives = {name: int(values["label_vector_positives"])
                       for name, values in export_check["references"].items()}
    endpoints = reference_endpoints(pd.read_csv(inputs / "score_eval_grn_baselines_immune.csv"), label_positives)
    endpoints.to_csv(outputs / "extended_reference_endpoints.csv", index=False)
    audited = reference_row_audit(pd.read_csv(inputs / AXIS_FILES["reference"]), endpoints)
    audited.to_csv(outputs / "extended_reference_rows.csv", index=False)
    reference_summary(audited).to_csv(outputs / "extended_reference_summary.csv", index=False)
    zero_ap, zero_base = tie_aware_zero_recovery_ap(200000, 50, 5000, arguments.seed)

    frames = normalized_axis_rows(inputs)
    sweep, margin_quantiles = normalized_margin_sweep(frames, arguments.n_bootstrap, arguments.seed)
    sweep.to_csv(outputs / "extended_normalized_margin_sweep.csv", index=False)
    margin_quantiles.to_csv(outputs / "extended_normalized_margin_quantiles.csv", index=False)
    widths = pd.read_csv(inputs / "source_bootstrap_ci_widths.csv")
    noise_yardstick(frames, widths).to_csv(outputs / "extended_noise_yardstick.csv", index=False)

    table_a1 = pd.read_csv(inputs / "table_a1.csv")
    matched = pd.read_csv(outputs / "protocol_metric_matched_rows.csv")
    rounding, rounding_keys = metric_rounding_simulation(table_a1, matched, arguments.n_rounding, arguments.seed)
    rounding.to_csv(outputs / "extended_metric_rounding_summary.csv", index=False)
    rounding_keys.to_csv(outputs / "extended_metric_rounding_keys.csv", index=False)
    exact_tests = metric_exact_tests(matched, table_a1)

    release = pd.read_csv(inputs / "source_release_method_summary_by_tissue.csv")
    release_check = release_value_check(table_a1, release)
    release_check.to_csv(outputs / "extended_release_value_check.csv", index=False)
    ties = metric_tie_resolution(table_a1, release_check)
    ties.to_csv(outputs / "extended_metric_tie_resolution.csv", index=False)
    fixed = {(r.gene_set, r.prediction_method, r.metric): r.release
             for r in release_check[release_check["usable"]].itertuples(index=False)}
    conditional, conditional_keys = metric_rounding_simulation(table_a1, matched, arguments.n_rounding,
                                                               arguments.seed, fixed=fixed)
    conditional.to_csv(outputs / "extended_metric_rounding_conditional_summary.csv", index=False)
    conditional_keys.to_csv(outputs / "extended_metric_rounding_conditional_keys.csv", index=False)
    resolved = resolved_metric_comparison(table_a1, release_check, matched)
    reproduced = release_check.groupby(["deterministic_scorer", release_check["gene_set"] == "immune"])["reproduced"]
    release_summary = {
        "cells": int(len(release_check)), "reproduced": int(release_check["reproduced"].sum()),
        "usable": int(release_check["usable"].sum()),
        "validated_tissues": sorted(release_check.loc[release_check["tissue_validated"], "gene_set"].unique()),
        "deterministic_kidney_lung": {"cells": int(reproduced.size().loc[(True, False)]),
                                      "reproduced": int(reproduced.sum().loc[(True, False)])},
        "tree_ensemble_kidney_lung": {"cells": int(reproduced.size().loc[(False, False)]),
                                      "reproduced": int(reproduced.sum().loc[(False, False)])},
        "immune": {"cells": int((release_check["gene_set"] == "immune").sum()),
                   "reproduced": int(release_check.loc[release_check["gene_set"] == "immune", "reproduced"].sum())},
        "tied_auroc_keys": int(len(ties)), "resolved_tied_keys": int(ties["resolved"].sum()),
        "resolved_tied_keys_reversing": int(ties.loc[ties["resolved"], "release_reversal"].astype(bool).sum()),
        "resolved_matched_comparison": resolved,
    }

    dependence, leave_two = [], []
    for axis, filename in AXIS_FILES.items():
        comparisons = pd.read_csv(inputs / filename)
        dependence.append({"axis": axis, **method_level_resampling(comparisons, arguments.n_bootstrap,
                                                                   arguments.seed)})
        dropped = leave_two_methods_out(comparisons)
        dropped.insert(0, "axis", axis)
        leave_two.append(dropped)
    pd.DataFrame(dependence).to_csv(outputs / "extended_method_level_resampling.csv", index=False)
    pd.concat(leave_two).to_csv(outputs / "extended_leave_two_methods_out.csv", index=False)
    interval_widths(outputs, pd.DataFrame(dependence), pd.concat(leave_two)).to_csv(
        outputs / "extended_interval_widths.csv", index=False)

    candidate = pd.read_csv(inputs / AXIS_FILES["candidate"])
    screening, power, null_draws = screening_feasibility(candidate, arguments.n_permutations,
                                                         arguments.n_power, arguments.seed)
    power.to_csv(outputs / "extended_screening_power.csv", index=False)
    null_draws.to_csv(outputs / "extended_screening_null_draws.csv", index=False)
    variants, variant_meta = screening_power_variants(candidate, arguments.n_power_variants, arguments.seed)
    variants.to_csv(outputs / "extended_screening_power_variants.csv", index=False)
    screening["power_variants"] = variant_meta
    held_out_shift_bound(candidate).to_csv(outputs / "extended_heldout_shift_bound.csv", index=False)

    derived = derived_quantities(inputs, endpoints)
    summary = {
        "seed": arguments.seed, "n_bootstrap": arguments.n_bootstrap,
        "n_rounding_draws": arguments.n_rounding, "n_permutations": arguments.n_permutations,
        "n_power_draws": arguments.n_power, "n_power_variant_draws": arguments.n_power_variants,
        "reference": {
            "zero_recovery_endpoints": endpoints.groupby("reference")["zero_recovery"].sum().astype(int).to_dict(),
            "rows": int(len(audited)), "reversals": int(audited["reversal"].sum()),
            "rows_with_two_informative_endpoints": int((audited["n_informative_endpoints"] == 2).sum()),
            "rows_tied_under_tie_aware_ap": int(audited["tied_under_tie_aware_ap"].sum()),
            "max_min_normalized_margin_reversal": float(audited.loc[audited["reversal"], "min_normalized_margin"].max()),
            "tie_aware_ap_synthetic_check": {"ap": zero_ap, "base_rate": zero_base},
        },
        "metric_exact_tests": exact_tests,
        "release_value_check": release_summary,
        "derived": derived,
        "screening": screening,
        "interpretation": ("descriptive sensitivity summaries of archived summary scores; intervals are "
                           "conditional on the observed grid and are not confidence intervals"),
    }
    (outputs / "extended_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

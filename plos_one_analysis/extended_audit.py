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
WIDTH_MULTIPLES = (0.25, 0.5, 1.0, 2.0)
POWER_AUROCS = (0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
BOUND_QUANTILES = (0.5, 0.75, 0.9, 1.0)
CANDIDATES = ["all_pairs", "tf_sources", "tf_sources_targets"]
SCREEN_FEATURES = ["abs_delta_from", "base_rate_from", "b_change", "abs_gap_from"]


# ---------------------------------------------------------------------------
# Reference endpoints: recovered edges and tie-aware average precision
# ---------------------------------------------------------------------------

def reference_endpoints(scores: pd.DataFrame) -> pd.DataFrame:
    """Recover predicted reference edges from archived precision and recall.

    Precision times predicted-list length and recall times reference size
    must agree; a disagreement would mean the archived rows are not the
    counts they claim to be.
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
    # reference edge exactly the base rate (see tie_aware_zero_recovery_ap).
    endpoints["tie_aware_ap_if_zero_recovery"] = np.where(
        endpoints["zero_recovery"], endpoints["base_rate"], np.nan)
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
        below = reversals["min_width_multiple"] < 1
        rows.append({"axis": axis, "statistic": "reversals_with_a_margin_below_one_width",
                     "ci_width_multiple": 1.0, "n": len(reversals), "k": int(below.sum()),
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
                               n_draws: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reversal counts when each printed median is any value that rounds to it.

    Exact zeros are kept fixed. Draws are independent uniform values within
    each rounding interval; this describes rounding ambiguity only.
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
        values = table_a1[column].to_numpy(dtype=float)
        half_units = np.array([four_significant_half_unit(v) for v in values])
        draws = values + rng.uniform(-1.0, 1.0, size=(n_draws, len(values))) * half_units
        delta_from = draws[:, index[:, 0]] - draws[:, index[:, 1]]
        delta_to = draws[:, index[:, 2]] - draws[:, index[:, 3]]
        reversed_ = delta_from * delta_to < 0
        for scope, mask in [("all_keys", np.ones(len(keys), bool)),
                            ("matched_keys", in_matched)]:
            counts = reversed_[:, mask].sum(axis=1)
            summaries.append({"metric": metric, "scope": scope, "n_keys": int(mask.sum()),
                              "mean_reversals": counts.mean(),
                              "q025": np.quantile(counts, 0.025), "median": np.median(counts),
                              "q975": np.quantile(counts, 0.975),
                              "p_zero": float((counts == 0).mean()),
                              "p_at_least_one": float((counts >= 1).mean()),
                              "p_at_least_two": float((counts >= 2).mean())})
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


# ---------------------------------------------------------------------------
# Screening feasibility
# ---------------------------------------------------------------------------

def screening_frame(candidate_rows: pd.DataFrame) -> pd.DataFrame:
    frame = candidate_rows.copy()
    frame["abs_delta_from"] = frame["delta_from"].abs()
    frame["b_change"] = frame["base_rate_to"] - frame["base_rate_from"]
    frame["abs_gap_from"] = frame["gap_from"].abs()
    return frame


def held_out_logistic_scores(frame: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """Leave-one-tissue-out logistic scores, identical to protocol_audit."""
    features = frame[SCREEN_FEATURES].to_numpy()
    tissues = frame["gene_set"].to_numpy()
    scores = np.zeros(len(frame))
    for tissue in sorted(set(tissues)):
        train, test = tissues != tissue, tissues == tissue
        mean, sd = features[train].mean(axis=0), features[train].std(axis=0) + 1e-12
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        model.fit((features[train] - mean) / sd, labels[train].astype(int))
        scores[test] = model.predict_proba((features[test] - mean) / sd)[:, 1]
    return scores


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
    return {
        "longest_list_pairs": longest,
        "expected_chance_recovery_longest_list": {name: longest * positives[name] / n_candidates
                                                  for name in ("hpn_dream", "beeline_gsd")},
        "expected_chance_recovery_random_list_dorothea": random_length * positives["dorothea_trrust_union_immune"] / n_candidates,
        "random_score_median_over_median_base_rate": {"min": float(ratio.min()), "max": float(ratio.max())},
        "probability_six_method_draw_has_at_most_three_distinct": sum(len(set(d)) <= 3 for d in draws) / len(draws),
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
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    inputs, outputs = arguments.input_dir, arguments.output_dir
    outputs.mkdir(parents=True, exist_ok=True)

    endpoints = reference_endpoints(pd.read_csv(inputs / "score_eval_grn_baselines_immune.csv"))
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

    candidate = pd.read_csv(inputs / AXIS_FILES["candidate"])
    screening, power, null_draws = screening_feasibility(candidate, arguments.n_permutations,
                                                         arguments.n_power, arguments.seed)
    power.to_csv(outputs / "extended_screening_power.csv", index=False)
    null_draws.to_csv(outputs / "extended_screening_null_draws.csv", index=False)
    held_out_shift_bound(candidate).to_csv(outputs / "extended_heldout_shift_bound.csv", index=False)

    derived = derived_quantities(inputs, endpoints)
    summary = {
        "seed": arguments.seed, "n_bootstrap": arguments.n_bootstrap,
        "n_rounding_draws": arguments.n_rounding, "n_permutations": arguments.n_permutations,
        "n_power_draws": arguments.n_power,
        "reference": {
            "zero_recovery_endpoints": endpoints.groupby("reference")["zero_recovery"].sum().astype(int).to_dict(),
            "rows": int(len(audited)), "reversals": int(audited["reversal"].sum()),
            "rows_with_two_informative_endpoints": int((audited["n_informative_endpoints"] == 2).sum()),
            "rows_tied_under_tie_aware_ap": int(audited["tied_under_tie_aware_ap"].sum()),
            "max_min_normalized_margin_reversal": float(audited.loc[audited["reversal"], "min_normalized_margin"].max()),
            "tie_aware_ap_synthetic_check": {"ap": zero_ap, "base_rate": zero_base},
        },
        "metric_exact_tests": exact_tests,
        "derived": derived,
        "screening": screening,
        "interpretation": ("descriptive sensitivity summaries of archived summary scores; intervals are "
                           "conditional on the observed grid and are not confidence intervals"),
    }
    (outputs / "extended_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

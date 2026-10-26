#!/usr/bin/env python3
"""Audit protocol-conditional reversals using archived summary-score inputs.

The primary estimate is the fraction of eligible, non-tied comparison rows
that reverse. Bootstrap resamples retain every row in a sampled cluster and
therefore use the ratio of sampled event totals to sampled row totals. These
intervals are descriptive sensitivity summaries: unordered pair clusters
share methods, shift cells share endpoints, and only three tissues exist.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve


METRICS = {"AUPR": "aupr_median", "AUROC": "auroc_median", "F1": "f1_median"}
PAIR_KEY = ["tissue_from", "tissue_to", "method_a", "method_b", "method_pair"]
QUANTILES = (0.0, 0.05, 0.10, 0.25, 0.50, 0.75)
AXIS_FILES = {
    "candidate": "candidate_shift_pairwise_rows.csv",
    "tissue": "tissue_shift_pairwise_rows_candidate_conditional.csv",
    "reference": "reference_shift_pairwise_rows.csv",
    "mapping": "mapping_policy_pairwise_rows_f1.csv",
}
CELL_COLUMNS = {
    "candidate": ["gene_set", "shift_from", "shift_to"],
    "tissue": ["candidate_set", "tissue_from", "tissue_to"],
    "reference": ["reference_from", "reference_to"],
    "mapping": ["reference", "policy_from", "policy_to"],
}


def row_weighted_cluster_interval(
    comparisons: pd.DataFrame,
    flag_column: str,
    cluster_columns: list[str],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Resample complete clusters, retaining their unequal numbers of rows.

    An unweighted mean of cluster means targets a different estimand when
    cluster sizes differ. We expose that mean only to audit the old pipeline.
    A zero-event input has a degenerate resampling interval; that does not
    establish a zero population reversal probability.
    """
    if comparisons.empty:
        return {"n": 0, "k": 0, "rate": np.nan, "interval_low": np.nan,
                "interval_high": np.nan, "n_clusters": 0, "equal_cluster_mean": np.nan}
    grouped = comparisons.groupby(cluster_columns, sort=True)[flag_column].agg(["sum", "size"])
    event_totals = grouped["sum"].to_numpy(dtype=float)
    row_totals = grouped["size"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(grouped), size=(n_bootstrap, len(grouped)))
    rates = event_totals[draws].sum(axis=1) / row_totals[draws].sum(axis=1)
    return {
        "n": int(len(comparisons)), "k": int(comparisons[flag_column].sum()),
        "rate": float(comparisons[flag_column].mean()),
        "interval_low": float(np.quantile(rates, 0.025)),
        "interval_high": float(np.quantile(rates, 0.975)),
        "n_clusters": int(len(grouped)),
        "equal_cluster_mean": float(np.mean(event_totals / row_totals)),
    }


def aggregate_tissue_candidates(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Use each (tissue transition, method pair) as a secondary summary unit."""
    return comparisons.groupby(PAIR_KEY, sort=True).agg(
        reversal_any_candidate=("reversal", "any"),
        reversal_all_candidates=("reversal", "all"),
        reversal_count=("reversal", "sum"),
        n_candidate_sets=("candidate_set", "nunique"),
    ).reset_index()


def cluster_dependence(comparisons: pd.DataFrame) -> dict:
    """Count pairs of unordered pair clusters that have a method in common."""
    method_pairs = sorted({tuple(sorted((a, b))) for a, b in
                           comparisons[["method_a", "method_b"]].itertuples(index=False, name=None)})
    cluster_pairs = list(itertools.combinations(method_pairs, 2))
    shared_count = sum(bool(set(first) & set(second)) for first, second in cluster_pairs)
    return {
        "n_methods": len({method for pair in method_pairs for method in pair}),
        "n_method_pair_clusters": len(method_pairs),
        "n_pairs_of_clusters": len(cluster_pairs),
        "n_cluster_pairs_sharing_method": shared_count,
        "fraction_cluster_pairs_sharing_method": shared_count / len(cluster_pairs)
        if cluster_pairs else np.nan,
    }


def endpoint_status(values: list[float]) -> str:
    """Classify the four method scores before computing a reversal event."""
    if not np.isfinite(values).all():
        return "missing_score"
    delta_from, delta_to = values[0] - values[1], values[2] - values[3]
    if delta_from == 0 and delta_to == 0:
        return "tie_both"
    if delta_from == 0:
        return "tie_from"
    if delta_to == 0:
        return "tie_to"
    return "eligible"


def metric_pair_rows(tissue_scores: pd.DataFrame) -> pd.DataFrame:
    """Construct every exact key under each metric, including ties/missingness.

    Table A1 pools candidate sets and other protocols before computing its
    medians. Its tissue transitions are a separate aggregate sensitivity
    analysis, not candidate-conditional comparisons from Table A3.
    """
    if tissue_scores.duplicated(["gene_set", "prediction_method"]).any():
        raise ValueError("Table A1 must contain one summary per tissue and method")
    lookup = tissue_scores.set_index(["gene_set", "prediction_method"])
    tissues = sorted(tissue_scores["gene_set"].unique())
    methods = sorted(tissue_scores["prediction_method"].unique())
    comparisons = []
    for tissue_from, tissue_to in itertools.combinations(tissues, 2):
        for method_a, method_b in itertools.combinations(methods, 2):
            for metric, column in METRICS.items():
                values = [lookup.loc[(tissue, method), column]
                          if (tissue, method) in lookup.index else np.nan
                          for tissue, method in [(tissue_from, method_a), (tissue_from, method_b),
                                                (tissue_to, method_a), (tissue_to, method_b)]]
                delta_from, delta_to = values[0] - values[1], values[2] - values[3]
                status = endpoint_status(values)
                comparisons.append({
                    "tissue_from": tissue_from, "tissue_to": tissue_to,
                    "method_a": method_a, "method_b": method_b,
                    "method_pair": method_a + "__" + method_b, "metric": metric,
                    "delta_from": delta_from, "delta_to": delta_to, "status": status,
                    "reversal": bool(delta_from * delta_to < 0) if status == "eligible" else np.nan,
                })
    return pd.DataFrame(comparisons)


def source_eligibility_audit(input_dir: Path) -> pd.DataFrame:
    """Reconstruct included/excluded comparisons from pre-pair source scores.

    Only methods present at both transition endpoints define potential rows;
    a missing score and an exact tie are explicit exclusions. The mapping
    policy files contain differing reference grids, so absent references do
    not create artificial comparisons.
    """
    eligibility = []

    def append_transition(axis: str, metadata: dict, scores_from: pd.Series,
                          scores_to: pd.Series, metric: str) -> None:
        methods = sorted(set(scores_from.index) & set(scores_to.index))
        for method_a, method_b in itertools.combinations(methods, 2):
            values = [scores_from[method_a], scores_from[method_b],
                      scores_to[method_a], scores_to[method_b]]
            status = endpoint_status(values)
            delta_from, delta_to = values[0] - values[1], values[2] - values[3]
            eligibility.append({
                "axis": axis, "metric": metric, **metadata,
                "method_a": method_a, "method_b": method_b,
                "method_pair": method_a + "__" + method_b,
                "delta_from": delta_from, "delta_to": delta_to, "status": status,
                "reversal": bool(delta_from * delta_to < 0) if status == "eligible" else np.nan,
            })

    source_scores = pd.read_csv(input_dir / "table_a3.csv")
    candidate_order = ["all_pairs", "tf_sources", "tf_sources_targets"]
    tissue_order = sorted(source_scores["gene_set"].unique())
    for tissue in tissue_order:
        scores = source_scores[source_scores["gene_set"] == tissue]
        for first, second in itertools.combinations(candidate_order, 2):
            append_transition("candidate", {"gene_set": tissue, "shift_from": first, "shift_to": second},
                              scores[scores["candidate_set"] == first].set_index("prediction_method")["aupr_median"],
                              scores[scores["candidate_set"] == second].set_index("prediction_method")["aupr_median"], "AUPR")
    for candidate in candidate_order:
        scores = source_scores[source_scores["candidate_set"] == candidate]
        for first, second in itertools.combinations(tissue_order, 2):
            append_transition("tissue", {"candidate_set": candidate, "tissue_from": first, "tissue_to": second},
                              scores[scores["gene_set"] == first].set_index("prediction_method")["aupr_median"],
                              scores[scores["gene_set"] == second].set_index("prediction_method")["aupr_median"], "AUPR")
    reference_scores = pd.read_csv(input_dir / "score_eval_grn_baselines_immune.csv")
    for first, second in itertools.combinations(sorted(reference_scores["reference"].unique()), 2):
        append_transition("reference", {"reference_from": first, "reference_to": second},
                          reference_scores[reference_scores["reference"] == first].set_index("method")["aupr"],
                          reference_scores[reference_scores["reference"] == second].set_index("method")["aupr"], "AUPR")
    policy_files = {"legacy_symbols": "score_eval_probe_priors.csv",
                    "full_genes": "score_eval_probe_priors_full_genes.csv",
                    "crosswalk": "score_eval_probe_priors_full_genes_crosswalk.csv",
                    "omnipath_ref": "score_eval_probe_priors_full_genes_omnipath.csv"}
    policy_scores = {policy: pd.read_csv(input_dir / filename) for policy, filename in policy_files.items()}
    references = sorted(set().union(*(set(frame["reference"]) for frame in policy_scores.values())))
    for reference in references:
        relevant = {policy: frame[frame["reference"] == reference].set_index("method")
                    for policy, frame in policy_scores.items() if reference in set(frame["reference"])}
        for first, second in itertools.combinations(sorted(relevant), 2):
            for metric, column in [("F1", "f1"), ("AUPR", "aupr")]:
                append_transition("mapping", {"reference": reference, "policy_from": first, "policy_to": second},
                                  relevant[first][column], relevant[second][column], metric)
    return pd.DataFrame(eligibility)


def match_metric_pairs(comparisons: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require a non-tied eligible comparison on the same key for all metrics."""
    matched, excluded = [], []
    for key, group in comparisons.groupby(PAIR_KEY, sort=True):
        key_row = dict(zip(PAIR_KEY, key))
        eligible = group.set_index("metric")["status"].eq("eligible")
        if all(metric in eligible.index and eligible[metric] for metric in METRICS):
            key_row.update(group.set_index("metric")["reversal"].to_dict())
            matched.append(key_row)
        else:
            key_row["excluded_reasons"] = "; ".join(
                metric + ":" + str(group.set_index("metric").loc[metric, "status"])
                for metric in METRICS if metric in eligible.index and not eligible[metric]
            )
            excluded.append(key_row)
    return (pd.DataFrame(matched, columns=PAIR_KEY + list(METRICS)),
            pd.DataFrame(excluded, columns=PAIR_KEY + ["excluded_reasons"]))


def metric_overlap(comparisons: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    """Report overlap and paired discordance on common eligible keys only."""
    overlap_rows = []
    for scope in ["all_three_metrics_matched", "pairwise_common_eligible"]:
        for first, second in itertools.combinations(METRICS, 2):
            if scope == "all_three_metrics_matched":
                paired = matched[[first, second]].astype(bool)
            else:
                eligible = comparisons[comparisons["status"] == "eligible"]
                wide = eligible.pivot(index=PAIR_KEY, columns="metric", values="reversal")
                paired = wide[[first, second]].dropna().astype(bool)
            both = int((paired[first] & paired[second]).sum())
            first_only = int((paired[first] & ~paired[second]).sum())
            second_only = int((~paired[first] & paired[second]).sum())
            neither = int((~paired[first] & ~paired[second]).sum())
            union = both + first_only + second_only
            overlap_rows.append({
                "scope": scope, "metric_a": first, "metric_b": second,
                "n_common": len(paired), "both_reversal": both,
                "a_only_reversal": first_only, "b_only_reversal": second_only,
                "neither_reversal": neither, "reversal_union": union,
                "jaccard": both / union if union else np.nan,
                "discordance_fraction": (first_only + second_only) / len(paired)
                if len(paired) else np.nan,
            })
    return pd.DataFrame(overlap_rows)


def classifier_audit(candidate_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Reproduce the original LR and heuristic while exposing fold dependence.

    No destination margin enters the learned features. Destination base rate
    is a known-design feature only when benchmark reference/candidate labels
    are already defined. Standardization is fitted exclusively on training
    tissues. Repeated pairs/methods and transition designs remain in both
    train and test, so leave-one-tissue-out is not independent validation of
    new methods, references, or unknown future candidate sets.
    """
    comparisons = candidate_rows.copy()
    comparisons["abs_delta_from"] = comparisons["delta_from"].abs()
    comparisons["b_change"] = comparisons["base_rate_to"] - comparisons["base_rate_from"]
    comparisons["abs_gap_from"] = comparisons["gap_from"].abs()
    feature_columns = ["abs_delta_from", "base_rate_from", "b_change", "abs_gap_from"]
    predictions, folds = [], []
    for tissue in sorted(comparisons["gene_set"].unique()):
        train = comparisons[comparisons["gene_set"] != tissue]
        test = comparisons[comparisons["gene_set"] == tissue]
        training_features = train[feature_columns].to_numpy()
        training_mean = training_features.mean(axis=0)
        training_sd = training_features.std(axis=0) + 1e-12
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        classifier.fit((training_features - training_mean) / training_sd,
                       train["reversal"].astype(int))
        probabilities = classifier.predict_proba(
            (test[feature_columns].to_numpy() - training_mean) / training_sd
        )[:, 1]
        heuristic_scores = -test["delta_from"].abs().to_numpy()
        common_pairs = len(set(train["method_pair"]) & set(test["method_pair"]))
        fold = {
            "held_tissue": tissue, "n_train": len(train), "n_test": len(test),
            "train_reversals": int(train["reversal"].sum()),
            "test_reversals": int(test["reversal"].sum()),
            "test_prevalence": float(test["reversal"].mean()),
            "learned_average_precision": float(average_precision_score(test["reversal"], probabilities)),
            "heuristic_average_precision": float(average_precision_score(test["reversal"], heuristic_scores)),
            "shared_method_pairs_train_test": common_pairs,
            "shared_methods_train_test": len(set(train["method_a"]) | set(train["method_b"])),
        }
        for feature, mean, sd, coefficient in zip(feature_columns, training_mean, training_sd,
                                                   classifier.coef_.ravel()):
            fold["training_mean_" + feature] = float(mean)
            fold["training_sd_" + feature] = float(sd)
            fold["coef_" + feature] = float(coefficient)
        fold["intercept"] = float(classifier.intercept_[0])
        for (_, row), probability, score in zip(test.iterrows(), probabilities, heuristic_scores):
            pool = train[(train["shift_from"] == row["shift_from"]) &
                         (train["shift_to"] == row["shift_to"])]
            radius = float(pool["abs_delta_change"].quantile(0.25))
            predictions.append({
                "gene_set": tissue, "shift_from": row["shift_from"], "shift_to": row["shift_to"],
                "method_pair": row["method_pair"], "reversal": bool(row["reversal"]),
                "learned_probability": float(probability), "heuristic_score": float(score),
                "heuristic_q25_radius": radius,
                "heuristic_q25_selected": bool(abs(row["delta_from"]) <= radius),
            })
        folds.append(fold)
    predictions = pd.DataFrame(predictions)
    selected = predictions["heuristic_q25_selected"]
    true_positive = int((selected & predictions["reversal"]).sum())
    summary = {
        "n_rows": len(predictions), "n_positive_rows": int(predictions["reversal"].sum()),
        "baseline_prevalence": float(predictions["reversal"].mean()), "n_folds": len(folds),
        "average_precision_definition": "sklearn.metrics.average_precision_score; non-interpolated AP, not trapezoidal PR area",
        "learned_average_precision": float(average_precision_score(predictions["reversal"], predictions["learned_probability"])),
        "heuristic_average_precision": float(average_precision_score(predictions["reversal"], predictions["heuristic_score"])),
        "mean_fold_learned_average_precision": float(np.mean([fold["learned_average_precision"] for fold in folds])),
        "mean_fold_heuristic_average_precision": float(np.mean([fold["heuristic_average_precision"] for fold in folds])),
        "q25_n_selected": int(selected.sum()), "q25_true_positive": true_positive,
        "q25_precision": true_positive / int(selected.sum()),
        "q25_recall": true_positive / int(predictions["reversal"].sum()),
        "q25_f1": 2 * true_positive / (int(selected.sum()) + int(predictions["reversal"].sum())),
        "future_base_rate_feature": "available before score shift only for a pre-specified candidate set and reference labels",
        "feature_redundancy": "abs_gap_from = abs_delta_from / base_rate_from; base rates identify only 3 transition designs and are tissue-invariant",
        "leakage_audit": "training-only scaling; no held-tissue destination score feature; historical delta_change trains heuristic radius; same methods and pairs recur across folds",
        "validation_scope": "three related tissues; descriptive discrimination of these archived summary rows only",
    }
    return predictions, pd.DataFrame(folds), summary


def axis_analyses(axis_rows: dict[str, pd.DataFrame], n_bootstrap: int, seed: int) -> dict:
    headlines, intervals, memberships, dependence, deletion, tissue_deletion, margins, cells = ([] for _ in range(8))
    for axis, comparisons in axis_rows.items():
        if comparisons["delta_from"].eq(0).any() or comparisons["delta_to"].eq(0).any():
            raise ValueError(axis + " archived eligible rows contain ties")
        expected = comparisons["delta_from"] * comparisons["delta_to"] < 0
        if not expected.equals(comparisons["reversal"]):
            raise ValueError(axis + " reversal flag differs from its endpoints")
        summary = row_weighted_cluster_interval(comparisons, "reversal", ["method_pair"], n_bootstrap, seed)
        headlines.append({"axis": axis, "metric": "F1" if axis == "mapping" else "AUPR", **summary})
        schemes = {"unordered_method_pair": ["method_pair"], "transition_cell": CELL_COLUMNS[axis]}
        if axis == "candidate":
            schemes["tissue_three_clusters"] = ["gene_set"]
        if axis == "tissue":
            schemes["tissue_transition_three_clusters"] = ["tissue_from", "tissue_to"]
        for scheme, columns in schemes.items():
            sensitivity = row_weighted_cluster_interval(comparisons, "reversal", columns, n_bootstrap, seed)
            intervals.append({"axis": axis, "scheme": scheme, **sensitivity})
        dependence.append({"axis": axis, **cluster_dependence(comparisons)})
        cell_columns = CELL_COLUMNS[axis]
        for row_number, row in comparisons.reset_index(drop=True).iterrows():
            memberships.append({"axis": axis, "comparison_row": row_number,
                                "method_pair": row["method_pair"],
                                "transition_cell": "__".join(str(row[column]) for column in cell_columns),
                                "reversal": bool(row["reversal"])})
        for key, group in comparisons.groupby(cell_columns, sort=True):
            key = key if isinstance(key, tuple) else (key,)
            cells.append({"axis": axis, **dict(zip(cell_columns, key)), "n": len(group),
                          "k": int(group["reversal"].sum()), "rate": float(group["reversal"].mean())})
        methods = sorted(set(comparisons["method_a"]) | set(comparisons["method_b"]))
        for method in methods:
            retained = comparisons[(comparisons["method_a"] != method) & (comparisons["method_b"] != method)]
            deletion.append({"axis": axis, "omitted_method": method, "n": len(retained),
                             "k": int(retained["reversal"].sum()), "rate": float(retained["reversal"].mean())})
        if axis in {"candidate", "tissue"}:
            tissue_names = sorted(comparisons["gene_set"].unique()) if axis == "candidate" else sorted(
                set(comparisons["tissue_from"]) | set(comparisons["tissue_to"]))
            for tissue in tissue_names:
                if axis == "candidate":
                    retained = comparisons[comparisons["gene_set"] != tissue]
                else:
                    retained = comparisons[(comparisons["tissue_from"] != tissue) & (comparisons["tissue_to"] != tissue)]
                tissue_deletion.append({"axis": axis, "omitted_tissue": tissue, "n": len(retained),
                                       "k": int(retained["reversal"].sum()), "rate": float(retained["reversal"].mean())})
        for quantile in QUANTILES:
            threshold = float(comparisons["min_abs_margin"].quantile(quantile))
            retained = comparisons[comparisons["min_abs_margin"] >= threshold]
            summary = row_weighted_cluster_interval(retained, "reversal", ["method_pair"], n_bootstrap, seed)
            margins.append({"axis": axis, "quantile": quantile, "threshold": threshold,
                            "n": summary["n"], "k": summary["k"], "rate": summary["rate"],
                            "ci_low": summary["interval_low"], "ci_high": summary["interval_high"],
                            "n_clusters": summary["n_clusters"],
                            "equal_cluster_mean": summary["equal_cluster_mean"]})
    return {"headline_rates": pd.DataFrame(headlines), "cluster_sensitivity": pd.DataFrame(intervals),
            "cluster_membership": pd.DataFrame(memberships), "cluster_dependence": pd.DataFrame(dependence),
            "leave_one_method_out": pd.DataFrame(deletion), "leave_one_tissue_out": pd.DataFrame(tissue_deletion),
            "margin_sweep": pd.DataFrame(margins), "cell_rates": pd.DataFrame(cells)}


def reference_inventory() -> pd.DataFrame:
    """Record the aligned-grid search and retain distinct settings separately."""
    return pd.DataFrame([
        {"artifact": "score_eval_grn_baselines_immune.csv", "tissues": "immune", "n_rows": 27,
         "methods": "9 baseline methods", "references": "beeline_gsd;dorothea_trrust_union_immune;hpn_dream",
         "alignment": "the sole existing complete 9-method 3-reference grid; primary reference slice"},
        {"artifact": "table_a3.csv", "tissues": "immune;kidney;lung", "n_rows": 54,
         "methods": "6 methods", "references": "references pooled into medians",
         "alignment": "no reference identifier retained; cannot estimate per-reference shifts"},
        {"artifact": "hESC_metrics_inventory.csv", "tissues": "hESC cell-line setting", "n_rows": 144,
         "methods": "16 baseline and attention aggregation variants", "references": "3 BEELINE gold-standard types",
         "alignment": "different cells, methods, reference networks and candidate universes; inventoried, not pooled"},
        {"artifact": "hHep_metrics_inventory.csv", "tissues": "hHep cell-line setting", "n_rows": 135,
         "methods": "15 baseline and attention aggregation variants", "references": "3 BEELINE gold-standard types",
         "alignment": "different cells, methods, reference networks and candidate universes; inventoried, not pooled"},
        {"artifact": "repository search for score_eval_grn_baselines_{kidney,lung} and aligned per-reference metrics",
         "tissues": "kidney;lung", "n_rows": 0, "methods": "not available", "references": "not available",
         "alignment": "no aligned kidney/lung reference-shift CSV found; no original model run undertaken"},
        {"artifact": "score_eval_probe_priors*.csv", "tissues": "immune probe-prior subset", "n_rows": 108,
         "methods": "6 probe/attention methods", "references": "reference grids differ across policy files; 108 source rows",
         "alignment": "mapping analysis uses available common non-tied rows; incomparable to the baseline grid"},
    ])


def plot_audit_outputs(output_dir: Path, audit: dict, predictions: pd.DataFrame,
                       folds: pd.DataFrame, classifier_summary: dict,
                       metric_summary: pd.DataFrame, matched: pd.DataFrame) -> None:
    """Create publication-size figures with explicit observational units."""
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 300})
    colors = {"candidate": "#156082", "tissue": "#E97132", "reference": "#196B24", "mapping": "#78206E"}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), gridspec_kw={"width_ratios": [1, 1.3]})
    headline = audit["headline_rates"]
    for row in headline.itertuples(index=False):
        y = list(colors).index(row.axis)
        axes[0].errorbar(row.rate, y, xerr=[[row.rate - row.interval_low], [row.interval_high - row.rate]],
                         fmt="o", color=colors[row.axis], capsize=4, markersize=7)
        axes[0].text(0.50, y, f"{row.k}/{row.n}", va="center", fontsize=9)
    axes[0].set(yticks=range(4), yticklabels=["Candidate (AUPR)", "Tissue (AUPR)",
                                             "Reference (AUPR)", "Mapping (F1)"],
                xlim=(-0.02, 0.62), xlabel="Eligible comparison reversal fraction", title="A  Row-weighted pair resampling")
    axes[0].invert_yaxis()
    method_deletion = audit["leave_one_method_out"]
    for axis, group in method_deletion.groupby("axis", sort=False):
        y = list(colors).index(axis)
        axes[1].scatter(group["rate"], np.full(len(group), y), color=colors[axis], alpha=0.65, s=24)
        axes[1].scatter(headline.loc[headline["axis"] == axis, "rate"], [y],
                        facecolors="none", edgecolors="black", s=95, linewidths=1.2)
    axes[1].set(yticks=range(4), yticklabels=list(colors), xlabel="Fraction after deleting one method",
                title="B  Method deletion sensitivity", xlim=(-0.02, 0.52))
    axes[1].invert_yaxis()
    fig.text(0.5, 0.01, "Intervals summarize empirical cluster resampling; clusters share methods and settings.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(figure_dir / "protocol_axis_sensitivity.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.8))
    for metric, color in zip(METRICS, ["#156082", "#E97132", "#78206E"]):
        row = metric_summary[(metric_summary["scope"] == "all_three_metrics_matched") &
                             (metric_summary["metric"] == metric)].iloc[0]
        x = list(METRICS).index(metric)
        axes[0].bar(x, row["rate"], color=color, width=0.6)
        axes[0].text(x, row["rate"] + 0.004, f"{int(row['k'])}/{int(row['n'])}", ha="center")
    axes[0].set(xticks=range(3), xticklabels=list(METRICS), ylim=(0, 0.085),
                ylabel="Reversal fraction", title="A  Same 40 eligible keys")
    reversal_keys = matched[matched[list(METRICS)].astype(bool).any(axis=1)]
    heat = reversal_keys[list(METRICS)].astype(int).to_numpy()
    axes[1].imshow(heat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axes[1].set(xticks=range(3), xticklabels=list(METRICS), yticks=range(len(reversal_keys)),
                yticklabels=[r.tissue_from + "–" + r.tissue_to + "\nGENIE3–GRNBoost2"
                             for r in reversal_keys.itertuples(index=False)],
                title="B  Matched reversal membership")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            axes[1].text(j, i, "Reverse" if heat[i, j] else "Stable", ha="center", va="center",
                         color="white" if heat[i, j] else "black", fontsize=8)
    fig.text(0.5, 0.01, "Table A1 pools candidate settings; five AUROC-tied keys are excluded from every metric.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(figure_dir / "protocol_matched_metrics.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.8))
    for column, label, color in [("heuristic_score", "Initial-margin heuristic", "#156082"),
                                 ("learned_probability", "Four-feature logistic model", "#78206E")]:
        precision, recall, _ = precision_recall_curve(predictions["reversal"], predictions[column])
        axes[0].step(recall, precision, where="post", color=color, linewidth=1.8,
                     label=label + f" (AP {average_precision_score(predictions['reversal'], predictions[column]):.3f})")
    axes[0].axhline(classifier_summary["baseline_prevalence"], color="#666666", linestyle="--", linewidth=1,
                    label=f"Prevalence {classifier_summary['baseline_prevalence']:.3f}")
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision", title="Pooled held-out-tissue predictions")
    axes[0].legend(fontsize=8, frameon=False, loc="upper right")
    positions = np.arange(len(folds))
    axes[1].bar(positions - 0.17, folds["heuristic_average_precision"], width=0.34, color="#156082", label="Heuristic AP")
    axes[1].bar(positions + 0.17, folds["learned_average_precision"], width=0.34, color="#78206E", label="Logistic AP")
    axes[1].scatter(positions, folds["test_prevalence"], color="black", marker="_", s=300, label="Fold prevalence")
    axes[1].set(xticks=positions, xticklabels=folds["held_tissue"], ylim=(0, 0.65),
                ylabel="AP and prevalence", title="Only three related tissue folds")
    axes[1].legend(fontsize=8, frameon=False)
    fig.text(0.5, 0.01, "Every fold shares all six methods and 15 method pairs with training; AP is non-interpolated.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(figure_dir / "protocol_screening_audit.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent / "inputs")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = {axis: pd.read_csv(arguments.input_dir / name) for axis, name in AXIS_FILES.items()}
    audit = axis_analyses(comparisons, arguments.n_bootstrap, arguments.seed)
    for label, frame in audit.items():
        frame.to_csv(arguments.output_dir / ("protocol_" + label + ".csv"), index=False)
    eligibility = source_eligibility_audit(arguments.input_dir)
    eligibility.to_csv(arguments.output_dir / "protocol_source_eligibility.csv", index=False)
    eligibility_summary = eligibility.groupby(["axis", "metric", "status"]).size().reset_index(name="n")
    eligibility_summary.to_csv(arguments.output_dir / "protocol_source_exclusion_counts.csv", index=False)
    for axis, archived in comparisons.items():
        metric = "F1" if axis == "mapping" else "AUPR"
        rebuilt = eligibility[(eligibility["axis"] == axis) & (eligibility["metric"] == metric) &
                              (eligibility["status"] == "eligible")]
        key_columns = CELL_COLUMNS[axis] + ["method_pair"]
        archived_keys = archived.set_index(key_columns)["reversal"].sort_index()
        rebuilt_keys = rebuilt.set_index(key_columns)["reversal"].sort_index().astype(bool)
        if not archived_keys.equals(rebuilt_keys):
            raise ValueError(axis + " archived comparisons do not match eligible source-score keys")
    tissue_aggregation = aggregate_tissue_candidates(comparisons["tissue"])
    tissue_aggregation.to_csv(arguments.output_dir / "protocol_tissue_candidate_aggregation.csv", index=False)
    tissue_summaries = []
    for flag in ["reversal_any_candidate", "reversal_all_candidates"]:
        tissue_summaries.append({"secondary_event": flag,
                                **row_weighted_cluster_interval(tissue_aggregation, flag, ["method_pair"],
                                                               arguments.n_bootstrap, arguments.seed)})
    pd.DataFrame(tissue_summaries).to_csv(arguments.output_dir / "protocol_tissue_secondary_rates.csv", index=False)
    metric_rows = metric_pair_rows(pd.read_csv(arguments.input_dir / "table_a1.csv"))
    matched, exclusions = match_metric_pairs(metric_rows)
    metric_rows.to_csv(arguments.output_dir / "protocol_metric_pair_rows.csv", index=False)
    matched.to_csv(arguments.output_dir / "protocol_metric_matched_rows.csv", index=False)
    exclusions.to_csv(arguments.output_dir / "protocol_metric_exclusions.csv", index=False)
    overlap = metric_overlap(metric_rows, matched)
    overlap.to_csv(arguments.output_dir / "protocol_metric_overlap.csv", index=False)
    metric_summaries = []
    for metric in METRICS:
        eligible = metric_rows[(metric_rows["metric"] == metric) & (metric_rows["status"] == "eligible")]
        metric_summaries.append({"scope": "each_metric_eligible", "metric": metric, "n": len(eligible),
                                 "k": int(eligible["reversal"].sum()), "rate": float(eligible["reversal"].mean())})
        metric_summaries.append({"scope": "all_three_metrics_matched", "metric": metric, "n": len(matched),
                                 "k": int(matched[metric].sum()), "rate": float(matched[metric].mean())})
    metric_summary = pd.DataFrame(metric_summaries)
    metric_summary.to_csv(arguments.output_dir / "protocol_metric_summary.csv", index=False)
    predictions, folds, classifier_summary = classifier_audit(comparisons["candidate"])
    predictions.to_csv(arguments.output_dir / "protocol_classifier_predictions.csv", index=False)
    folds.to_csv(arguments.output_dir / "protocol_classifier_folds.csv", index=False)
    (arguments.output_dir / "protocol_classifier_summary.json").write_text(json.dumps(classifier_summary, indent=2) + "\n")
    reference_inventory().to_csv(arguments.output_dir / "protocol_reference_inventory.csv", index=False)
    plot_audit_outputs(arguments.output_dir, audit, predictions, folds, classifier_summary, metric_summary, matched)
    summary = {
        "seed": arguments.seed, "n_bootstrap": arguments.n_bootstrap,
        "primary_estimand": "fraction of eligible non-tied transition-level method-pair comparison rows that reverse",
        "interval_interpretation": "descriptive row-weighted percentile cluster-resampling intervals; no rigorous coverage guarantee because clusters share methods or settings",
        "headlines": audit["headline_rates"].to_dict(orient="records"),
        "secondary_tissue_events": tissue_summaries,
        "cluster_dependence": audit["cluster_dependence"].to_dict(orient="records"),
        "source_exclusion_counts": eligibility_summary.to_dict(orient="records"),
        "metric_summaries": metric_summaries, "metric_overlap": overlap.to_dict(orient="records"),
        "classifier": classifier_summary,
        "metric_limitations": "Table A1 medians pool candidate sets and other protocol dimensions; AUROC ties occur at the reported score precision; only 2 matched reversals, so no paired inferential claim",
        "reference_limitations": "immune-only 9-method 3-reference slice; no aligned kidney/lung reference slice found; different hESC/hHep setting not pooled",
        "mapping_limitations": "different 6-method probe subset; non-tied row denominator; degenerate bootstrap does not imply a zero population rate",
    }
    (arguments.output_dir / "protocol_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(audit["headline_rates"].to_string(index=False))
    print(metric_summary.to_string(index=False))
    print(json.dumps(classifier_summary, indent=2))


if __name__ == "__main__":
    main()

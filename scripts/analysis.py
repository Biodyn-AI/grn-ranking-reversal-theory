#!/usr/bin/env python3
"""Analysis pipeline for the ranking-reversal manuscript.

Run from the repository root:

    python scripts/analysis.py

Reads input CSVs from ``data/`` and writes regenerated tables to
``results/tables/`` and figures to ``results/figures/``. Deterministic given
fixed seeds (project seed 42; permutation seed 42; bootstrap seed 42).

Pipeline:
    - Pairwise reversal indicator per (method-pair, shift-cell) per axis.
    - Cluster-bootstrap 95% CIs (cluster on method pair).
    - Magnitude decomposition (calibration vs base-rate term) on candidate
      shifts; Mann-Whitney comparison reversal vs non-reversal rows.
    - Per-cell BH-FDR-adjusted rates.
    - Margin-magnitude threshold sweep per axis.
    - Three null models: joint method relabel, score-noise bootstrap,
      within-cell rank permutation.
    - Leave-one-tissue-out instability classifiers: quantile heuristic and
      learned logistic regression (PR-AUC).
    - Metric robustness check (AUPR / AUROC / F1).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 200,
    }
)


CANDIDATE_ORDER = ("all_pairs", "tf_sources", "tf_sources_targets")
POLICY_FILES = {
    "legacy_symbols": "score_eval_probe_priors.csv",
    "full_genes": "score_eval_probe_priors_full_genes.csv",
    "crosswalk": "score_eval_probe_priors_full_genes_crosswalk.csv",
    "omnipath_ref": "score_eval_probe_priors_full_genes_omnipath.csv",
}


REPO_ROOT = Path(__file__).resolve().parent.parent  # scripts/ -> repo root


# ---------- I/O helpers ----------


def load_tables(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three summary-median tables (A1/A2/A3) from data/."""
    a1 = pd.read_csv(data_dir / "table_a1_method_summary_by_tissue.csv")
    a2 = pd.read_csv(data_dir / "table_a2_candidate_summary_by_tissue.csv")
    a3 = pd.read_csv(data_dir / "table_a3_method_by_candidate_by_tissue.csv")
    for col in ("aupr_median", "f1_median", "auroc_median"):
        if col in a1.columns:
            a1[col] = pd.to_numeric(a1[col], errors="raise")
    for col in ("aupr_median", "base_rate_median", "candidate_size_median"):
        if col in a2.columns:
            a2[col] = pd.to_numeric(a2[col], errors="raise")
    a3["aupr_median"] = pd.to_numeric(a3["aupr_median"], errors="raise")
    return a1, a2, a3


def load_mapping_rows(policy_dir: Path) -> pd.DataFrame:
    frames = []
    for policy, fname in POLICY_FILES.items():
        df = pd.read_csv(policy_dir / fname)
        df["policy"] = policy
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    for col in ("f1", "aupr", "precision", "recall", "ref_node_overlap_pct"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


# ---------- Intervals ----------


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (center - half, center + half)


def cluster_bootstrap_ci(
    df: pd.DataFrame,
    flag_col: str,
    cluster_cols: Iterable[str],
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Cluster bootstrap on the indicator mean.

    Resamples *clusters* (e.g., method pairs) with replacement, then computes
    the within-resample mean of `flag_col`. Returns mean estimate + percentile
    CI. This gives valid uncertainty under within-cluster dependence (the same
    method pair appears across many shifts / tissues / candidates), which is
    what Wilson intervals miss.
    """
    cluster_cols = list(cluster_cols)
    rng = np.random.default_rng(seed)
    keys = df[cluster_cols].drop_duplicates().reset_index(drop=True)
    grouped = {tuple(row): df.loc[(df[cluster_cols] == row.values).all(axis=1), flag_col].to_numpy()
               for _, row in keys.iterrows()}
    cluster_means = []
    for vals in grouped.values():
        if len(vals) == 0:
            continue
        cluster_means.append(vals.mean())
    cluster_means_arr = np.array(cluster_means, dtype=float)
    if cluster_means_arr.size == 0:
        return {"mean": math.nan, "ci95_low": math.nan, "ci95_high": math.nan, "n_clusters": 0}

    n_cl = cluster_means_arr.size
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_cl, size=n_cl)
        boot[b] = cluster_means_arr[idx].mean()
    return {
        "mean": float(cluster_means_arr.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "n_clusters": int(n_cl),
    }


# ---------- Pair builders ----------


def _add_decomposition(rows: list[dict], gene_set, c1, c2, m1, m2, s1, s2, b1, b2):
    d1 = float(s1[m1] - s1[m2])
    d2 = float(s2[m1] - s2[m2])
    if d1 == 0.0 or d2 == 0.0:
        return
    g1 = d1 / b1
    g2 = d2 / b2
    br = (b2 - b1) * g1
    cal = b2 * (g2 - g1)
    dd = d2 - d1
    pair_key = tuple(sorted([m1, m2]))
    rows.append(
        {
            "gene_set": gene_set,
            "shift_from": c1,
            "shift_to": c2,
            "method_a": m1,
            "method_b": m2,
            "method_pair": "__".join(pair_key),
            "base_rate_from": b1,
            "base_rate_to": b2,
            "delta_from": d1,
            "delta_to": d2,
            "delta_change": dd,
            "abs_delta_from": abs(d1),
            "abs_delta_to": abs(d2),
            "min_abs_margin": min(abs(d1), abs(d2)),
            "gap_from": g1,
            "gap_to": g2,
            "base_rate_term": br,
            "calibration_term": cal,
            "reversal": d1 * d2 < 0.0,
            # Counterfactuals:
            # (a) hold g fixed at g1 -> delta_2_hold_g = b2 * g1
            "delta_to_hold_g_fixed": float(b2 * g1),
            "reversal_if_only_b_changes": (d1 * (b2 * g1)) < 0.0,
            # (b) hold b fixed at b1 -> delta_2_hold_b = b1 * g2
            "delta_to_hold_b_fixed": float(b1 * g2),
            "reversal_if_only_g_changes": (d1 * (b1 * g2)) < 0.0,
            # Min-shift safety margin (Prop 3 in revised paper)
            "safety_margin": abs(d1),
            "abs_delta_change": abs(dd),
        }
    )


def build_candidate_shift_rows(a2: pd.DataFrame, a3: pd.DataFrame) -> pd.DataFrame:
    base = {(r.gene_set, r.candidate_set): float(r.base_rate_median) for r in a2.itertuples(index=False)}
    rows: list[dict] = []
    for tissue in sorted(a3["gene_set"].unique()):
        sub = a3[a3["gene_set"] == tissue]
        for c1, c2 in itertools.combinations(CANDIDATE_ORDER, 2):
            s1 = sub[sub["candidate_set"] == c1].set_index("prediction_method")["aupr_median"].to_dict()
            s2 = sub[sub["candidate_set"] == c2].set_index("prediction_method")["aupr_median"].to_dict()
            methods = sorted(set(s1) & set(s2))
            b1 = base[(tissue, c1)]
            b2 = base[(tissue, c2)]
            for m1, m2 in itertools.combinations(methods, 2):
                _add_decomposition(rows, tissue, c1, c2, m1, m2, s1, s2, b1, b2)
    out = pd.DataFrame(rows)
    if not out.empty:
        err = (out["delta_change"] - out["base_rate_term"] - out["calibration_term"]).abs().max()
        if err > 1e-9:
            raise ValueError(f"Decomposition closure failed: max err={err}")
    return out


def build_tissue_shift_rows(a3: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tissues = sorted(a3["gene_set"].unique())
    conditional_rows: list[dict] = []
    dedup_rows: list[dict] = []
    for c in CANDIDATE_ORDER:
        sub = a3[a3["candidate_set"] == c]
        for t1, t2 in itertools.combinations(tissues, 2):
            s1 = sub[sub["gene_set"] == t1].set_index("prediction_method")["aupr_median"].to_dict()
            s2 = sub[sub["gene_set"] == t2].set_index("prediction_method")["aupr_median"].to_dict()
            methods = sorted(set(s1) & set(s2))
            for m1, m2 in itertools.combinations(methods, 2):
                d1 = float(s1[m1] - s1[m2])
                d2 = float(s2[m1] - s2[m2])
                if d1 == 0.0 or d2 == 0.0:
                    continue
                pair_key = "__".join(sorted([m1, m2]))
                conditional_rows.append(
                    {
                        "candidate_set": c,
                        "tissue_from": t1,
                        "tissue_to": t2,
                        "method_a": m1,
                        "method_b": m2,
                        "method_pair": pair_key,
                        "delta_from": d1,
                        "delta_to": d2,
                        "min_abs_margin": min(abs(d1), abs(d2)),
                        "reversal": d1 * d2 < 0.0,
                    }
                )
    # Deduplicated rate: collapse over candidate_set per (tissue_from, tissue_to, method_a, method_b)
    # using "any reversal across candidate sets" as the indicator at the pair level.
    cdf = pd.DataFrame(conditional_rows)
    if not cdf.empty:
        for (t1, t2, m1, m2), g in cdf.groupby(["tissue_from", "tissue_to", "method_a", "method_b"]):
            dedup_rows.append(
                {
                    "tissue_from": t1,
                    "tissue_to": t2,
                    "method_a": m1,
                    "method_b": m2,
                    "method_pair": "__".join(sorted([m1, m2])),
                    "reversal_any_candidate": bool(g["reversal"].any()),
                    "reversal_all_candidates": bool(g["reversal"].all()),
                    "reversal_count": int(g["reversal"].sum()),
                    "n_candidate_sets": int(len(g)),
                }
            )
    return cdf, pd.DataFrame(dedup_rows)


def build_reference_shift_rows(score_eval_csv: Path) -> pd.DataFrame:
    """Reference-shift rows with a coverage * precision decomposition.

    For each (reference, method) we observe AUPR. For the decomposition we use
    the natural product form available in the source CSV:
        F1 = precision * recall_ratio,  where recall_ratio := recall.
    But for cross-reference comparison the more interpretable factorization is:
        AUPR = base_rate * lift,  where lift := AUPR / base_rate.
    This matches the b*g form but here `lift` is treated as the "discrimination
    relative to chance" quantity and we no longer claim that lift is dimensionless
    across references. We report magnitudes only, not sign attribution, because
    base rate is near-constant across references in this slice (see
    `reviewer_critique.md` C6).
    """
    df = pd.read_csv(score_eval_csv)
    df["aupr"] = pd.to_numeric(df["aupr"], errors="coerce")
    df["true_edges"] = pd.to_numeric(df["true_edges"], errors="coerce")
    df["candidate_edges"] = pd.to_numeric(df["candidate_edges"], errors="coerce")
    df = df.dropna(subset=["aupr", "true_edges", "candidate_edges"]).copy()
    df["base_rate"] = df["true_edges"] / df["candidate_edges"]
    base_rate = df.drop_duplicates("reference").set_index("reference")["base_rate"].to_dict()

    rows: list[dict] = []
    refs = sorted(df["reference"].unique())
    for r1, r2 in itertools.combinations(refs, 2):
        s1 = df[df["reference"] == r1].set_index("method")["aupr"].to_dict()
        s2 = df[df["reference"] == r2].set_index("method")["aupr"].to_dict()
        methods = sorted(set(s1) & set(s2))
        b1, b2 = float(base_rate[r1]), float(base_rate[r2])
        for m1, m2 in itertools.combinations(methods, 2):
            d1 = float(s1[m1] - s1[m2])
            d2 = float(s2[m1] - s2[m2])
            if d1 == 0.0 or d2 == 0.0:
                continue
            pair_key = "__".join(sorted([m1, m2]))
            rows.append(
                {
                    "reference_from": r1,
                    "reference_to": r2,
                    "method_a": m1,
                    "method_b": m2,
                    "method_pair": pair_key,
                    "base_rate_from": b1,
                    "base_rate_to": b2,
                    "delta_from": d1,
                    "delta_to": d2,
                    "min_abs_margin": min(abs(d1), abs(d2)),
                    "reversal": d1 * d2 < 0.0,
                }
            )
    return pd.DataFrame(rows)


def build_mapping_pair_rows(mapping_rows: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    rows: list[dict] = []
    for reference in sorted(mapping_rows["reference"].dropna().unique()):
        sub = mapping_rows[mapping_rows["reference"] == reference]
        policies = sorted(sub["policy"].dropna().unique())
        for p1, p2 in itertools.combinations(policies, 2):
            s1 = sub[sub["policy"] == p1].set_index("method")[metric].to_dict()
            s2 = sub[sub["policy"] == p2].set_index("method")[metric].to_dict()
            methods = sorted(set(s1) & set(s2))
            for m1, m2 in itertools.combinations(methods, 2):
                d1 = s1[m1] - s1[m2]
                d2 = s2[m1] - s2[m2]
                if pd.isna(d1) or pd.isna(d2) or d1 == 0.0 or d2 == 0.0:
                    continue
                pair_key = "__".join(sorted([m1, m2]))
                rows.append(
                    {
                        "metric": metric,
                        "reference": reference,
                        "policy_from": p1,
                        "policy_to": p2,
                        "method_a": m1,
                        "method_b": m2,
                        "method_pair": pair_key,
                        "delta_from": float(d1),
                        "delta_to": float(d2),
                        "min_abs_margin": min(abs(float(d1)), abs(float(d2))),
                        "reversal": float(d1) * float(d2) < 0.0,
                    }
                )
    return pd.DataFrame(rows)


# ---------- Summaries ----------


def summarize_with_clustered_ci(
    df: pd.DataFrame,
    flag_col: str,
    cluster_col: str,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    if df.empty:
        return {"n": 0, "k": 0, "rate": math.nan, "wilson_low": math.nan, "wilson_high": math.nan,
                "cluster_mean": math.nan, "cluster_low": math.nan, "cluster_high": math.nan, "n_clusters": 0}
    n = int(len(df))
    k = int(df[flag_col].sum())
    rate = k / n
    wlow, whigh = wilson_interval(k, n)
    cl = cluster_bootstrap_ci(df, flag_col, [cluster_col], n_boot=n_boot, seed=seed)
    return {
        "n": n,
        "k": k,
        "rate": float(rate),
        "wilson_low": float(wlow),
        "wilson_high": float(whigh),
        "cluster_mean": cl["mean"],
        "cluster_low": cl["ci95_low"],
        "cluster_high": cl["ci95_high"],
        "n_clusters": cl["n_clusters"],
    }


def bh_fdr(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p_values, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def per_cell_summary_with_fdr(rows: pd.DataFrame, group_cols: list[str], scope: str) -> pd.DataFrame:
    out_rows = []
    for key, g in rows.groupby(group_cols, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        n = int(len(g))
        k = int(g["reversal"].sum())
        rate = k / n if n else math.nan
        wlow, whigh = wilson_interval(k, n)
        # Two-sided binomial test vs null p=0.5 (for tail flagging only)
        # We use a simple normal approximation z-score for FDR ranking.
        if n > 0:
            z = (k - 0.5 * n) / math.sqrt(0.25 * n)
            # two-sided p
            from math import erf, sqrt as _sqrt
            p_two = 2 * (1 - 0.5 * (1 + erf(abs(z) / _sqrt(2))))
        else:
            p_two = math.nan
        row = {"scope": scope, "n": n, "k": k, "rate": rate,
               "wilson_low": wlow, "wilson_high": whigh, "p_two_vs_half": p_two}
        for c, v in zip(group_cols, key):
            row[c] = v
        out_rows.append(row)
    df = pd.DataFrame(out_rows)
    if not df.empty:
        df["q_fdr_bh"] = bh_fdr(df["p_two_vs_half"].to_numpy())
    return df


# ---------- Counterfactual analysis ----------


def magnitude_decomposition_summary(candidate_rows: pd.DataFrame) -> dict:
    """Empirical magnitude statistics on decomposition terms in reversal vs non-reversal rows.

    The 'sign' counterfactuals (reversal_if_only_b_changes / reversal_if_only_g_changes)
    are algebraically pinned whenever both base rates are positive (as in our data):
      - reversal_if_only_b_changes := d1*(b2*g1)<0 = (b2/b1)*d1^2 < 0, ALWAYS FALSE.
      - reversal_if_only_g_changes := d1*(b1*g2)<0 = (b1/b2)*d1*d2 < 0, ALWAYS TRUE for reversal rows.
    So 0/22 and 22/22 carry no empirical information beyond "base rates are positive in our data."

    The genuinely empirical content is the magnitude relationship between the two
    decomposition terms in reversal rows. Necessary condition for reversal (from Prop 1):
      |T_cal| - |T_br| > |Delta_1|   when sign(T_br) == sign(Delta_1) (i.e. b2 > b1).
    We test this empirically AND we compare the magnitude ratio against non-reversal rows.
    """
    rev = candidate_rows[candidate_rows["reversal"]]
    nonrev = candidate_rows[~candidate_rows["reversal"]]
    if len(rev) == 0:
        return {"n_reversals": 0}

    rev_ratio = (rev["calibration_term"].abs() / (rev["base_rate_term"].abs() + 1e-30)).to_numpy()
    nonrev_ratio = (nonrev["calibration_term"].abs() / (nonrev["base_rate_term"].abs() + 1e-30)).to_numpy()

    # Mann-Whitney U: are reversal-row ratios stochastically larger than non-reversal?
    try:
        from scipy.stats import mannwhitneyu
        u_stat, u_p = mannwhitneyu(rev_ratio, nonrev_ratio, alternative="greater")
    except Exception:
        u_stat, u_p = math.nan, math.nan

    # Necessary-condition check from Prop 1: in reversal rows (where T_br supports d_1
    # because b_2 > b_1 in our candidate ordering), we need |T_cal| - |T_br| > |Delta_1|.
    necc_holds = (rev["calibration_term"].abs() - rev["base_rate_term"].abs() > rev["delta_from"].abs()).sum()

    return {
        "n_reversals": int(len(rev)),
        "n_non_reversals": int(len(nonrev)),
        # Algebraic (pinned) sign-attribution counterfactuals, kept for transparency:
        "structurally_pinned_frac_b_only_reverses": float(rev["reversal_if_only_b_changes"].mean()),
        "structurally_pinned_frac_g_only_reverses": float(rev["reversal_if_only_g_changes"].mean()),
        # Genuinely empirical magnitude statistics:
        "rev_ratio_mean": float(rev_ratio.mean()),
        "rev_ratio_median": float(np.median(rev_ratio)),
        "rev_frac_calibration_dominates": float((rev_ratio > 1).mean()),
        "nonrev_ratio_mean": float(nonrev_ratio.mean()),
        "nonrev_ratio_median": float(np.median(nonrev_ratio)),
        "nonrev_frac_calibration_dominates": float((nonrev_ratio > 1).mean()),
        "mannwhitney_u": float(u_stat),
        "mannwhitney_p_one_sided_greater": float(u_p),
        # Necessary-condition check (Prop 1 specialization for b_2 > b_1):
        "necc_cond_holds_count": int(necc_holds),
    }


# ---------- Margin-threshold sweep ----------


def margin_threshold_sweep(
    df: pd.DataFrame,
    thresholds: Iterable[float],
    label: str,
    margin_col: str = "min_abs_margin",
    cluster_col: str = "method_pair",
) -> pd.DataFrame:
    rows = []
    for tau in thresholds:
        kept = df[df[margin_col] >= tau]
        n = int(len(kept))
        k = int(kept["reversal"].sum()) if n else 0
        rate = k / n if n else math.nan
        wlow, whigh = wilson_interval(k, n) if n else (math.nan, math.nan)
        cl = cluster_bootstrap_ci(kept, "reversal", [cluster_col]) if n > 0 else \
            {"mean": math.nan, "ci95_low": math.nan, "ci95_high": math.nan, "n_clusters": 0}
        rows.append({
            "scope": label,
            "tau": float(tau),
            "n_kept": n,
            "n_reversals": k,
            "rate": rate,
            "wilson_low": wlow,
            "wilson_high": whigh,
            "cluster_mean": cl["mean"],
            "cluster_low": cl["ci95_low"],
            "cluster_high": cl["ci95_high"],
            "n_clusters": cl["n_clusters"],
        })
    return pd.DataFrame(rows)


# ---------- Null models ----------


def _gather_score_lookup(a3: pd.DataFrame, methods: list[str]) -> dict:
    score = {}
    for tissue in sorted(a3["gene_set"].unique()):
        for c in CANDIDATE_ORDER:
            vals = (
                a3[(a3["gene_set"] == tissue) & (a3["candidate_set"] == c)]
                .set_index("prediction_method")["aupr_median"]
                .to_dict()
            )
            score[(tissue, c)] = {m: vals[m] for m in methods if m in vals}
    return score


def _candidate_reversal_rate(score: dict, methods: list[str]) -> float:
    rev, total = 0, 0
    for tissue in sorted({t for (t, _) in score.keys()}):
        for c1, c2 in itertools.combinations(CANDIDATE_ORDER, 2):
            s1 = score[(tissue, c1)]
            s2 = score[(tissue, c2)]
            for m1, m2 in itertools.combinations(methods, 2):
                if m1 not in s1 or m2 not in s1 or m1 not in s2 or m2 not in s2:
                    continue
                d1 = s1[m1] - s1[m2]
                d2 = s2[m1] - s2[m2]
                if d1 == 0.0 or d2 == 0.0:
                    continue
                total += 1
                if d1 * d2 < 0.0:
                    rev += 1
    return rev / total if total else math.nan


def null_joint_relabel(a3: pd.DataFrame, n_perm: int, seed: int) -> dict:
    """Original (weak) null: permute method identity in condition c2 only."""
    rng = np.random.default_rng(seed)
    methods = sorted(a3["prediction_method"].unique())
    score = _gather_score_lookup(a3, methods)
    observed = _candidate_reversal_rate(score, methods)
    rates = []
    for _ in range(n_perm):
        score_perm = {k: dict(v) for k, v in score.items()}
        for tissue in sorted({t for (t, _) in score.keys()}):
            for c in CANDIDATE_ORDER[1:]:  # leave c1 alone
                perm = methods.copy()
                rng.shuffle(perm)
                src = score[(tissue, c)]
                # Reassign: method m receives the value of method perm[i] where i = index(m)
                score_perm[(tissue, c)] = {m: src[perm[i]] for i, m in enumerate(methods) if perm[i] in src}
        rates.append(_candidate_reversal_rate(score_perm, methods))
    rates = np.array(rates, dtype=float)
    return {
        "null_label": "joint_relabel_c2",
        "observed_rate": observed,
        "null_mean": float(np.nanmean(rates)),
        "null_q025": float(np.nanquantile(rates, 0.025)),
        "null_q975": float(np.nanquantile(rates, 0.975)),
        "p_left": float(np.mean(rates <= observed)),
        "n_perm": int(n_perm),
    }


def null_condition_swap(a3: pd.DataFrame, n_perm: int, seed: int) -> dict:
    """Per-pair condition-swap null.

    For each method pair and each candidate-shift transition, randomly swap the
    roles of c1 and c2 (i.e., randomly flip the sign of `delta_change`) with
    probability 0.5. Reversal is preserved because sign(d1)*sign(d2) is
    invariant to which is labelled "1" vs "2". So this null is degenerate as a
    test of *reversal rate*; we include it as a sanity check on the analysis
    pipeline (it should be near 0 perturbation of the rate).
    """
    rng = np.random.default_rng(seed)
    methods = sorted(a3["prediction_method"].unique())
    score = _gather_score_lookup(a3, methods)
    observed = _candidate_reversal_rate(score, methods)
    rates = []
    for _ in range(n_perm):
        rates.append(observed)  # invariant by construction; documented sanity check
    rates = np.array(rates, dtype=float)
    return {
        "null_label": "condition_swap (invariant sanity check)",
        "observed_rate": observed,
        "null_mean": float(np.nanmean(rates)),
        "null_q025": float(np.nanquantile(rates, 0.025)),
        "null_q975": float(np.nanquantile(rates, 0.975)),
        "p_left": math.nan,
        "n_perm": int(n_perm),
    }


def null_score_noise(a3: pd.DataFrame, n_perm: int, seed: int, sigma_frac: float = 0.10) -> dict:
    """Score-noise null: add Gaussian noise (sigma = sigma_frac * median(|delta|))
    to each method score and recompute reversal rate. Tells the reader what
    fraction of reversals are within plausible noise of the AUPR estimates.

    sigma_frac is the noise scale relative to the median magnitude of
    method-vs-best-method differences within a (tissue, candidate) cell. This
    is a conservative proxy for replicate-level uncertainty in the absence of
    raw replicate-level data.
    """
    rng = np.random.default_rng(seed)
    methods = sorted(a3["prediction_method"].unique())
    score = _gather_score_lookup(a3, methods)
    observed = _candidate_reversal_rate(score, methods)

    # Calibrate noise scale per (tissue, candidate) cell
    sigma = {}
    for key, vals in score.items():
        arr = np.array(list(vals.values()), dtype=float)
        if arr.size < 2:
            sigma[key] = 0.0
            continue
        diffs = np.abs(arr[:, None] - arr[None, :])
        triu = diffs[np.triu_indices_from(diffs, k=1)]
        sigma[key] = float(sigma_frac * np.median(triu)) if triu.size else 0.0

    rates = []
    for _ in range(n_perm):
        score_perm = {}
        for key, vals in score.items():
            s = sigma[key]
            score_perm[key] = {m: v + rng.normal(0.0, s) for m, v in vals.items()}
        rates.append(_candidate_reversal_rate(score_perm, methods))
    rates = np.array(rates, dtype=float)
    return {
        "null_label": f"score_noise_sigma_frac_{sigma_frac:.2f}",
        "observed_rate": observed,
        "null_mean": float(np.nanmean(rates)),
        "null_q025": float(np.nanquantile(rates, 0.025)),
        "null_q975": float(np.nanquantile(rates, 0.975)),
        "p_left": float(np.mean(rates <= observed)),
        "n_perm": int(n_perm),
    }


def null_rank_permutation(a3: pd.DataFrame, n_perm: int, seed: int) -> dict:
    """Rank-permutation null: within each (tissue, candidate) cell, randomly
    permute *which method gets which rank* — preserves the empirical score
    distribution per cell while breaking method identity across cells. This is
    the natural null for "ranking stability".
    """
    rng = np.random.default_rng(seed)
    methods = sorted(a3["prediction_method"].unique())
    score = _gather_score_lookup(a3, methods)
    observed = _candidate_reversal_rate(score, methods)
    rates = []
    for _ in range(n_perm):
        score_perm = {}
        for key, vals in score.items():
            vals_arr = np.array(list(vals.values()), dtype=float)
            perm = rng.permutation(len(vals_arr))
            score_perm[key] = {m: vals_arr[perm[i]] for i, m in enumerate(methods) if m in vals}
        rates.append(_candidate_reversal_rate(score_perm, methods))
    rates = np.array(rates, dtype=float)
    return {
        "null_label": "rank_permutation_within_cell",
        "observed_rate": observed,
        "null_mean": float(np.nanmean(rates)),
        "null_q025": float(np.nanquantile(rates, 0.025)),
        "null_q975": float(np.nanquantile(rates, 0.975)),
        "p_left": float(np.mean(rates <= observed)),
        "n_perm": int(n_perm),
    }


# ---------- Instability classifier ----------


def loo_instability_quantile(candidate_rows: pd.DataFrame, q: float) -> pd.DataFrame:
    rows = []
    for _, row in candidate_rows.iterrows():
        pool = candidate_rows[
            (candidate_rows["shift_from"] == row["shift_from"])
            & (candidate_rows["shift_to"] == row["shift_to"])
            & (candidate_rows["gene_set"] != row["gene_set"])
        ]
        if pool.empty:
            continue
        radius = float(pool["abs_delta_change"].quantile(q))
        pred_score = -abs(float(row["delta_from"]))  # smaller |Δ₁| ⇒ more unstable
        pred_unstable = abs(float(row["delta_from"])) <= radius
        rows.append({
            "gene_set": row["gene_set"],
            "shift_from": row["shift_from"],
            "shift_to": row["shift_to"],
            "method_pair": row["method_pair"],
            "reversal": bool(row["reversal"]),
            "predicted_unstable": bool(pred_unstable),
            "score": pred_score,
            "radius": radius,
            "quantile": q,
        })
    return pd.DataFrame(rows)


def loo_quantile_sweep(candidate_rows: pd.DataFrame, qs: Iterable[float]) -> pd.DataFrame:
    out = []
    for q in qs:
        tagged = loo_instability_quantile(candidate_rows, q=q)
        if tagged.empty:
            continue
        tp = int((tagged["predicted_unstable"] & tagged["reversal"]).sum())
        fp = int((tagged["predicted_unstable"] & ~tagged["reversal"]).sum())
        fn = int((~tagged["predicted_unstable"] & tagged["reversal"]).sum())
        tn = int((~tagged["predicted_unstable"] & ~tagged["reversal"]).sum())
        prec = tp / (tp + fp) if (tp + fp) else math.nan
        rec = tp / (tp + fn) if (tp + fn) else math.nan
        spec = tn / (tn + fp) if (tn + fp) else math.nan
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else math.nan
        # PR-AUC using continuous score (smaller |Δ₁| ⇒ higher unstable score)
        try:
            pr_auc = average_precision_score(tagged["reversal"].astype(int), tagged["score"])
        except Exception:
            pr_auc = math.nan
        out.append({
            "quantile": float(q), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "specificity": spec, "f1": f1,
            "pr_auc": float(pr_auc),
            "predicted_unstable_rate": float(tagged["predicted_unstable"].mean()),
        })
    return pd.DataFrame(out)


def loo_learned_classifier(candidate_rows: pd.DataFrame) -> dict:
    """LOO-by-tissue logistic classifier on (|Δ₁|, b1, b2-b1, |g1|) features.

    Returns PR-AUC on the held-out tissues' rows (pooled), and the feature
    coefficients (for interpretability).
    """
    df = candidate_rows.copy()
    df["abs_delta_from"] = df["delta_from"].abs()
    df["b_change"] = df["base_rate_to"] - df["base_rate_from"]
    df["abs_gap_from"] = df["gap_from"].abs()
    feature_cols = ["abs_delta_from", "base_rate_from", "b_change", "abs_gap_from"]
    tissues = sorted(df["gene_set"].unique())
    held_probs = []
    held_labels = []
    coefs_per_fold = []
    for held in tissues:
        train = df[df["gene_set"] != held]
        test = df[df["gene_set"] == held]
        if train["reversal"].nunique() < 2 or test.empty:
            continue
        X_train = train[feature_cols].to_numpy()
        y_train = train["reversal"].astype(int).to_numpy()
        X_test = test[feature_cols].to_numpy()
        y_test = test["reversal"].astype(int).to_numpy()
        # Standardize per fold (simple)
        mu = X_train.mean(axis=0)
        sd = X_train.std(axis=0) + 1e-12
        Xtr = (X_train - mu) / sd
        Xte = (X_test - mu) / sd
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        clf.fit(Xtr, y_train)
        held_probs.extend(clf.predict_proba(Xte)[:, 1].tolist())
        held_labels.extend(y_test.tolist())
        coefs_per_fold.append({
            "held_tissue": held,
            **{f"coef_{c}": float(w) for c, w in zip(feature_cols, clf.coef_.ravel())},
            "intercept": float(clf.intercept_[0]),
        })
    if not held_labels:
        return {"pr_auc": math.nan, "fold_coefs": []}
    pr_auc = average_precision_score(held_labels, held_probs)
    return {
        "pr_auc": float(pr_auc),
        "fold_coefs": coefs_per_fold,
        "n_test": int(len(held_labels)),
        "baseline_prevalence": float(np.mean(held_labels)),
    }


# ---------- Metric robustness ----------


def metric_robustness_tissue(a1: pd.DataFrame) -> pd.DataFrame:
    """Per (metric) compute tissue-shift reversal rate using A1 (no candidate
    stratification). A1 has aupr_median, auroc_median, f1_median per tissue.
    """
    metrics = [c for c in ("aupr_median", "auroc_median", "f1_median") if c in a1.columns]
    out = []
    for met in metrics:
        rows = []
        tissues = sorted(a1["gene_set"].unique())
        for t1, t2 in itertools.combinations(tissues, 2):
            s1 = a1[a1["gene_set"] == t1].set_index("prediction_method")[met].to_dict()
            s2 = a1[a1["gene_set"] == t2].set_index("prediction_method")[met].to_dict()
            methods = sorted(set(s1) & set(s2))
            for m1, m2 in itertools.combinations(methods, 2):
                d1 = float(s1[m1] - s1[m2])
                d2 = float(s2[m1] - s2[m2])
                if d1 == 0.0 or d2 == 0.0:
                    continue
                rows.append({"reversal": d1 * d2 < 0.0})
        df = pd.DataFrame(rows)
        n = len(df)
        k = int(df["reversal"].sum()) if n else 0
        out.append({"metric": met, "n": n, "k": k, "rate": (k / n) if n else math.nan,
                    "wilson_low": wilson_interval(k, n)[0], "wilson_high": wilson_interval(k, n)[1]})
    return pd.DataFrame(out)


# ---------- Figures ----------


def fig_magnitude_ratio(out: Path, candidate_rows: pd.DataFrame) -> None:
    """Magnitude-ratio comparison: |calibration| / |base-rate| in reversal vs non-reversal rows.

    This is the empirical (non-pinned) decomposition finding. The sign-tally
    figures (0/22 and 22/22) are structural consequences of positive base rates
    and are not plotted here.
    """
    rev = candidate_rows[candidate_rows["reversal"]]
    nonrev = candidate_rows[~candidate_rows["reversal"]]
    rev_ratio = (rev["calibration_term"].abs() / (rev["base_rate_term"].abs() + 1e-30)).to_numpy()
    nonrev_ratio = (nonrev["calibration_term"].abs() / (nonrev["base_rate_term"].abs() + 1e-30)).to_numpy()

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    # log-scale on x because ratios span a wide range
    bins = np.logspace(np.log10(max(1e-3, min(rev_ratio.min(), nonrev_ratio.min()))),
                       np.log10(max(rev_ratio.max(), nonrev_ratio.max())),
                       30)
    ax.hist(nonrev_ratio, bins=bins, alpha=0.55, color="#6c757d",
            label=f"No reversal (n={len(nonrev)}, median={np.median(nonrev_ratio):.2f})",
            density=True)
    ax.hist(rev_ratio, bins=bins, alpha=0.75, color="#c1121f",
            label=f"Reversal (n={len(rev)}, median={np.median(rev_ratio):.2f})",
            density=True)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0, label="|cal| = |base-rate|")
    ax.set_xscale("log")
    ax.set_xlabel("|calibration term| / |base-rate term| (log scale)")
    ax.set_ylabel("Density")
    ax.set_title("Calibration vs base-rate magnitude ratio: reversal vs non-reversal rows")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_margin_threshold(out: Path, sweeps: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    colors = {"candidate": "#c1121f", "tissue": "#003049", "reference": "#2a9d8f", "mapping": "#6a4c93"}
    for label, sweep in sweeps.items():
        if sweep.empty:
            continue
        ax.plot(sweep["tau"], sweep["rate"], label=label, color=colors.get(label, "black"), linewidth=1.8)
        ax.fill_between(sweep["tau"], sweep["cluster_low"], sweep["cluster_high"],
                        color=colors.get(label, "black"), alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("Minimum-margin threshold τ (log scale)")
    ax.set_ylabel("Reversal rate (kept pairs)")
    ax.set_title("Reversal rate vs margin threshold (cluster-bootstrap 95% band)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_nulls_comparison(out: Path, nulls: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    labels = [n["null_label"] for n in nulls]
    obs = nulls[0]["observed_rate"]
    null_means = [n["null_mean"] for n in nulls]
    err_low = [max(0.0, n["null_mean"] - n["null_q025"]) for n in nulls]
    err_high = [max(0.0, n["null_q975"] - n["null_mean"]) for n in nulls]
    xs = np.arange(len(nulls))
    ax.bar(xs, null_means, color="#003049", alpha=0.55, label="Null mean")
    ax.errorbar(xs, null_means, yerr=[err_low, err_high], fmt="none", ecolor="black", capsize=4)
    ax.axhline(obs, color="#c1121f", linewidth=2, label=f"Observed = {obs:.3f}")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Candidate-shift reversal rate")
    ax.set_title("Observed candidate-shift reversal rate vs alternative null models")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_instability_pr_curves(out: Path, sweep: pd.DataFrame, learned: dict, baseline_prev: float) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(sweep["quantile"], sweep["precision"], label="Quantile heuristic — precision", color="#003049")
    ax.plot(sweep["quantile"], sweep["recall"], label="Quantile heuristic — recall", color="#c1121f")
    ax.plot(sweep["quantile"], sweep["f1"], label="Quantile heuristic — F1", color="#2a9d8f")
    ax.axhline(baseline_prev, color="gray", linestyle="--", linewidth=1.0, label=f"Random precision ({baseline_prev:.2f})")
    if not math.isnan(learned["pr_auc"]):
        ax.axhline(learned["pr_auc"], color="#6a4c93", linestyle=":", linewidth=1.6,
                   label=f"Learned (LR) PR-AUC = {learned['pr_auc']:.2f}")
    ax.set_xlabel("Instability quantile threshold")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("LOO-by-tissue instability screening — quantile vs learned")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_metric_robustness(out: Path, metric_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    labels = [m.replace("_median", "").upper() for m in metric_df["metric"]]
    rates = metric_df["rate"].to_numpy()
    lo = metric_df["wilson_low"].to_numpy()
    hi = metric_df["wilson_high"].to_numpy()
    xs = np.arange(len(labels))
    ax.bar(xs, rates, color="#003049", alpha=0.75)
    ax.errorbar(xs, rates, yerr=[rates - lo, hi - rates], fmt="none", ecolor="black", capsize=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Tissue-shift reversal rate")
    ax.set_title("Metric-robustness: tissue-shift reversal across metrics")
    for i, v in enumerate(rates):
        ax.text(i, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------- Main ----------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate all tables and figures from data/ for the ranking-reversal study."
    )
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"),
                        help="Folder with table_a1/a2/a3 CSVs and score-eval CSVs.")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "tables"),
                        help="Folder for generated CSV/JSON tables.")
    parser.add_argument("--fig-dir", default=str(REPO_ROOT / "results" / "figures"),
                        help="Folder for generated figures.")
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    a1, a2, a3 = load_tables(data_dir)

    methods_in_a3 = sorted(a3["prediction_method"].unique())

    candidate_rows = build_candidate_shift_rows(a2, a3)
    tissue_cond_rows, tissue_dedup_rows = build_tissue_shift_rows(a3)
    reference_rows = build_reference_shift_rows(data_dir / "score_eval_grn_baselines_immune.csv")
    mapping_rows = load_mapping_rows(data_dir)
    mapping_pair_rows_f1 = build_mapping_pair_rows(mapping_rows, metric="f1")
    mapping_pair_rows_aupr = build_mapping_pair_rows(mapping_rows, metric="aupr") if "aupr" in mapping_rows.columns else pd.DataFrame()

    # --- Headline rates with clustered CIs ---
    headlines = {
        "candidate_shift": summarize_with_clustered_ci(candidate_rows, "reversal", "method_pair", n_boot=args.n_boot, seed=args.seed),
        "tissue_shift_candidate_conditional": summarize_with_clustered_ci(tissue_cond_rows, "reversal", "method_pair", n_boot=args.n_boot, seed=args.seed),
        "tissue_shift_deduplicated_any": summarize_with_clustered_ci(tissue_dedup_rows, "reversal_any_candidate", "method_pair", n_boot=args.n_boot, seed=args.seed) if not tissue_dedup_rows.empty else {},
        "tissue_shift_deduplicated_all": summarize_with_clustered_ci(tissue_dedup_rows, "reversal_all_candidates", "method_pair", n_boot=args.n_boot, seed=args.seed) if not tissue_dedup_rows.empty else {},
        "reference_shift": summarize_with_clustered_ci(reference_rows, "reversal", "method_pair", n_boot=args.n_boot, seed=args.seed),
        "mapping_policy_f1": summarize_with_clustered_ci(mapping_pair_rows_f1, "reversal", "method_pair", n_boot=args.n_boot, seed=args.seed),
        "mapping_policy_aupr": summarize_with_clustered_ci(mapping_pair_rows_aupr, "reversal", "method_pair", n_boot=args.n_boot, seed=args.seed) if not mapping_pair_rows_aupr.empty else {},
    }

    # --- Magnitude decomposition (replaces the algebraically-pinned counterfactual) ---
    counterfactual = magnitude_decomposition_summary(candidate_rows)

    # --- Per-cell rates with BH-FDR (for highlighted cells) ---
    candidate_cells = per_cell_summary_with_fdr(candidate_rows, ["gene_set", "shift_from", "shift_to"], "candidate_shift_cells")
    tissue_cells = per_cell_summary_with_fdr(tissue_cond_rows, ["candidate_set", "tissue_from", "tissue_to"], "tissue_shift_cells")
    reference_cells = per_cell_summary_with_fdr(reference_rows, ["reference_from", "reference_to"], "reference_shift_cells")

    # --- Margin-threshold sweeps ---
    # Use per-axis percentiles of min_abs_margin for adaptive thresholds
    def _taus(df, ps=(0.0, 0.05, 0.10, 0.25, 0.50, 0.75)):
        if df.empty:
            return []
        m = df["min_abs_margin"].to_numpy()
        return [float(np.quantile(m, p)) for p in ps]

    sweep_candidate = margin_threshold_sweep(candidate_rows, _taus(candidate_rows), "candidate")
    sweep_tissue = margin_threshold_sweep(tissue_cond_rows, _taus(tissue_cond_rows), "tissue")
    sweep_reference = margin_threshold_sweep(reference_rows, _taus(reference_rows), "reference") if not reference_rows.empty else pd.DataFrame()
    sweep_mapping = margin_threshold_sweep(mapping_pair_rows_f1, _taus(mapping_pair_rows_f1), "mapping") if not mapping_pair_rows_f1.empty else pd.DataFrame()

    # --- Null models ---
    null_a = null_joint_relabel(a3, n_perm=args.n_perm, seed=args.seed)
    null_b = null_score_noise(a3, n_perm=args.n_perm, seed=args.seed, sigma_frac=0.10)
    null_c = null_rank_permutation(a3, n_perm=args.n_perm, seed=args.seed)
    nulls = [null_a, null_b, null_c]

    # --- Instability classifier ---
    quant_sweep = loo_quantile_sweep(candidate_rows, np.linspace(0.05, 0.95, 19))
    learned = loo_learned_classifier(candidate_rows)
    baseline_prev = float(candidate_rows["reversal"].mean())

    # --- Metric robustness on tissue shift (A1) ---
    metric_robustness = metric_robustness_tissue(a1)

    # --- Save artifacts ---
    candidate_rows.to_csv(out_dir / "candidate_shift_pairwise_rows.csv", index=False)
    tissue_cond_rows.to_csv(out_dir / "tissue_shift_pairwise_rows_candidate_conditional.csv", index=False)
    tissue_dedup_rows.to_csv(out_dir / "tissue_shift_pairwise_rows_deduplicated.csv", index=False)
    reference_rows.to_csv(out_dir / "reference_shift_pairwise_rows.csv", index=False)
    mapping_pair_rows_f1.to_csv(out_dir / "mapping_policy_pairwise_rows_f1.csv", index=False)
    if not mapping_pair_rows_aupr.empty:
        mapping_pair_rows_aupr.to_csv(out_dir / "mapping_policy_pairwise_rows_aupr.csv", index=False)

    candidate_cells.to_csv(out_dir / "candidate_shift_cells_fdr.csv", index=False)
    tissue_cells.to_csv(out_dir / "tissue_shift_cells_fdr.csv", index=False)
    reference_cells.to_csv(out_dir / "reference_shift_cells_fdr.csv", index=False)

    sweep_candidate.to_csv(out_dir / "margin_sweep_candidate.csv", index=False)
    sweep_tissue.to_csv(out_dir / "margin_sweep_tissue.csv", index=False)
    sweep_reference.to_csv(out_dir / "margin_sweep_reference.csv", index=False)
    sweep_mapping.to_csv(out_dir / "margin_sweep_mapping.csv", index=False)

    pd.DataFrame(nulls).to_csv(out_dir / "nulls_comparison.csv", index=False)
    quant_sweep.to_csv(out_dir / "instability_quantile_sweep.csv", index=False)
    pd.DataFrame([{"pr_auc": learned["pr_auc"], "n_test": learned.get("n_test", 0),
                   "baseline_prevalence": learned.get("baseline_prevalence", baseline_prev)}]
                ).to_csv(out_dir / "instability_learned_summary.csv", index=False)
    pd.DataFrame(learned["fold_coefs"]).to_csv(out_dir / "instability_learned_fold_coefs.csv", index=False)

    metric_robustness.to_csv(out_dir / "metric_robustness_tissue.csv", index=False)

    # --- Figures (names match those referenced in paper/paper.md) ---
    fig_magnitude_ratio(fig_dir / "fig_magnitude_ratio.png", candidate_rows)
    fig_margin_threshold(
        fig_dir / "fig_margin_sweep.png",
        {"candidate": sweep_candidate, "tissue": sweep_tissue,
         "reference": sweep_reference, "mapping": sweep_mapping},
    )
    fig_nulls_comparison(fig_dir / "fig_nulls.png", nulls)
    fig_instability_pr_curves(fig_dir / "fig_instability_screening.png",
                              quant_sweep, learned, baseline_prev)
    fig_metric_robustness(fig_dir / "fig_metric_robustness.png", metric_robustness)

    heat = candidate_cells.pivot_table(index="gene_set",
                                       columns=["shift_from", "shift_to"],
                                       values="rate", aggfunc="first")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    im = ax.imshow(heat.values, aspect="auto", cmap="YlOrRd",
                   vmin=0.0, vmax=max(0.01, float(np.nanmax(heat.values))))
    ax.set_yticks(range(len(heat.index))); ax.set_yticklabels(heat.index)
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels([f"{a}->{b}" for a, b in heat.columns], rotation=20, ha="right")
    ax.set_title("Candidate-Set Shift Reversal Rate (per cell)")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax).set_label("Reversal rate")
    fig.tight_layout(); fig.savefig(fig_dir / "fig_candidate_heatmap.png"); plt.close(fig)

    # --- Summary JSON ---
    summary = {
        "methods_in_A3": methods_in_a3,
        "n_methods": len(methods_in_a3),
        "n_tissues": int(a3["gene_set"].nunique()),
        "n_candidate_sets": int(a3["candidate_set"].nunique()),
        "candidate_shift": headlines["candidate_shift"],
        "tissue_shift_candidate_conditional": headlines["tissue_shift_candidate_conditional"],
        "tissue_shift_deduplicated_any": headlines["tissue_shift_deduplicated_any"],
        "tissue_shift_deduplicated_all": headlines["tissue_shift_deduplicated_all"],
        "reference_shift": headlines["reference_shift"],
        "mapping_policy_f1": headlines["mapping_policy_f1"],
        "mapping_policy_aupr": headlines["mapping_policy_aupr"],
        "magnitude_decomposition": counterfactual,
        "nulls": nulls,
        "learned_instability_pr_auc": learned["pr_auc"],
        "learned_instability_baseline_prevalence": learned.get("baseline_prevalence", baseline_prev),
        "metric_robustness_tissue": metric_robustness.to_dict(orient="records"),
        "n_perm": args.n_perm,
        "n_boot": args.n_boot,
        "seed": args.seed,
    }
    # Place summary.json one level up so it sits at results/summary.json
    summary_path = out_dir.parent / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print("Wrote tables to", out_dir, "figures to", fig_dir, "summary to", summary_path)


if __name__ == "__main__":
    main()

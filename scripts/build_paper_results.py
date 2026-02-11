#!/usr/bin/env python3
"""Build research-grade results for Proposal 2 (ranking reversal theory).

This script extends the initial Proposal 2 run with:
- candidate-set shift reversals and decomposition,
- tissue-shift reversals (per candidate set),
- reference-shift reversals from GRN baseline outputs,
- mapping-policy reversal/decomposition checks,
- permutation null for reversal rate,
- leave-one-tissue-out instability diagnostics,
- publication-ready figures.
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

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 200,
    }
)

CANDIDATE_ORDER = ["all_pairs", "tf_sources", "tf_sources_targets"]
POLICY_FILES = {
    "legacy_symbols": "score_eval_probe_priors.csv",
    "full_genes": "score_eval_probe_priors_full_genes.csv",
    "crosswalk": "score_eval_probe_priors_full_genes_crosswalk.csv",
    "omnipath_ref": "score_eval_probe_priors_full_genes_omnipath.csv",
}


def parse_markdown_table(path: Path, heading_prefix: str) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8").splitlines()
    idx = next((i for i, line in enumerate(lines) if line.startswith(heading_prefix)), None)
    if idx is None:
        raise ValueError(f"Heading {heading_prefix!r} not found in {path}")

    table_lines: list[str] = []
    in_table = False
    for line in lines[idx + 1 :]:
        stripped = line.strip()
        if stripped.startswith("|"):
            table_lines.append(stripped)
            in_table = True
            continue
        if in_table:
            break

    if len(table_lines) < 3:
        raise ValueError(f"Could not parse table under heading {heading_prefix!r}")

    headers = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in table_lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append(dict(zip(headers, cells)))
    return pd.DataFrame(rows)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (math.nan, math.nan)
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return (center - half, center + half)


def summarize_binary(df: pd.DataFrame, flag_col: str, group_cols: Iterable[str], scope: str) -> pd.DataFrame:
    group_cols = list(group_cols)
    records: list[dict[str, object]] = []
    iterator = df.groupby(group_cols, sort=True) if group_cols else [((), df)]
    for key, g in iterator:
        if group_cols and not isinstance(key, tuple):
            key = (key,)
        n_total = int(len(g))
        n_true = int(g[flag_col].sum())
        rate = n_true / n_total if n_total else math.nan
        low, high = wilson_interval(n_true, n_total)
        row: dict[str, object] = {
            "scope": scope,
            "n": n_total,
            "k": n_true,
            "rate": rate,
            "ci95_low": low,
            "ci95_high": high,
        }
        for col, val in zip(group_cols, key):
            row[col] = val
        records.append(row)
    return pd.DataFrame(records)


def load_tables(report_md: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    a1 = parse_markdown_table(report_md, "### Table A1:")
    a2 = parse_markdown_table(report_md, "### Table A2:")
    a3 = parse_markdown_table(report_md, "### Table A3:")

    a1["aupr_median"] = pd.to_numeric(a1["aupr_median"], errors="raise")
    a1["f1_median"] = pd.to_numeric(a1["f1_median"], errors="raise")

    a2["aupr_median"] = pd.to_numeric(a2["aupr_median"], errors="raise")
    a2["base_rate_median"] = pd.to_numeric(a2["base_rate_median"], errors="raise")
    a2["candidate_size_median"] = pd.to_numeric(a2["candidate_size_median"], errors="raise")

    a3["aupr_median"] = pd.to_numeric(a3["aupr_median"], errors="raise")
    return a1, a2, a3


def build_candidate_shift_rows(a2: pd.DataFrame, a3: pd.DataFrame) -> pd.DataFrame:
    base = {(r.gene_set, r.candidate_set): float(r.base_rate_median) for r in a2.itertuples(index=False)}
    rows: list[dict[str, object]] = []
    for tissue in sorted(a3["gene_set"].unique()):
        sub = a3[a3["gene_set"] == tissue]
        for c1, c2 in itertools.combinations(CANDIDATE_ORDER, 2):
            s1 = sub[sub["candidate_set"] == c1].set_index("prediction_method")["aupr_median"].to_dict()
            s2 = sub[sub["candidate_set"] == c2].set_index("prediction_method")["aupr_median"].to_dict()
            methods = sorted(set(s1) & set(s2))
            b1 = base[(tissue, c1)]
            b2 = base[(tissue, c2)]
            for m1, m2 in itertools.combinations(methods, 2):
                d1 = float(s1[m1] - s1[m2])
                d2 = float(s2[m1] - s2[m2])
                if d1 == 0.0 or d2 == 0.0:
                    continue
                g1 = d1 / b1
                g2 = d2 / b2
                br = (b2 - b1) * g1
                cal = b2 * (g2 - g1)
                dd = d2 - d1
                rows.append(
                    {
                        "gene_set": tissue,
                        "shift_from": c1,
                        "shift_to": c2,
                        "method_a": m1,
                        "method_b": m2,
                        "base_rate_from": b1,
                        "base_rate_to": b2,
                        "delta_from": d1,
                        "delta_to": d2,
                        "delta_change": dd,
                        "delta_change_abs": abs(dd),
                        "gap_from": g1,
                        "gap_to": g2,
                        "base_rate_term": br,
                        "calibration_term": cal,
                        "reversal": d1 * d2 < 0.0,
                        "calibration_opposes_initial": cal * d1 < 0.0,
                        "base_rate_opposes_initial": br * d1 < 0.0,
                        "min_extra_shift_for_reversal": max(0.0, abs(d1 + dd)) if d1 * d2 >= 0.0 else 0.0,
                    }
                )
    out = pd.DataFrame(rows)
    err = (out["delta_change"] - out["base_rate_term"] - out["calibration_term"]).abs().max()
    if err > 1e-9:
        raise ValueError(f"Candidate decomposition check failed: max err={err}")
    return out


def build_tissue_shift_rows(a3: pd.DataFrame) -> pd.DataFrame:
    tissues = sorted(a3["gene_set"].unique())
    rows: list[dict[str, object]] = []
    for candidate_set in CANDIDATE_ORDER:
        subset = a3[a3["candidate_set"] == candidate_set]
        for t1, t2 in itertools.combinations(tissues, 2):
            s1 = subset[subset["gene_set"] == t1].set_index("prediction_method")["aupr_median"].to_dict()
            s2 = subset[subset["gene_set"] == t2].set_index("prediction_method")["aupr_median"].to_dict()
            methods = sorted(set(s1) & set(s2))
            for m1, m2 in itertools.combinations(methods, 2):
                d1 = float(s1[m1] - s1[m2])
                d2 = float(s2[m1] - s2[m2])
                if d1 == 0.0 or d2 == 0.0:
                    continue
                rows.append(
                    {
                        "candidate_set": candidate_set,
                        "tissue_from": t1,
                        "tissue_to": t2,
                        "method_a": m1,
                        "method_b": m2,
                        "delta_from": d1,
                        "delta_to": d2,
                        "reversal": d1 * d2 < 0.0,
                    }
                )
    return pd.DataFrame(rows)


def build_overall_tissue_shift_rows(a1: pd.DataFrame) -> pd.DataFrame:
    tissues = sorted(a1["gene_set"].unique())
    rows: list[dict[str, object]] = []
    for t1, t2 in itertools.combinations(tissues, 2):
        s1 = a1[a1["gene_set"] == t1].set_index("prediction_method")["aupr_median"].to_dict()
        s2 = a1[a1["gene_set"] == t2].set_index("prediction_method")["aupr_median"].to_dict()
        methods = sorted(set(s1) & set(s2))
        for m1, m2 in itertools.combinations(methods, 2):
            d1 = float(s1[m1] - s1[m2])
            d2 = float(s2[m1] - s2[m2])
            if d1 == 0.0 or d2 == 0.0:
                continue
            rows.append(
                {
                    "tissue_from": t1,
                    "tissue_to": t2,
                    "method_a": m1,
                    "method_b": m2,
                    "delta_from": d1,
                    "delta_to": d2,
                    "reversal": d1 * d2 < 0.0,
                }
            )
    return pd.DataFrame(rows)


def build_reference_shift_rows(score_eval_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(score_eval_csv)
    df["aupr"] = pd.to_numeric(df["aupr"], errors="coerce")
    df["true_edges"] = pd.to_numeric(df["true_edges"], errors="coerce")
    df["candidate_edges"] = pd.to_numeric(df["candidate_edges"], errors="coerce")
    df = df.dropna(subset=["aupr", "true_edges", "candidate_edges"])

    base_rate = (
        df[["reference", "true_edges", "candidate_edges"]]
        .drop_duplicates(subset=["reference"])
        .assign(base_rate=lambda x: x["true_edges"] / x["candidate_edges"])
        .set_index("reference")["base_rate"]
        .to_dict()
    )

    rows: list[dict[str, object]] = []
    refs = sorted(df["reference"].unique())
    for r1, r2 in itertools.combinations(refs, 2):
        s1 = df[df["reference"] == r1].set_index("method")["aupr"].to_dict()
        s2 = df[df["reference"] == r2].set_index("method")["aupr"].to_dict()
        methods = sorted(set(s1) & set(s2))
        b1 = float(base_rate[r1])
        b2 = float(base_rate[r2])
        for m1, m2 in itertools.combinations(methods, 2):
            d1 = float(s1[m1] - s1[m2])
            d2 = float(s2[m1] - s2[m2])
            if d1 == 0.0 or d2 == 0.0:
                continue
            g1 = d1 / b1
            g2 = d2 / b2
            br = (b2 - b1) * g1
            cal = b2 * (g2 - g1)
            rows.append(
                {
                    "reference_from": r1,
                    "reference_to": r2,
                    "method_a": m1,
                    "method_b": m2,
                    "base_rate_from": b1,
                    "base_rate_to": b2,
                    "delta_from": d1,
                    "delta_to": d2,
                    "delta_change": d2 - d1,
                    "base_rate_term": br,
                    "calibration_term": cal,
                    "reversal": d1 * d2 < 0.0,
                    "calibration_opposes_initial": cal * d1 < 0.0,
                    "base_rate_opposes_initial": br * d1 < 0.0,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        err = (out["delta_change"] - out["base_rate_term"] - out["calibration_term"]).abs().max()
        if err > 1e-9:
            raise ValueError(f"Reference decomposition check failed: max err={err}")
    return out


def load_mapping_rows(policy_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for policy, fname in POLICY_FILES.items():
        df = pd.read_csv(policy_dir / fname)
        df["policy"] = policy
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["f1"] = pd.to_numeric(out["f1"], errors="coerce")
    out["ref_node_overlap_pct"] = pd.to_numeric(out["ref_node_overlap_pct"], errors="coerce")
    return out


def build_mapping_pair_rows(mapping_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for reference in sorted(mapping_rows["reference"].dropna().unique()):
        sub = mapping_rows[mapping_rows["reference"] == reference]
        policies = sorted(sub["policy"].dropna().unique())
        for p1, p2 in itertools.combinations(policies, 2):
            s1 = sub[sub["policy"] == p1].set_index("method")["f1"].to_dict()
            s2 = sub[sub["policy"] == p2].set_index("method")["f1"].to_dict()
            methods = sorted(set(s1) & set(s2))
            for m1, m2 in itertools.combinations(methods, 2):
                d1 = s1[m1] - s1[m2]
                d2 = s2[m1] - s2[m2]
                if pd.isna(d1) or pd.isna(d2) or d1 == 0.0 or d2 == 0.0:
                    continue
                rows.append(
                    {
                        "reference": reference,
                        "policy_from": p1,
                        "policy_to": p2,
                        "method_a": m1,
                        "method_b": m2,
                        "delta_from": float(d1),
                        "delta_to": float(d2),
                        "reversal": d1 * d2 < 0.0,
                    }
                )
    return pd.DataFrame(rows)


def build_mapping_method_decomposition(mapping_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    policy_order = list(POLICY_FILES.keys())
    for reference in sorted(mapping_rows["reference"].dropna().unique()):
        sub = mapping_rows[mapping_rows["reference"] == reference]
        for method in sorted(sub["method"].dropna().unique()):
            m = sub[sub["method"] == method].set_index("policy")
            for p1, p2 in itertools.combinations(policy_order, 2):
                if p1 not in m.index or p2 not in m.index:
                    continue
                f1_1 = m.at[p1, "f1"]
                f1_2 = m.at[p2, "f1"]
                c1 = m.at[p1, "ref_node_overlap_pct"] / 100.0
                c2 = m.at[p2, "ref_node_overlap_pct"] / 100.0
                if pd.isna(f1_1) or pd.isna(f1_2) or pd.isna(c1) or pd.isna(c2) or c1 <= 0.0 or c2 <= 0.0:
                    continue
                q1 = f1_1 / c1
                q2 = f1_2 / c2
                cov_term = (c2 - c1) * q1
                qual_term = c2 * (q2 - q1)
                delta = f1_2 - f1_1
                rows.append(
                    {
                        "reference": reference,
                        "method": method,
                        "policy_from": p1,
                        "policy_to": p2,
                        "f1_from": float(f1_1),
                        "f1_to": float(f1_2),
                        "coverage_from": float(c1),
                        "coverage_to": float(c2),
                        "quality_from": float(q1),
                        "quality_to": float(q2),
                        "delta_f1": float(delta),
                        "coverage_term": float(cov_term),
                        "quality_term": float(qual_term),
                        "decomposition_error": float(delta - cov_term - qual_term),
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty and out["decomposition_error"].abs().max() > 1e-9:
        raise ValueError("Mapping decomposition error exceeds tolerance")
    return out


def decomposition_summary(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Summarize decomposition behavior for reversal vs non-reversal rows."""
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for reversal_flag, label in [(False, "non_reversal"), (True, "reversal")]:
        sub = df[df["reversal"] == reversal_flag]
        if sub.empty:
            continue
        rows.append(
            {
                "scope": scope,
                "group": label,
                "n_rows": int(len(sub)),
                "calibration_opposes_initial_rate": float(sub["calibration_opposes_initial"].mean()),
                "base_rate_opposes_initial_rate": float(sub["base_rate_opposes_initial"].mean()),
                "mean_abs_calibration_term": float(sub["calibration_term"].abs().mean()),
                "mean_abs_base_rate_term": float(sub["base_rate_term"].abs().mean()),
                "mean_abs_calibration_to_base_ratio": float(
                    (sub["calibration_term"].abs() / (sub["base_rate_term"].abs() + 1e-12)).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def candidate_permutation_null(a3: pd.DataFrame, n_perm: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    methods = sorted(a3["prediction_method"].unique())

    score = {}
    for tissue in sorted(a3["gene_set"].unique()):
        for c in CANDIDATE_ORDER:
            vals = (
                a3[(a3["gene_set"] == tissue) & (a3["candidate_set"] == c)]
                .set_index("prediction_method")["aupr_median"]
                .to_dict()
            )
            score[(tissue, c)] = vals

    def reversal_rate(shuffled: bool) -> float:
        rev = 0
        total = 0
        for tissue in sorted(a3["gene_set"].unique()):
            for c1, c2 in itertools.combinations(CANDIDATE_ORDER, 2):
                s1 = score[(tissue, c1)]
                s2 = score[(tissue, c2)]
                # keep c1 fixed; optionally shuffle c2 method association
                if shuffled:
                    perm_methods = methods.copy()
                    rng.shuffle(perm_methods)
                    s2_eff = {m: s2[perm_methods[i]] for i, m in enumerate(methods)}
                else:
                    s2_eff = s2
                for m1, m2 in itertools.combinations(methods, 2):
                    d1 = s1[m1] - s1[m2]
                    d2 = s2_eff[m1] - s2_eff[m2]
                    if d1 == 0.0 or d2 == 0.0:
                        continue
                    total += 1
                    if d1 * d2 < 0.0:
                        rev += 1
        return rev / total

    observed = reversal_rate(shuffled=False)
    null_rates = np.array([reversal_rate(shuffled=True) for _ in range(n_perm)], dtype=float)
    p_left = float((null_rates <= observed).mean())
    p_right = float((null_rates >= observed).mean())

    return pd.DataFrame(
        {
            "observed_rate": [observed],
            "null_mean": [float(null_rates.mean())],
            "null_std": [float(null_rates.std(ddof=1))],
            "null_q025": [float(np.quantile(null_rates, 0.025))],
            "null_q975": [float(np.quantile(null_rates, 0.975))],
            "p_left": [p_left],
            "p_right": [p_right],
            "n_perm": [n_perm],
        }
    )


def loo_instability(candidate_rows: pd.DataFrame, q: float = 0.5) -> pd.DataFrame:
    rows = []
    for idx, row in candidate_rows.iterrows():
        pool = candidate_rows[
            (candidate_rows["shift_from"] == row["shift_from"])
            & (candidate_rows["shift_to"] == row["shift_to"])
            & (candidate_rows["gene_set"] != row["gene_set"])
        ]
        if pool.empty:
            continue
        radius = float(pool["delta_change_abs"].quantile(q))
        pred = abs(float(row["delta_from"])) <= radius
        rows.append(
            {
                "gene_set": row["gene_set"],
                "shift_from": row["shift_from"],
                "shift_to": row["shift_to"],
                "method_a": row["method_a"],
                "method_b": row["method_b"],
                "reversal": bool(row["reversal"]),
                "predicted_unstable": bool(pred),
                "radius": radius,
                "quantile": q,
            }
        )
    return pd.DataFrame(rows)


def loo_instability_quantile_sweep(
    candidate_rows: pd.DataFrame, quantiles: Iterable[float]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for q in quantiles:
        tagged = loo_instability(candidate_rows, q=float(q))
        if tagged.empty:
            continue
        tp = int((tagged["predicted_unstable"] & tagged["reversal"]).sum())
        fp = int((tagged["predicted_unstable"] & ~tagged["reversal"]).sum())
        fn = int((~tagged["predicted_unstable"] & tagged["reversal"]).sum())
        tn = int((~tagged["predicted_unstable"] & ~tagged["reversal"]).sum())
        precision = tp / (tp + fp) if (tp + fp) else math.nan
        recall = tp / (tp + fn) if (tp + fn) else math.nan
        specificity = tn / (tn + fp) if (tn + fp) else math.nan
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) else math.nan
        balanced_acc = (recall + specificity) / 2.0 if not (math.isnan(recall) or math.isnan(specificity)) else math.nan
        rows.append(
            {
                "quantile": float(q),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
                "balanced_accuracy": balanced_acc,
                "predicted_unstable_rate": float(tagged["predicted_unstable"].mean()),
            }
        )
    return pd.DataFrame(rows)


def eval_classifier(df: pd.DataFrame, pred_col: str, truth_col: str, scope: str, group_cols: Iterable[str] = ()) -> pd.DataFrame:
    group_cols = list(group_cols)
    records = []
    iterator = df.groupby(group_cols, sort=True) if group_cols else [((), df)]
    for key, g in iterator:
        if group_cols and not isinstance(key, tuple):
            key = (key,)
        tp = int((g[pred_col] & g[truth_col]).sum())
        fp = int((g[pred_col] & ~g[truth_col]).sum())
        fn = int((~g[pred_col] & g[truth_col]).sum())
        tn = int((~g[pred_col] & ~g[truth_col]).sum())
        precision = tp / (tp + fp) if (tp + fp) else math.nan
        recall = tp / (tp + fn) if (tp + fn) else math.nan
        specificity = tn / (tn + fp) if (tn + fp) else math.nan
        row = {
            "scope": scope,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
        }
        for col, value in zip(group_cols, key):
            row[col] = value
        records.append(row)
    return pd.DataFrame(records)


def make_figures(
    out_fig_dir: Path,
    candidate_summary: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    tissue_summary: pd.DataFrame,
    reference_summary: pd.DataFrame,
    perm_summary: pd.DataFrame,
    instability_sweep: pd.DataFrame,
) -> None:
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Candidate-shift reversal heatmap
    heat = candidate_summary.pivot_table(
        index="gene_set",
        columns=["shift_from", "shift_to"],
        values="rate",
        aggfunc="first",
    )
    fig, ax = plt.subplots(figsize=(8, 3.8))
    im = ax.imshow(heat.values, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=max(0.01, np.nanmax(heat.values)))
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels([f"{a}->{b}" for a, b in heat.columns], rotation=20, ha="right")
    ax.set_title("Candidate-Set Shift Reversal Rate")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Reversal rate")
    fig.tight_layout()
    fig.savefig(out_fig_dir / "fig1_candidate_shift_reversal_heatmap.png")
    plt.close(fig)

    # Figure 2: Base-rate term vs calibration term
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    non_rev = candidate_rows[~candidate_rows["reversal"]]
    rev = candidate_rows[candidate_rows["reversal"]]
    ax.scatter(non_rev["base_rate_term"], non_rev["calibration_term"], s=18, alpha=0.45, label="No reversal", color="#6c757d")
    ax.scatter(rev["base_rate_term"], rev["calibration_term"], s=26, alpha=0.85, label="Reversal", color="#c1121f")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Base-rate term")
    ax.set_ylabel("Calibration term")
    ax.set_title("Candidate-Shift Decomposition")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_fig_dir / "fig2_decomposition_scatter.png")
    plt.close(fig)

    # Figure 3: Tissue-shift reversal by candidate set
    ts = tissue_summary.pivot_table(
        index="candidate_set",
        columns=["tissue_from", "tissue_to"],
        values="rate",
        aggfunc="first",
    )
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    im = ax.imshow(ts.values, aspect="auto", cmap="Blues", vmin=0.0, vmax=max(0.01, np.nanmax(ts.values)))
    ax.set_yticks(range(len(ts.index)))
    ax.set_yticklabels(ts.index)
    ax.set_xticks(range(len(ts.columns)))
    ax.set_xticklabels([f"{a}->{b}" for a, b in ts.columns], rotation=20, ha="right")
    ax.set_title("Tissue Shift Reversal Rate by Candidate Set")
    for i in range(ts.shape[0]):
        for j in range(ts.shape[1]):
            ax.text(j, i, f"{ts.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Reversal rate")
    fig.tight_layout()
    fig.savefig(out_fig_dir / "fig3_tissue_shift_reversal_heatmap.png")
    plt.close(fig)

    # Figure 4: Reference-shift reversal summary
    if not reference_summary.empty:
        labels = [f"{r.reference_from}->{r.reference_to}" for r in reference_summary.itertuples(index=False)]
        vals = reference_summary["rate"].to_numpy()
        fig, ax = plt.subplots(figsize=(8.0, 3.6))
        ax.bar(labels, vals, color="#1d3557")
        ax.set_ylabel("Reversal rate")
        ax.set_ylim(0, max(vals.max() * 1.25, 0.1))
        ax.set_title("Reference Shift Reversal Rate (GRN Baselines, Immune)")
        ax.tick_params(axis="x", rotation=20)
        for x, y in enumerate(vals):
            ax.text(x, y + 0.005, f"{y:.2f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_fig_dir / "fig4_reference_shift_reversal_bar.png")
        plt.close(fig)

    # Figure 5: Observed vs permutation null reversal rate
    r = perm_summary.iloc[0]
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.bar(["Observed", "Null mean"], [r["observed_rate"], r["null_mean"]], color=["#d62828", "#003049"])
    ax.errorbar([1], [r["null_mean"]], yerr=[[r["null_mean"] - r["null_q025"]], [r["null_q975"] - r["null_mean"]]], fmt="none", ecolor="black", capsize=4)
    ax.set_ylabel("Reversal rate")
    ax.set_title("Observed Candidate-Shift Reversals vs Permutation Null")
    fig.tight_layout()
    fig.savefig(out_fig_dir / "fig5_observed_vs_null.png")
    plt.close(fig)

    # Figure 6: Instability quantile sweep (LOO)
    if not instability_sweep.empty:
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        ax.plot(instability_sweep["quantile"], instability_sweep["precision"], label="Precision", color="#003049")
        ax.plot(instability_sweep["quantile"], instability_sweep["recall"], label="Recall", color="#d62828")
        ax.plot(instability_sweep["quantile"], instability_sweep["balanced_accuracy"], label="Balanced acc.", color="#2a9d8f")
        ax.set_xlabel("Instability quantile threshold")
        ax.set_ylabel("Metric value")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Leave-One-Tissue-Out Instability Sweep")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_fig_dir / "fig6_instability_quantile_sweep.png")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Proposal 2 paper-grade results")
    parser.add_argument(
        "--report-markdown",
        default="reports/evaluation_bias_protocol/workshop/eval_bias_paper_draft.md",
    )
    parser.add_argument(
        "--score-eval-baseline",
        default="subproject_02_evaluation_bias_protocol/implementation/network_inference_materials/outputs/score_eval_grn_baselines_immune.csv",
    )
    parser.add_argument(
        "--policy-dir",
        default="subproject_02_evaluation_bias_protocol/implementation/network_inference_materials/outputs",
    )
    parser.add_argument(
        "--out-dir",
        default="market_research/ambitious_paper_questions/proposal_02_ranking_reversal_theory/artifacts_paper",
    )
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    a1, a2, a3 = load_tables(Path(args.report_markdown))

    candidate_rows = build_candidate_shift_rows(a2, a3)
    tissue_rows = build_tissue_shift_rows(a3)
    tissue_overall_rows = build_overall_tissue_shift_rows(a1)
    reference_rows = build_reference_shift_rows(Path(args.score_eval_baseline))

    mapping_rows = load_mapping_rows(Path(args.policy_dir))
    mapping_pair_rows = build_mapping_pair_rows(mapping_rows)
    mapping_method_decomp = build_mapping_method_decomposition(mapping_rows)

    # Summaries
    candidate_summary = summarize_binary(candidate_rows, "reversal", ["gene_set", "shift_from", "shift_to"], "candidate_shift")
    candidate_overall = summarize_binary(candidate_rows, "reversal", [], "candidate_shift_overall")

    tissue_summary = summarize_binary(tissue_rows, "reversal", ["candidate_set", "tissue_from", "tissue_to"], "tissue_shift")
    tissue_overall = summarize_binary(tissue_rows, "reversal", ["candidate_set"], "tissue_shift_overall_by_candidate")
    tissue_global = summarize_binary(tissue_rows, "reversal", [], "tissue_shift_overall")

    tissue_overall_method_summary = summarize_binary(
        tissue_overall_rows,
        "reversal",
        ["tissue_from", "tissue_to"],
        "overall_method_tissue_shift",
    )
    tissue_overall_method_global = summarize_binary(
        tissue_overall_rows,
        "reversal",
        [],
        "overall_method_tissue_shift_global",
    )

    reference_summary = summarize_binary(reference_rows, "reversal", ["reference_from", "reference_to"], "reference_shift") if not reference_rows.empty else pd.DataFrame()
    reference_overall = summarize_binary(reference_rows, "reversal", [], "reference_shift_overall") if not reference_rows.empty else pd.DataFrame()

    mapping_pair_summary = summarize_binary(mapping_pair_rows, "reversal", ["reference", "policy_from", "policy_to"], "mapping_policy_shift") if not mapping_pair_rows.empty else pd.DataFrame()
    mapping_pair_overall = summarize_binary(mapping_pair_rows, "reversal", [], "mapping_policy_shift_overall") if not mapping_pair_rows.empty else pd.DataFrame()
    candidate_decomp_summary = decomposition_summary(candidate_rows, "candidate_shift")
    reference_decomp_summary = decomposition_summary(reference_rows, "reference_shift")

    # Instability regions
    instability_q50 = loo_instability(candidate_rows, q=0.5)
    instability_q90 = loo_instability(candidate_rows, q=0.9)
    instability = pd.concat([instability_q50, instability_q90], ignore_index=True)
    instability_sweep = loo_instability_quantile_sweep(
        candidate_rows, quantiles=np.linspace(0.05, 0.95, 19)
    )
    instability_perf_overall = pd.concat(
        [
            eval_classifier(instability_q50, "predicted_unstable", "reversal", "loo_q50_overall"),
            eval_classifier(instability_q90, "predicted_unstable", "reversal", "loo_q90_overall"),
        ],
        ignore_index=True,
    )
    instability_perf_transition = pd.concat(
        [
            eval_classifier(instability_q50, "predicted_unstable", "reversal", "loo_q50_by_transition", ["shift_from", "shift_to"]),
            eval_classifier(instability_q90, "predicted_unstable", "reversal", "loo_q90_by_transition", ["shift_from", "shift_to"]),
        ],
        ignore_index=True,
    )

    # Null model
    permutation_summary = candidate_permutation_null(a3, n_perm=args.n_perm, seed=args.seed)

    # Save primary artifacts
    candidate_rows.sort_values(["gene_set", "shift_from", "shift_to", "method_a", "method_b"]).to_csv(
        out_dir / "candidate_shift_pairwise_rows.csv", index=False
    )
    candidate_summary.to_csv(out_dir / "candidate_shift_reversal_summary.csv", index=False)
    candidate_overall.to_csv(out_dir / "candidate_shift_reversal_overall.csv", index=False)

    tissue_rows.sort_values(["candidate_set", "tissue_from", "tissue_to", "method_a", "method_b"]).to_csv(
        out_dir / "tissue_shift_pairwise_rows.csv", index=False
    )
    tissue_summary.to_csv(out_dir / "tissue_shift_reversal_summary.csv", index=False)
    tissue_overall.to_csv(out_dir / "tissue_shift_reversal_overall_by_candidate.csv", index=False)
    tissue_global.to_csv(out_dir / "tissue_shift_reversal_overall.csv", index=False)

    tissue_overall_rows.sort_values(["tissue_from", "tissue_to", "method_a", "method_b"]).to_csv(
        out_dir / "overall_method_tissue_shift_rows.csv", index=False
    )
    tissue_overall_method_summary.to_csv(out_dir / "overall_method_tissue_shift_summary.csv", index=False)
    tissue_overall_method_global.to_csv(out_dir / "overall_method_tissue_shift_global.csv", index=False)

    reference_rows.sort_values(["reference_from", "reference_to", "method_a", "method_b"]).to_csv(
        out_dir / "reference_shift_pairwise_rows.csv", index=False
    )
    reference_summary.to_csv(out_dir / "reference_shift_reversal_summary.csv", index=False)
    reference_overall.to_csv(out_dir / "reference_shift_reversal_overall.csv", index=False)

    mapping_pair_rows.sort_values(["reference", "policy_from", "policy_to", "method_a", "method_b"]).to_csv(
        out_dir / "mapping_policy_pairwise_rows.csv", index=False
    )
    mapping_pair_summary.to_csv(out_dir / "mapping_policy_reversal_summary.csv", index=False)
    mapping_pair_overall.to_csv(out_dir / "mapping_policy_reversal_overall.csv", index=False)

    mapping_method_decomp.sort_values(["reference", "method", "policy_from", "policy_to"]).to_csv(
        out_dir / "mapping_policy_method_decomposition.csv", index=False
    )
    candidate_decomp_summary.to_csv(out_dir / "candidate_shift_decomposition_summary.csv", index=False)
    reference_decomp_summary.to_csv(out_dir / "reference_shift_decomposition_summary.csv", index=False)

    instability.sort_values(["quantile", "gene_set", "shift_from", "shift_to", "method_a", "method_b"]).to_csv(
        out_dir / "instability_loo_classification.csv", index=False
    )
    instability_perf_overall.to_csv(out_dir / "instability_loo_performance_overall.csv", index=False)
    instability_perf_transition.to_csv(out_dir / "instability_loo_performance_by_transition.csv", index=False)
    instability_sweep.to_csv(out_dir / "instability_loo_quantile_sweep.csv", index=False)

    permutation_summary.to_csv(out_dir / "candidate_shift_permutation_null.csv", index=False)

    make_figures(
        out_fig_dir=fig_dir,
        candidate_summary=candidate_summary,
        candidate_rows=candidate_rows,
        tissue_summary=tissue_summary,
        reference_summary=reference_summary,
        perm_summary=permutation_summary,
        instability_sweep=instability_sweep,
    )

    # top reversal-prone method pairs
    pair_rank = (
        candidate_rows[candidate_rows["reversal"]]
        .assign(method_pair=lambda d: d["method_a"] + "__" + d["method_b"])
        .groupby("method_pair")
        .size()
        .sort_values(ascending=False)
        .rename("reversal_count")
        .reset_index()
    )
    pair_rank.to_csv(out_dir / "candidate_shift_top_reversal_method_pairs.csv", index=False)

    summary = {
        "candidate_shift_pairs": int(len(candidate_rows)),
        "candidate_shift_reversals": int(candidate_rows["reversal"].sum()),
        "candidate_shift_rate": float(candidate_rows["reversal"].mean()),
        "tissue_shift_pairs": int(len(tissue_rows)),
        "tissue_shift_reversals": int(tissue_rows["reversal"].sum()),
        "tissue_shift_rate": float(tissue_rows["reversal"].mean()),
        "overall_method_tissue_shift_pairs": int(len(tissue_overall_rows)),
        "overall_method_tissue_shift_reversals": int(tissue_overall_rows["reversal"].sum()),
        "overall_method_tissue_shift_rate": float(tissue_overall_rows["reversal"].mean()),
        "reference_shift_pairs": int(len(reference_rows)),
        "reference_shift_reversals": int(reference_rows["reversal"].sum()) if not reference_rows.empty else 0,
        "reference_shift_rate": float(reference_rows["reversal"].mean()) if not reference_rows.empty else 0.0,
        "mapping_policy_pairs": int(len(mapping_pair_rows)),
        "mapping_policy_reversals": int(mapping_pair_rows["reversal"].sum()) if not mapping_pair_rows.empty else 0,
        "mapping_policy_rate": float(mapping_pair_rows["reversal"].mean()) if not mapping_pair_rows.empty else 0.0,
        "permutation_null": permutation_summary.iloc[0].to_dict(),
        "n_perm": args.n_perm,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Wrote paper-grade artifacts to", out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

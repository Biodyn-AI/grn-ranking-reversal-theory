#!/usr/bin/env python3
"""Proposal 2 execution: ranking-reversal theory sanity checks with existing artifacts.

This script intentionally uses only existing, committed outputs to stay low-compute.
It produces:
- candidate-set reversal counts,
- exact base-rate vs calibration decomposition for each method pair,
- instability-region diagnostics,
- mapping-policy reversal checks.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd


CANDIDATE_ORDER = ["all_pairs", "tf_sources", "tf_sources_targets"]
POLICY_FILES = {
    "legacy_symbols": "score_eval_probe_priors.csv",
    "full_genes": "score_eval_probe_priors_full_genes.csv",
    "crosswalk": "score_eval_probe_priors_full_genes_crosswalk.csv",
    "omnipath_ref": "score_eval_probe_priors_full_genes_omnipath.csv",
}


def parse_markdown_table(markdown_path: Path, heading_prefix: str) -> pd.DataFrame:
    """Parse a simple pipe table that follows a markdown heading.

    The parser is purpose-built for the workshop draft tables and keeps the logic
    explicit to avoid hidden parser behavior.
    """
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    heading_index = next(
        (idx for idx, line in enumerate(lines) if line.startswith(heading_prefix)),
        None,
    )
    if heading_index is None:
        raise ValueError(f"Heading '{heading_prefix}' not found in {markdown_path}")

    table_lines: list[str] = []
    in_table = False
    for line in lines[heading_index + 1 :]:
        stripped = line.strip()
        if stripped.startswith("|"):
            table_lines.append(stripped)
            in_table = True
            continue
        if in_table:
            break

    if len(table_lines) < 3:
        raise ValueError(
            f"Expected markdown table under '{heading_prefix}', found {len(table_lines)} lines"
        )

    headers = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for row_line in table_lines[2:]:
        cells = [cell.strip() for cell in row_line.strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append(dict(zip(headers, cells)))

    return pd.DataFrame(rows)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion."""
    if total <= 0:
        return (math.nan, math.nan)
    p_hat = successes / total
    denom = 1.0 + (z * z) / total
    center = (p_hat + (z * z) / (2.0 * total)) / denom
    half = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) + (z * z) / (4.0 * total)) / total)
    return (center - half, center + half)


def load_candidate_tables(report_markdown: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Table A2 (base rates) and A3 (method x candidate-set AUPR medians)."""
    table_a2 = parse_markdown_table(report_markdown, "### Table A2:")
    table_a3 = parse_markdown_table(report_markdown, "### Table A3:")

    table_a2 = table_a2.rename(
        columns={
            "gene_set": "gene_set",
            "candidate_set": "candidate_set",
            "base_rate_median": "base_rate_median",
        }
    )
    table_a2["base_rate_median"] = pd.to_numeric(table_a2["base_rate_median"], errors="coerce")

    table_a3 = table_a3.rename(
        columns={
            "gene_set": "gene_set",
            "candidate_set": "candidate_set",
            "prediction_method": "prediction_method",
            "aupr_median": "aupr_median",
        }
    )
    table_a3["aupr_median"] = pd.to_numeric(table_a3["aupr_median"], errors="coerce")

    if table_a2["base_rate_median"].isna().any():
        raise ValueError("Failed to parse numeric base rates from Table A2")
    if table_a3["aupr_median"].isna().any():
        raise ValueError("Failed to parse numeric AUPR medians from Table A3")

    return table_a2, table_a3


def build_candidate_pair_rows(
    table_a2: pd.DataFrame,
    table_a3: pd.DataFrame,
) -> pd.DataFrame:
    """Build pairwise method-gap rows for every tissue and candidate-set shift.

    The decomposition uses:
      delta(S) = M_A(S) - M_B(S) = b(S) * g(S)
    so that
      delta(S2)-delta(S1) = (b2-b1)*g1 + b2*(g2-g1)
    where the first term is the base-rate term and the second is the
    calibration/ranking-shape term.
    """
    base_lookup = {
        (row.gene_set, row.candidate_set): float(row.base_rate_median)
        for row in table_a2.itertuples(index=False)
    }

    rows: list[dict[str, object]] = []
    for tissue in sorted(table_a3["gene_set"].unique()):
        tissue_rows = table_a3[table_a3["gene_set"] == tissue]
        for shift_from, shift_to in itertools.combinations(CANDIDATE_ORDER, 2):
            from_scores = (
                tissue_rows[tissue_rows["candidate_set"] == shift_from]
                .set_index("prediction_method")["aupr_median"]
                .to_dict()
            )
            to_scores = (
                tissue_rows[tissue_rows["candidate_set"] == shift_to]
                .set_index("prediction_method")["aupr_median"]
                .to_dict()
            )

            common_methods = sorted(set(from_scores) & set(to_scores))
            if len(common_methods) < 2:
                continue

            base_from = base_lookup[(tissue, shift_from)]
            base_to = base_lookup[(tissue, shift_to)]

            for method_a, method_b in itertools.combinations(common_methods, 2):
                delta_from = float(from_scores[method_a] - from_scores[method_b])
                delta_to = float(to_scores[method_a] - to_scores[method_b])

                # Ties do not define a strict ranking, so we skip them.
                if delta_from == 0.0 or delta_to == 0.0:
                    continue

                gap_from = delta_from / base_from
                gap_to = delta_to / base_to
                base_rate_term = (base_to - base_from) * gap_from
                calibration_term = base_to * (gap_to - gap_from)
                delta_change = delta_to - delta_from

                rows.append(
                    {
                        "gene_set": tissue,
                        "shift_from": shift_from,
                        "shift_to": shift_to,
                        "method_a": method_a,
                        "method_b": method_b,
                        "base_rate_from": base_from,
                        "base_rate_to": base_to,
                        "delta_from": delta_from,
                        "delta_to": delta_to,
                        "delta_change": delta_change,
                        "delta_change_abs": abs(delta_change),
                        "gap_from": gap_from,
                        "gap_to": gap_to,
                        "base_rate_term": base_rate_term,
                        "calibration_term": calibration_term,
                        "reversal": delta_from * delta_to < 0.0,
                        "calibration_opposes_initial": calibration_term * delta_from < 0.0,
                        "base_rate_opposes_initial": base_rate_term * delta_from < 0.0,
                        "calibration_to_base_ratio": abs(calibration_term)
                        / (abs(base_rate_term) + 1e-12),
                    }
                )

    pair_rows = pd.DataFrame(rows)
    if pair_rows.empty:
        raise ValueError("No pair rows were generated from Table A3")

    # Sanity check: decomposition should be exact up to floating point precision.
    max_error = (pair_rows["delta_change"] - pair_rows["base_rate_term"] - pair_rows["calibration_term"]).abs().max()
    if max_error > 1e-9:
        raise ValueError(f"Decomposition check failed; max error={max_error}")

    return pair_rows


def summarize_reversals(
    pair_rows: pd.DataFrame,
    group_cols: Iterable[str],
    label: str,
) -> pd.DataFrame:
    """Summarize reversal counts with Wilson intervals."""
    group_cols = list(group_cols)
    records: list[dict[str, object]] = []
    if group_cols:
        iterator = pair_rows.groupby(group_cols, dropna=False, sort=True)
    else:
        iterator = [((), pair_rows)]

    for key, group in iterator:
        if group_cols and not isinstance(key, tuple):
            key = (key,)
        n_total = int(len(group))
        n_reversal = int(group["reversal"].sum())
        rate = n_reversal / n_total if n_total else math.nan
        ci_low, ci_high = wilson_interval(n_reversal, n_total)

        record: dict[str, object] = {"scope": label, "n_pairs": n_total, "n_reversals": n_reversal, "reversal_rate": rate, "ci95_low": ci_low, "ci95_high": ci_high}
        for col, value in zip(group_cols, key):
            record[col] = value
        records.append(record)

    return pd.DataFrame(records)


def build_instability_regions(pair_rows: pd.DataFrame, quantiles: tuple[float, ...] = (0.5, 0.9)) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construct instability-region diagnostics.

    Instability rule used here:
      unstable if |delta_from| <= B_transition(q)
    where B_transition(q) is the q-quantile of observed |delta_change| for the
    same candidate-set transition.

    This gives a transition-level perturbation radius and lets us compare
    broad theory-based regions to observed reversals.
    """
    threshold_rows: list[dict[str, object]] = []
    tagged_rows: list[pd.DataFrame] = []
    perf_rows: list[dict[str, object]] = []

    transitions = ["shift_from", "shift_to"]
    for q in quantiles:
        thresholds = (
            pair_rows.groupby(transitions, sort=True)["delta_change_abs"]
            .quantile(q)
            .reset_index(name="instability_radius")
        )
        thresholds["quantile"] = q
        threshold_rows.append(thresholds)

        tagged = pair_rows.merge(thresholds, on=transitions, how="left")
        tagged["quantile"] = q
        tagged["predicted_unstable"] = tagged["delta_from"].abs() <= tagged["instability_radius"]
        tagged_rows.append(tagged)

        for scope_cols, scope_name in [([], "overall"), (transitions, "by_transition")]:
            if scope_cols:
                iterator = tagged.groupby(scope_cols, sort=True)
            else:
                iterator = [((), tagged)]

            for scope_key, scope_df in iterator:
                if not isinstance(scope_key, tuple):
                    scope_key = (scope_key,)

                tp = int((scope_df["predicted_unstable"] & scope_df["reversal"]).sum())
                fp = int((scope_df["predicted_unstable"] & ~scope_df["reversal"]).sum())
                fn = int((~scope_df["predicted_unstable"] & scope_df["reversal"]).sum())
                tn = int((~scope_df["predicted_unstable"] & ~scope_df["reversal"]).sum())

                precision = tp / (tp + fp) if (tp + fp) else math.nan
                recall = tp / (tp + fn) if (tp + fn) else math.nan
                specificity = tn / (tn + fp) if (tn + fp) else math.nan

                row: dict[str, object] = {
                    "quantile": q,
                    "scope": scope_name,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "precision": precision,
                    "recall": recall,
                    "specificity": specificity,
                }
                for col, value in zip(scope_cols, scope_key):
                    row[col] = value
                perf_rows.append(row)

    thresholds_df = pd.concat(threshold_rows, ignore_index=True)
    tagged_df = pd.concat(tagged_rows, ignore_index=True)
    perf_df = pd.DataFrame(perf_rows)
    return thresholds_df, tagged_df, perf_df


def load_mapping_policy_rows(policy_dir: Path) -> pd.DataFrame:
    """Load score-eval rows across mapping/candidate policy variants."""
    frames: list[pd.DataFrame] = []
    for policy_name, filename in POLICY_FILES.items():
        csv_path = policy_dir / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing policy file: {csv_path}")
        frame = pd.read_csv(csv_path)
        frame["policy"] = policy_name
        frames.append(frame)

    rows = pd.concat(frames, ignore_index=True)
    rows["f1"] = pd.to_numeric(rows["f1"], errors="coerce")
    rows["ref_node_overlap_pct"] = pd.to_numeric(rows["ref_node_overlap_pct"], errors="coerce")
    return rows


def build_mapping_pair_reversals(policy_rows: pd.DataFrame) -> pd.DataFrame:
    """Count pairwise method-order reversals across policy variants (F1)."""
    summaries: list[dict[str, object]] = []

    for reference in sorted(policy_rows["reference"].dropna().unique()):
        ref_rows = policy_rows[policy_rows["reference"] == reference]
        policies = sorted(ref_rows["policy"].dropna().unique())

        for policy_a, policy_b in itertools.combinations(policies, 2):
            left = ref_rows[ref_rows["policy"] == policy_a].set_index("method")
            right = ref_rows[ref_rows["policy"] == policy_b].set_index("method")
            methods = sorted(set(left.index) & set(right.index))

            n_pairs = 0
            n_reversal = 0
            for method_a, method_b in itertools.combinations(methods, 2):
                delta_left = left.at[method_a, "f1"] - left.at[method_b, "f1"]
                delta_right = right.at[method_a, "f1"] - right.at[method_b, "f1"]

                if pd.isna(delta_left) or pd.isna(delta_right):
                    continue
                if delta_left == 0.0 or delta_right == 0.0:
                    continue

                n_pairs += 1
                if delta_left * delta_right < 0.0:
                    n_reversal += 1

            if n_pairs == 0:
                continue

            rate = n_reversal / n_pairs
            ci_low, ci_high = wilson_interval(n_reversal, n_pairs)
            summaries.append(
                {
                    "reference": reference,
                    "policy_a": policy_a,
                    "policy_b": policy_b,
                    "n_pairs": n_pairs,
                    "n_reversals": n_reversal,
                    "reversal_rate": rate,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )

    return pd.DataFrame(summaries)


def build_mapping_method_decomposition(policy_rows: pd.DataFrame) -> pd.DataFrame:
    """Method-level decomposition for mapping/candidate policy shifts.

    For each method/reference and policy pair:
      m2 - m1 = (c2-c1) * q1 + c2 * (q2-q1),  where m = c * q
    with c = overlap coverage fraction and q = coverage-adjusted quality.
    """
    rows: list[dict[str, object]] = []
    policy_order = list(POLICY_FILES.keys())

    for reference in sorted(policy_rows["reference"].dropna().unique()):
        ref_rows = policy_rows[policy_rows["reference"] == reference]
        for method in sorted(ref_rows["method"].dropna().unique()):
            method_rows = ref_rows[ref_rows["method"] == method].set_index("policy")
            for policy_a, policy_b in itertools.combinations(policy_order, 2):
                if policy_a not in method_rows.index or policy_b not in method_rows.index:
                    continue

                f1_a = method_rows.at[policy_a, "f1"]
                f1_b = method_rows.at[policy_b, "f1"]
                cov_a = method_rows.at[policy_a, "ref_node_overlap_pct"] / 100.0
                cov_b = method_rows.at[policy_b, "ref_node_overlap_pct"] / 100.0

                if pd.isna(f1_a) or pd.isna(f1_b) or pd.isna(cov_a) or pd.isna(cov_b):
                    continue
                if cov_a <= 0.0 or cov_b <= 0.0:
                    continue

                quality_a = f1_a / cov_a
                quality_b = f1_b / cov_b
                coverage_term = (cov_b - cov_a) * quality_a
                quality_term = cov_b * (quality_b - quality_a)
                delta = f1_b - f1_a

                rows.append(
                    {
                        "reference": reference,
                        "method": method,
                        "policy_a": policy_a,
                        "policy_b": policy_b,
                        "f1_a": f1_a,
                        "f1_b": f1_b,
                        "coverage_a": cov_a,
                        "coverage_b": cov_b,
                        "quality_a": quality_a,
                        "quality_b": quality_b,
                        "delta_f1": delta,
                        "coverage_term": coverage_term,
                        "quality_term": quality_term,
                        "decomposition_error": delta - coverage_term - quality_term,
                    }
                )

    decomposition = pd.DataFrame(rows)
    if not decomposition.empty:
        max_error = decomposition["decomposition_error"].abs().max()
        if max_error > 1e-9:
            raise ValueError(f"Mapping decomposition check failed; max error={max_error}")

    return decomposition


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute Proposal 2 ranking-reversal analysis.")
    parser.add_argument(
        "--report-markdown",
        default="reports/evaluation_bias_protocol/workshop/eval_bias_paper_draft.md",
        help="Path to workshop markdown containing Tables A2 and A3.",
    )
    parser.add_argument(
        "--policy-dir",
        default="subproject_02_evaluation_bias_protocol/implementation/network_inference_materials/outputs",
        help="Directory containing score_eval_probe_priors*.csv policy outputs.",
    )
    parser.add_argument(
        "--out-dir",
        default="market_research/ambitious_paper_questions/proposal_02_ranking_reversal_theory/artifacts",
        help="Output directory for generated CSV/JSON artifacts.",
    )
    args = parser.parse_args()

    report_markdown = Path(args.report_markdown)
    policy_dir = Path(args.policy_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table_a2, table_a3 = load_candidate_tables(report_markdown)
    pair_rows = build_candidate_pair_rows(table_a2, table_a3)

    reversal_by_transition = summarize_reversals(
        pair_rows,
        group_cols=["gene_set", "shift_from", "shift_to"],
        label="candidate_shift",
    )
    reversal_overall = summarize_reversals(
        pair_rows,
        group_cols=[],
        label="candidate_shift_overall",
    )

    instability_thresholds, instability_tagged, instability_perf = build_instability_regions(pair_rows)

    policy_rows = load_mapping_policy_rows(policy_dir)
    mapping_pair_reversals = build_mapping_pair_reversals(policy_rows)
    mapping_decomposition = build_mapping_method_decomposition(policy_rows)

    pair_rows.sort_values(
        ["gene_set", "shift_from", "shift_to", "method_a", "method_b"]
    ).to_csv(out_dir / "candidate_shift_pairwise_decomposition.csv", index=False)
    reversal_by_transition.to_csv(out_dir / "candidate_shift_reversal_summary.csv", index=False)
    reversal_overall.to_csv(out_dir / "candidate_shift_reversal_overall.csv", index=False)
    instability_thresholds.to_csv(out_dir / "instability_thresholds.csv", index=False)
    instability_tagged.sort_values(
        ["quantile", "gene_set", "shift_from", "shift_to", "method_a", "method_b"]
    ).to_csv(out_dir / "instability_classification.csv", index=False)
    instability_perf.to_csv(out_dir / "instability_performance.csv", index=False)
    mapping_pair_reversals.to_csv(out_dir / "mapping_policy_pair_reversals.csv", index=False)
    mapping_decomposition.sort_values(
        ["reference", "method", "policy_a", "policy_b"]
    ).to_csv(out_dir / "mapping_policy_method_decomposition.csv", index=False)

    overall_row = reversal_overall.iloc[0].to_dict()
    summary = {
        "candidate_pair_rows": int(len(pair_rows)),
        "candidate_reversal_rate": float(overall_row["reversal_rate"]),
        "candidate_reversal_count": int(overall_row["n_reversals"]),
        "candidate_pair_count": int(overall_row["n_pairs"]),
        "mapping_policy_rows": int(len(mapping_pair_reversals)),
        "mapping_policy_total_reversals": int(mapping_pair_reversals["n_reversals"].sum()) if not mapping_pair_reversals.empty else 0,
        "instability_quantiles": sorted(instability_thresholds["quantile"].unique().tolist()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Wrote Proposal 2 artifacts to", out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

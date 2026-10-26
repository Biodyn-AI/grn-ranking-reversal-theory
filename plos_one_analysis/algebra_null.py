"""Audit product accounting and run seeded controls for candidate-shift ranks.

The accounting identity is descriptive. In particular, a sign reversal and an
increasing positive base rate force the gap term to exceed the base-rate term.
The simulations demonstrate this constraint; they are not biological evidence.

The label-permutation control assigns the observed scores to method labels
independently within each tissue/candidate cell. All comparisons sharing a cell
use that same assignment. Gaussian score perturbations are sensitivity analyses
at arbitrary scales, not estimates of replicate-level measurement uncertainty.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.integrate import quad, trapezoid
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import norm


DEFAULT_ROOT = Path(__file__).resolve().parent


def decompose(b1, b2, g1, g2) -> pd.DataFrame:
    """Compute the exact forward product rule without epsilon-denominator bias.

    A zero base-rate term gives an infinite magnitude ratio when the gap term
    is nonzero, and an undefined ratio when both terms vanish. Nonpositive base
    rates are outside the scope of the sign and magnitude identities.
    """
    b1, b2, g1, g2 = [np.atleast_1d(values).astype(float) for values in np.broadcast_arrays(b1, b2, g1, g2)]
    if not all(np.isfinite(values).all() for values in (b1, b2, g1, g2)):
        raise ValueError("All endpoints must be finite")
    if np.any(b1 <= 0) or np.any(b2 <= 0):
        raise ValueError("Base rates must be strictly positive")
    delta1, delta2 = b1 * g1, b2 * g2
    base_term = (b2 - b1) * g1
    gap_term = b2 * (g2 - g1)
    with np.errstate(divide="ignore", invalid="ignore"):
        magnitude_ratio = np.abs(gap_term) / np.abs(base_term)
    return pd.DataFrame({
        "b1": b1,
        "b2": b2,
        "g1": g1,
        "g2": g2,
        "delta1": delta1,
        "delta2": delta2,
        "base_term": base_term,
        "gap_term": gap_term,
        "reversal": delta1 * delta2 < 0,
        "gap_dominant": np.abs(gap_term) > np.abs(base_term),
        "magnitude_ratio": magnitude_ratio,
        "closure_error": np.abs(delta2 - delta1 - base_term - gap_term),
    })


def audit_directions(candidate_rows: pd.DataFrame) -> pd.DataFrame:
    """Recompute accounting in both directions while retaining comparison keys."""
    audited_rows = []
    key_columns = ["gene_set", "shift_from", "shift_to", "method_a", "method_b", "method_pair"]
    for direction, reverse in [("recorded", False), ("reversed", True)]:
        first, second = ("to", "from") if reverse else ("from", "to")
        accounting = decompose(
            candidate_rows[f"base_rate_{first}"], candidate_rows[f"base_rate_{second}"],
            candidate_rows[f"gap_{first}"], candidate_rows[f"gap_{second}"],
        )
        for column in key_columns:
            accounting[column] = candidate_rows[column].to_numpy()
        if reverse:
            accounting[["shift_from", "shift_to"]] = accounting[["shift_to", "shift_from"]].to_numpy()
        accounting["direction"] = direction
        audited_rows.append(accounting)
    return pd.concat(audited_rows, ignore_index=True)


def summarize_accounting(accounting: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Report conditional dominance only when a group contains reversals."""
    summaries = []
    for group_key, group in accounting.groupby(group_columns, sort=False):
        group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        reversal = group[group["reversal"]]
        nonreversal = group[~group["reversal"]]
        summaries.append({
            **dict(zip(group_columns, group_key)),
            "n_rows": len(group),
            "base_rate_increases": int((group["b2"] > group["b1"]).sum()),
            "base_rate_equal": int((group["b2"] == group["b1"]).sum()),
            "base_rate_decreases": int((group["b2"] < group["b1"]).sum()),
            "reversals": len(reversal),
            "reversal_rate": len(reversal) / len(group),
            "dominant_on_reversal": int(reversal["gap_dominant"].sum()),
            "dominance_fraction_on_reversal": reversal["gap_dominant"].mean(),
            "median_ratio_on_reversal": reversal["magnitude_ratio"].median(),
            "dominant_on_nonreversal": int(nonreversal["gap_dominant"].sum()),
            "median_ratio_on_nonreversal": nonreversal["magnitude_ratio"].median(),
            "max_closure_error": group["closure_error"].max(),
        })
    return pd.DataFrame(summaries)


def simulate_gap_controls(candidate_rows: pd.DataFrame, n_draws: int, seed: int) -> pd.DataFrame:
    """Pair positive-rate draws with fixed, same-sign and sign-switched gaps.

    Sample the observed base-rate endpoint pairs with replacement. A normalized
    gap-magnitude ratio c is log-uniform on [0.01,100], its initial sign is fair,
    and its scale keeps all simulated margins within [-0.05,0.05]. Each sampled
    draw is evaluated in increasing and decreasing base-rate directions.
    """
    rng = np.random.default_rng(seed)
    source_index = rng.integers(0, len(candidate_rows), size=n_draws)
    sampled_b1 = candidate_rows["base_rate_from"].to_numpy()[source_index]
    sampled_b2 = candidate_rows["base_rate_to"].to_numpy()[source_index]
    gap_ratio = 10 ** rng.uniform(-2, 2, size=n_draws)
    initial_sign = rng.choice([-1, 1], size=n_draws)
    margin_scale = 10 ** rng.uniform(-5, np.log10(0.05), size=n_draws)
    gap_magnitude = margin_scale / (np.maximum(sampled_b1, sampled_b2) * np.maximum(1, gap_ratio))
    initial_gap = initial_sign * gap_magnitude
    draws = []
    for direction, b1, b2 in [("increasing", sampled_b1, sampled_b2), ("decreasing", sampled_b2, sampled_b1)]:
        for scenario, g2 in [
            ("fixed_gap", initial_gap),
            ("varying_gap_same_sign", gap_ratio * initial_gap),
            ("controlled_sign_switch", -gap_ratio * initial_gap),
        ]:
            accounting = decompose(b1, b2, initial_gap, g2)
            accounting["direction"] = direction
            accounting["scenario"] = scenario
            accounting["draw"] = np.arange(n_draws)
            draws.append(accounting)
    return pd.concat(draws, ignore_index=True)


def build_score_grid(score_rows: pd.DataFrame, candidate_rows: pd.DataFrame) -> dict:
    """Align actual method-cell scores to the archived comparison rows.

    Duplicate method-cell entries or missing endpoints are errors: silently
    replacing either would change the comparison unit or the shared-cell null.
    """
    cell_keys = ["gene_set", "candidate_set", "prediction_method"]
    if score_rows.duplicated(cell_keys).any():
        raise ValueError("Duplicate method-cell scores")
    cells = sorted(set(zip(score_rows["gene_set"], score_rows["candidate_set"])))
    methods = sorted(score_rows["prediction_method"].unique())
    cell_index = {cell: index for index, cell in enumerate(cells)}
    method_index = {method: index for index, method in enumerate(methods)}
    scores = np.full((len(cells), len(methods)), np.nan)
    for row in score_rows.itertuples(index=False):
        scores[cell_index[(row.gene_set, row.candidate_set)], method_index[row.prediction_method]] = row.aupr_median
    if not np.isfinite(scores).all():
        raise ValueError("A complete finite method-cell grid is required")
    comparisons = np.asarray([
        [cell_index[(row.gene_set, row.shift_from)], cell_index[(row.gene_set, row.shift_to)],
         method_index[row.method_a], method_index[row.method_b]]
        for row in candidate_rows.itertuples(index=False)
    ], dtype=int)
    return {"scores": scores, "comparisons": comparisons, "cells": cells, "methods": methods}


def comparison_status(scores: np.ndarray, comparisons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return eligibility and reversal arrays for all draws and comparisons."""
    first_cell, second_cell, first_method, second_method = comparisons.T
    delta1 = scores[:, first_cell, first_method] - scores[:, first_cell, second_method]
    delta2 = scores[:, second_cell, first_method] - scores[:, second_cell, second_method]
    eligible = (delta1 != 0) & (delta2 != 0)
    return eligible, (delta1 * delta2 < 0) & eligible


def permute_scores(scores: np.ndarray, n_draws: int, seed: int) -> np.ndarray:
    """Independently reassign the entire score vector once per cell and draw."""
    rng = np.random.default_rng(seed)
    assignments = rng.random((n_draws, *scores.shape)).argsort(axis=2)
    return np.take_along_axis(np.broadcast_to(scores, assignments.shape), assignments, axis=2)


def label_permutation(grid: dict, n_draws: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monte Carlo control for absence of persistent method identity across cells."""
    eligible, reversal = comparison_status(permute_scores(grid["scores"], n_draws, seed), grid["comparisons"])
    observed_eligible, observed_reversal = comparison_status(grid["scores"][None, :, :], grid["comparisons"])
    rates = reversal.sum(axis=1) / eligible.sum(axis=1)
    observed_rate = observed_reversal.sum() / observed_eligible.sum()
    left_count = int((rates <= observed_rate).sum())
    draws = pd.DataFrame({
        "draw": np.arange(n_draws), "eligible_comparisons": eligible.sum(axis=1),
        "reversals": reversal.sum(axis=1), "reversal_rate": rates,
    })
    summary = pd.DataFrame([{
        "control": "independent_method_label_permutation_within_tissue_candidate_cell",
        "n_draws": n_draws, "seed": seed, "n_cells": len(grid["cells"]), "n_methods": len(grid["methods"]),
        "observed_eligible": int(observed_eligible.sum()), "observed_reversals": int(observed_reversal.sum()),
        "observed_rate": observed_rate, "null_mean_rate": rates.mean(),
        "null_q025": np.quantile(rates, 0.025), "null_q975": np.quantile(rates, 0.975),
        "left_tail_draws": left_count, "left_tail_mc_p": (left_count + 1) / (n_draws + 1),
    }])
    return draws, summary


def noise_sensitivity(grid: dict, n_draws: int, seed: int, fractions=(0.0, 0.05, 0.10, 0.20)) -> pd.DataFrame:
    """Track retained, lost and new reversals under shared method-cell noise.

    The scale is a fraction of the median absolute pairwise score difference
    within each cell. Use the same standard-normal draws at every scale, and
    compare unbounded additive noise with clipping perturbed scores to [0,1].
    These arbitrary perturbations do not estimate unknown replicate variance.
    """
    scores = grid["scores"]
    first_method, second_method = np.triu_indices(scores.shape[1], k=1)
    cell_scale = np.median(np.abs(scores[:, first_method] - scores[:, second_method]), axis=1)
    normal_draws = np.random.default_rng(seed).normal(size=(n_draws, *scores.shape))
    _, observed_reversal = comparison_status(scores[None, :, :], grid["comparisons"])
    observed_reversal = observed_reversal[0]
    summaries = []
    for fraction in fractions:
        unbounded_scores = scores[None, :, :] + fraction * cell_scale[None, :, None] * normal_draws
        outside_bounds = (unbounded_scores < 0) | (unbounded_scores > 1)
        for treatment in ["additive_unbounded", "clipped_0_1"]:
            noisy_scores = unbounded_scores if treatment == "additive_unbounded" else np.clip(unbounded_scores, 0, 1)
            eligible, reversal = comparison_status(noisy_scores, grid["comparisons"])
            retained = (reversal & observed_reversal[None, :]).sum(axis=1)
            new_reversal = (reversal & ~observed_reversal[None, :]).sum(axis=1)
            eligible_count = eligible.sum(axis=1)
            reversal_rate = np.full(n_draws, np.nan)
            np.divide(reversal.sum(axis=1), eligible_count, out=reversal_rate, where=eligible_count != 0)
            summaries.append(pd.DataFrame({
                "draw": np.arange(n_draws), "noise_fraction": fraction, "treatment": treatment,
                "eligible_comparisons": eligible_count,
                "tied_comparisons": (~eligible).sum(axis=1),
                "reversals": reversal.sum(axis=1),
                "reversal_rate": reversal_rate,
                "retained_reversals": retained,
                "lost_reversals": observed_reversal.sum() - retained,
                "new_reversals": new_reversal,
                "fraction_method_cells_outside_bounds": outside_bounds.mean(axis=(1, 2)),
            }))
    return pd.concat(summaries, ignore_index=True)


def summarize_noise(draws: pd.DataFrame) -> pd.DataFrame:
    """Summarize perturbation distributions, retaining changes in denominators."""
    summaries = []
    measures = ["reversal_rate", "eligible_comparisons", "tied_comparisons", "reversals", "retained_reversals",
                "lost_reversals", "new_reversals", "fraction_method_cells_outside_bounds"]
    for (fraction, treatment), group in draws.groupby(["noise_fraction", "treatment"], sort=True):
        summary = {"noise_fraction": fraction, "treatment": treatment, "n_draws": len(group)}
        for measure in measures:
            summary[f"{measure}_mean"] = group[measure].mean()
            summary[f"{measure}_q025"] = group[measure].quantile(0.025)
            summary[f"{measure}_q975"] = group[measure].quantile(0.975)
        summaries.append(summary)
    return pd.DataFrame(summaries)


def fixed_gaussian_population(base_rates, n_tail: int = 8000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Integrate population PR curves with fixed class-conditional scores.

    Both methods have negative-class scores N(0,1). Method A's positive scores
    are N(0.5,2^2), and B's are N(2,0.5^2). A has the stronger far upper tail;
    B has the larger overall ROC area. Their ROC curves cross. Changing only
    prevalence can therefore reverse AUPR rankings without changing either
    class-conditional distribution or ROC curve. This is an illustrative
    population counterexample, not a fit to any biological score distribution.

    For recall r, threshold = mu + sigma*Phi^{-1}(1-r), FPR=1-Phi(threshold),
    and precision = b*r/[b*r+(1-b)*FPR]. A dense grid resolves both recall tails.
    Log-survival probabilities avoid underflow at extremely high thresholds.
    """
    prevalence = np.atleast_1d(base_rates).astype(float)
    if np.any(~np.isfinite(prevalence)) or np.any((prevalence <= 0) | (prevalence >= 1)):
        raise ValueError("Population base rates must be finite and strictly between zero and one")
    recall = np.unique(np.concatenate((
        [0.0], np.logspace(-12, -2, n_tail), np.linspace(0.01, 0.99, 4 * n_tail),
        1 - np.logspace(-12, -2, n_tail), [1.0],
    )))
    population = pd.DataFrame({"base_rate": prevalence})
    roc_rows = []
    for method, positive_mean, positive_sd in [("A", 0.5, 2.0), ("B", 2.0, 0.5)]:
        threshold = positive_mean + positive_sd * norm.isf(recall)
        log_fpr = norm.logsf(threshold[1:])
        precision = expit(np.log(prevalence[:, None] / (1 - prevalence[:, None]))
                          + np.log(recall[None, 1:]) - log_fpr[None, :])
        # The r->0 precision limit is 1 for A's heavier upper tail and 0 for B.
        endpoint_precision = np.ones(len(prevalence)) if positive_sd > 1 else np.zeros(len(prevalence))
        precision = np.column_stack((endpoint_precision, precision))
        population[f"aupr_{method.lower()}"] = trapezoid(precision, recall, axis=1)
        population[f"auroc_{method.lower()}"] = norm.cdf(positive_mean / np.sqrt(1 + positive_sd ** 2))
        roc_rows.append(pd.DataFrame({
            "method": method, "positive_mean": positive_mean, "positive_sd": positive_sd,
            "negative_mean": 0.0, "negative_sd": 1.0, "threshold": threshold,
            "recall": recall, "false_positive_rate": norm.sf(threshold),
        }))
    population["aupr_margin_a_minus_b"] = population["aupr_a"] - population["aupr_b"]
    population["normalized_gap"] = population["aupr_margin_a_minus_b"] / population["base_rate"]
    return population, pd.concat(roc_rows, ignore_index=True)


def save_prevalence_figure(population: pd.DataFrame, roc: pd.DataFrame, observed_base_rates: np.ndarray,
                           crossing_prevalence: float, figure_dir: Path) -> None:
    """Keep the population illustration distinct from observed GRN evidence."""
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), layout="constrained")
    colors = {"A": "#2166ac", "B": "#b2182b"}
    for method in ["A", "B"]:
        curve = roc[roc["method"] == method]
        auc = population[f"auroc_{method.lower()}"][0]
        axes[0].plot(curve["false_positive_rate"], curve["recall"], color=colors[method],
                     label=f"{method}: AUROC {auc:.3f}")
        axes[1].plot(population["base_rate"], population[f"aupr_{method.lower()}"], color=colors[method], label=method)
        actual_b = population[population["base_rate"].isin(observed_base_rates)]
        axes[1].scatter(actual_b["base_rate"], actual_b[f"aupr_{method.lower()}"], color=colors[method], s=20)
    axes[0].set(xscale="log", xlim=(1e-8, 1), ylim=(0, 1), xlabel="False-positive rate", ylabel="True-positive rate (recall)",
                title="A  Fixed class-conditional ROC curves")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axvspan(min(observed_base_rates), max(observed_base_rates), color="grey", alpha=0.12,
                    label="Observed base-rate range")
    axes[1].axvline(crossing_prevalence, color="black", linestyle="--", linewidth=1)
    axes[1].set(xscale="log", yscale="log", xlabel="Positive-class prevalence", ylabel="Population AUPR",
                title="B  Changing prevalence only")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    save_figure_files(figure, figure_dir / "algebra_prevalence_only")
    plt.close(figure)


def validate_population_integrals(population: pd.DataFrame) -> pd.DataFrame:
    """Cross-check recall integration against precision times the positive PDF.

    Integrating over thresholds is an independent numerical representation of
    the same population AUPR. The fixed normal distributions have negligible
    mass outside [-25,25], so adaptive quadrature provides a precise check.
    """
    validations = []
    for row in population.itertuples(index=False):
        for method, positive_mean, positive_sd in [("A", 0.5, 2.0), ("B", 2.0, 0.5)]:
            def integrand(threshold):
                log_tpr = norm.logsf((threshold - positive_mean) / positive_sd)
                log_fpr = norm.logsf(threshold)
                precision = expit(np.log(row.base_rate / (1 - row.base_rate)) + log_tpr - log_fpr)
                positive_pdf = norm.pdf((threshold - positive_mean) / positive_sd) / positive_sd
                return precision * positive_pdf
            threshold_integral, error_estimate = quad(integrand, -25, 25, epsabs=1e-11, epsrel=1e-10)
            recall_integral = getattr(row, f"aupr_{method.lower()}")
            validations.append({
                "base_rate": row.base_rate, "method": method, "aupr_recall_grid": recall_integral,
                "aupr_threshold_quadrature": threshold_integral,
                "absolute_difference": abs(recall_integral - threshold_integral),
                "quadrature_error_estimate": error_estimate,
            })
    return pd.DataFrame(validations)


def save_figure_files(figure, stem: Path) -> None:
    """Write a preview PNG and a compressed RGB TIFF without an alpha channel."""
    figure.savefig(stem.with_suffix(".png"), dpi=300)
    figure.savefig(stem.with_suffix(".tif"), dpi=300)
    with Image.open(stem.with_suffix(".tif")) as rendered:
        rgb = rendered.convert("RGB")
    rgb.save(stem.with_suffix(".tif"), compression="tiff_lzw", dpi=(300, 300))


def save_figures(audit: pd.DataFrame, controls: pd.DataFrame, label_draws: pd.DataFrame,
                 label_summary: pd.DataFrame, noise_summary: pd.DataFrame, output_dir: Path) -> None:
    """Generate detached 300-DPI figures with explicit data/control labels."""
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 9,
                         "axes.titlesize": 9, "axes.spines.top": False, "axes.spines.right": False})
    observed_reversal = audit[audit["reversal"]]
    figure, axes = plt.subplots(1, 3, figsize=(7.4, 3.4), layout="constrained")
    for index, (direction, label) in enumerate([("recorded", "Recorded\nincreasing b"), ("reversed", "Reversed\ndecreasing b")]):
        ratios = observed_reversal.loc[observed_reversal["direction"] == direction, "magnitude_ratio"].to_numpy()
        offset = np.linspace(-0.10, 0.10, len(ratios))
        axes[0].scatter(index + offset, ratios, s=19, alpha=0.8, color=["#2166ac", "#b2182b"][index])
        axes[0].text(index, 60, f"{(ratios > 1).sum()}/{len(ratios)}", ha="center", fontweight="bold")
    axes[0].axhline(1, color="black", linestyle="--", linewidth=1)
    axes[0].set(yscale="log", ylim=(0.002, 100), xticks=[0, 1], xticklabels=["Recorded\nincreasing b", "Reversed\ndecreasing b"],
                ylabel=r"$|T_g|/|T_b|$", title="A  Observed reversals")
    axes[0].text(0.5, 0.02, "22 reversals in each direction", transform=axes[0].transAxes, ha="center", fontsize=8)
    increasing_controls = controls[controls["direction"] == "increasing"]
    axes[1].bar(np.arange(3), increasing_controls["reversal_rate"], color=["#969696", "#969696", "#2166ac"])
    axes[1].set(ylim=(0, 1.15), xticks=[0, 1, 2], xticklabels=["Fixed\ng", "Same\nsign", "Sign\nflip"],
                ylabel="Reversal fraction", title="B  Positive-rate controls")
    for index, row in enumerate(increasing_controls.itertuples(index=False)):
        axes[1].text(index, row.reversal_rate + 0.03, f"{row.reversals:,}/{row.n_rows:,}", ha="center", fontsize=8)
    switch_controls = controls[controls["scenario"] == "controlled_sign_switch"]
    axes[2].bar([0, 1], switch_controls["dominance_fraction_on_reversal"], color=["#2166ac", "#b2182b"])
    axes[2].set(ylim=(0, 1.15), xticks=[0, 1], xticklabels=["Increasing b", "Decreasing b"],
                ylabel=r"Fraction with $|T_g|>|T_b|$", title="C  Sign-switch controls")
    for index, row in enumerate(switch_controls.itertuples(index=False)):
        axes[2].text(index, row.dominance_fraction_on_reversal + 0.03,
                     f"{row.dominant_on_reversal:,}/{row.reversals:,}", ha="center", fontsize=8)
    save_figure_files(figure, figure_dir / "algebra_structure")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), layout="constrained")
    observed_rate = label_summary.loc[0, "observed_rate"]
    # Reversal counts take discrete values; one bar per attainable count avoids merged histogram bins.
    eligible = label_draws["eligible_comparisons"].unique()
    if len(eligible) != 1:
        raise ValueError("label permutation must keep a fixed denominator")
    counts, draws = np.unique(label_draws["reversals"], return_counts=True)
    axes[0].bar(counts / eligible[0], draws, width=1.6 / eligible[0], color="#8da0cb", edgecolor="white")
    axes[0].axvline(observed_rate, color="#b2182b", linewidth=2)
    axes[0].annotate(f"Observed: {observed_rate:.3f}", xy=(observed_rate, 0.97), xycoords=("data", "axes fraction"),
                     xytext=(5, 0), textcoords="offset points", ha="left", va="top", color="#b2182b", fontsize=8)
    axes[0].set(xlabel="Candidate-transition reversal fraction", ylabel="Permutation draws",
                title="A  Within-cell method-label permutation")
    bounded = noise_summary[noise_summary["treatment"] == "clipped_0_1"]
    for measure, label, color in [("retained_reversals", "Retained", "#1b9e77"),
                                  ("lost_reversals", "Lost", "#d95f02"), ("new_reversals", "New", "#7570b3")]:
        axes[1].plot(bounded["noise_fraction"], bounded[f"{measure}_mean"], "o-", label=label, color=color)
        axes[1].fill_between(bounded["noise_fraction"], bounded[f"{measure}_q025"],
                             bounded[f"{measure}_q975"], alpha=0.12, color=color)
    axes[1].set(xlabel="Noise SD / cell median absolute pair margin", ylabel="Reversal-row count",
                title="B  Gaussian score sensitivity (clipped)", xticks=bounded["noise_fraction"])
    axes[1].legend(frameon=False, ncol=3, fontsize=8)
    save_figure_files(figure, figure_dir / "algebra_label_noise_controls")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--n-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    if arguments.n_draws < 1:
        parser.error("--n-draws must be positive")
    input_dir = arguments.project_root / "inputs"
    output_dir = arguments.project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows = pd.read_csv(input_dir / "candidate_shift_pairwise_rows.csv")
    score_rows = pd.read_csv(input_dir / "table_a3.csv")
    audit = audit_directions(candidate_rows)
    audit_summary = summarize_accounting(audit, ["direction"])
    controls = simulate_gap_controls(candidate_rows, arguments.n_draws, arguments.seed)
    control_summary = summarize_accounting(controls, ["direction", "scenario"])
    grid = build_score_grid(score_rows, candidate_rows)
    first_cell, second_cell, first_method, second_method = grid["comparisons"].T
    for endpoint, cell_index in [("from", first_cell), ("to", second_cell)]:
        reconstructed_margin = grid["scores"][cell_index, first_method] - grid["scores"][cell_index, second_method]
        np.testing.assert_allclose(reconstructed_margin, candidate_rows[f"delta_{endpoint}"], atol=1e-14, rtol=0)
    # Distinct deterministic streams keep experiments stable when another is extended.
    label_draws, label_summary = label_permutation(grid, arguments.n_draws, arguments.seed + 1)
    noise_draws = noise_sensitivity(grid, arguments.n_draws, arguments.seed + 2)
    noise_summary = summarize_noise(noise_draws)
    observed_base_rates = np.unique(np.concatenate((candidate_rows["base_rate_from"], candidate_rows["base_rate_to"])))
    population_base_rates = np.unique(np.r_[np.geomspace(1e-6, 0.5, 151), observed_base_rates])
    population, roc = fixed_gaussian_population(population_base_rates)
    coarser_population, _ = fixed_gaussian_population(population_base_rates, n_tail=4000)
    integration_difference = np.max(np.abs(population[["aupr_a", "aupr_b"]].to_numpy()
                                            - coarser_population[["aupr_a", "aupr_b"]].to_numpy()))
    crossing_prevalence = brentq(
        lambda base_rate: fixed_gaussian_population([base_rate])[0]["aupr_margin_a_minus_b"].iloc[0],
        min(observed_base_rates), max(observed_base_rates), xtol=1e-12,
    )
    population["is_observed_base_rate"] = population["base_rate"].isin(observed_base_rates)
    population_validation = validate_population_integrals(population[population["is_observed_base_rate"]])
    if population_validation["absolute_difference"].max() > 5e-8:
        raise AssertionError("Population AUPR integrations disagree beyond 5e-8")
    for name, frame in [("algebra_direction_rows", audit), ("algebra_direction_summary", audit_summary),
                        ("algebra_gap_control_draws", controls), ("algebra_gap_control_summary", control_summary),
                        ("algebra_label_permutation_draws", label_draws), ("algebra_label_permutation_summary", label_summary),
                        ("algebra_noise_draws", noise_draws), ("algebra_noise_summary", noise_summary),
                        ("algebra_population_prevalence", population), ("algebra_population_roc", roc),
                        ("algebra_population_validation", population_validation)]:
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    proof = {
        "seed_gap_controls": arguments.seed, "seed_label_control": arguments.seed + 1, "seed_noise_sensitivity": arguments.seed + 2,
        "draws_per_control": arguments.n_draws,
        "identity": "delta2-delta1=(b2-b1)*g1+b2*(g2-g1)",
        "reversal_ratio": "With r=b2/b1>0 and t=abs(delta2)/abs(delta1)>0, abs(Tg)/abs(Tb)=(r+t)/abs(r-1).",
        "increasing_rates": "If r>=1, abs(Tg)>abs(Tb) is forced on all reversal rows (Tb=0 if r=1).",
        "decreasing_rates": "If 0<r<1, abs(Tg)>abs(Tb) iff 2*r+t>1; equality and either direction are possible.",
        "fixed_gap_null": "For g2=g1 and positive b1,b2, delta1*delta2=b1*b2*g1^2>=0: reversal is impossible; conditional dominance is undefined.",
        "fixed_gap_scope": "Fixed g=delta/b is not fixed statistical discrimination. AUPR can change rank under prevalence alone even when class-conditional score distributions and ROC curves are fixed.",
        "population_control_parameters": {"negative_mean_both_methods": 0.0, "negative_sd_both_methods": 1.0,
                                          "positive_mean_A": 0.5, "positive_sd_A": 2.0,
                                          "positive_mean_B": 2.0, "positive_sd_B": 0.5},
        "population_control_precision": "precision=b*TPR/(b*TPR+(1-b)*FPR)",
        "population_control_aupr_crossing_prevalence": crossing_prevalence,
        "population_control_max_dense_coarse_integration_difference": integration_difference,
        "population_control_max_recall_threshold_integration_difference": population_validation["absolute_difference"].max(),
        "population_control_scope": "Illustrative population distributions chosen to have crossing ROC curves, not fitted to GRN scores; no biological mechanism follows from this counterexample.",
        "population_control_parameter_selection": "A coarse exploratory grid of positive-class means {0.5,1,1.5,2,2.5,3} and SDs {0.5,0.75,1,1.5,2,2.5,3} identified a pair chosen to exhibit an AUPR ranking crossover within the observed prevalence range. This is a constructed existence example, not an estimate of how often prevalence-only reversals occur.",
        "safety_remark": "Reversal implies abs(delta2-delta1)>abs(delta1). A bound B<=abs(delta1) therefore excludes reversal. The triangle bound abs(b2-b1)*abs(g1)+b2*abs(g2-g1)>abs(delta1) is necessary, not sufficient, and needs externally justified bounds to predict future shifts.",
        "limitations": "g=delta/b is defined by division and is not an independently measured mechanism. Controls are illustrative, and Gaussian scales are arbitrary summary-score sensitivities; bands are perturbation quantiles, not confidence intervals.",
    }
    (output_dir / "algebra_proof_and_design.json").write_text(json.dumps(proof, indent=2) + "\n")
    save_figures(audit, control_summary, label_draws, label_summary, noise_summary, output_dir)
    save_prevalence_figure(population, roc, observed_base_rates, crossing_prevalence, output_dir / "figures")
    print(audit_summary.to_string(index=False))
    print(control_summary.to_string(index=False))
    print(label_summary.to_string(index=False))
    print(noise_summary[["noise_fraction", "treatment", "reversal_rate_mean", "retained_reversals_mean",
                         "lost_reversals_mean", "new_reversals_mean", "tied_comparisons_mean"]].to_string(index=False))
    print(population[population["is_observed_base_rate"]].to_string(index=False))
    print(f"Population AUPR crossing prevalence: {crossing_prevalence:.10f}; integration check: {integration_difference:.3g}")


if __name__ == "__main__":
    main()

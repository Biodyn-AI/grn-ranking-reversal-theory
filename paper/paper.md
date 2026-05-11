# Ranking Reversal Theory Under Candidate-Set Shift in GRN Benchmarking

## Abstract
Gene regulatory network (GRN) benchmarks are routinely used to justify scientific claims about method quality, yet the stability of method *ranking* under plausible evaluation-protocol choices is rarely audited. We develop a framework for analyzing pairwise ranking reversals under protocol shift, supply a minimum-shift necessary condition for reversal, and quantify reversal behavior on existing single-cell GRN benchmark outputs across four protocol axes: candidate-set restriction, tissue, reference network, and gene-symbol mapping policy. Using cluster-bootstrap confidence intervals over method pairs — which correctly account for the dependence Wilson intervals miss — we find candidate-set shifts produce pairwise reversals in 16% of method pairs (cluster-bootstrap 95% CI 7–27%), tissue shifts in 19% (CI 9–31%), and reference-network shifts in 32% (CI 20–43%); mapping-policy shifts produce 0/165. The decomposition `Δ = b · g` (base rate × discrimination) is structural and forces certain sign-attribution claims under positive base rates; we therefore report the genuinely empirical magnitude content: in every observed candidate-shift reversal the calibration-term magnitude exceeds the base-rate-term magnitude (22/22), versus only 56/113 in non-reversal rows; under a cluster-restricted permutation Mann–Whitney test that respects the method-pair dependence between rows (5,000 permutations), the magnitude advantage is highly significant (rank-biserial *r* = 0.67, permutation *p* < 2 × 10⁻⁴). A margin-magnitude sweep shows that no reversals occur above the 75th percentile of pair-margin magnitudes for candidate shifts, indicating the phenomenon is concentrated in near-tie pairs. AUROC produces 0/40 tissue-shift reversals where AUPR produces 4/45, suggesting reversal risk depends materially on metric choice. The practical recommendation: benchmark rank should not be interpreted as method-intrinsic evidence until protocol-axis stability and margin-magnitude regimes are reported.

## 1. Introduction
Mechanistic interpretability and benchmarking for biological models increasingly use leaderboard rank in GRN inference to justify claims about biological plausibility and downstream utility [1, 2]. The evaluation pipeline, however, involves several protocol choices — which candidate edges are scored, which reference network is used, how gene identifiers are mapped, and which tissue context is evaluated — that are rarely reported or controlled [3, 6].

If ranking is unstable under plausible protocol variation, downstream biological decisions can flip: which regulators are prioritized for experimental validation, which mechanistic narrative is emphasized, and which model is treated as scientifically credible. The field therefore needs explicit ranking-stability theory and diagnostics, not larger metric tables.

We make three contributions. (1) We define the pairwise reversal indicator under protocol shift and supply a minimum-shift necessary condition giving a closed-form "safety margin" per pair (Section 3). (2) We separate the algebraic content of the standard `Δ = b · g` decomposition (which forces certain sign-attribution claims when base rates are positive) from its empirical content (the magnitudes of the two terms in reversal vs non-reversal rows), and report only the latter as findings (Section 5.2). (3) We quantify reversal behavior across four protocol axes with cluster-bootstrap confidence intervals over method pairs, three alternative null models, a per-axis margin-magnitude sweep, BH-FDR-adjusted per-cell rates, and a metric-robustness check (Sections 5.1–5.8). We treat the instability-region heuristic as a screening problem with an honest negative finding: neither a quantile heuristic nor a learned logistic-regression classifier exceeds baseline prevalence on this dataset (Section 5.7).

## 2. Related Work

**GRN benchmarking and evaluation bias.** Systematic GRN benchmarking has been addressed by [3, 4, 5], typically on fixed protocols. [6] showed that protocol choices such as symbol mapping and candidate-set restriction can shift AUPR by orders of magnitude, but focused on bias quantification rather than ranking stability. Our analysis is downstream: given those documented shifts, when does the leaderboard reorder?

**Benchmark sensitivity in machine learning.** The sensitivity of benchmark conclusions to evaluation choices is widely studied: BLEU rank instability under tokenization [11], benchmark-lottery effects [12], and rank stability under bootstrap [13]. We treat ranking reversal as a finite-margin event and supply an explicit minimum-shift bound, which the ML literature on rank stability has largely framed as average-case rather than worst-case.

**Rank stability in statistics.** Rank stability under sample perturbation has been studied via Kendall's τ, top-k stability bounds, and Critchlow's rank metrics [14]. Our cluster-bootstrap CI over method pairs is the rank-stability analogue of cluster-robust inference on pairwise comparisons.

## 3. Problem Setup and Theory

### 3.1 Notation and reversal definition
Let `M_m(S, π, R)` denote a scalar evaluation metric for method `m` under candidate set `S`, mapping policy `π`, and reference `R`. For methods `A` and `B`, define the **margin**:

`Δ = M_A − M_B`.

For two protocol settings (1, 2) with margins `Δ_1, Δ_2` and shift `δΔ = Δ_2 − Δ_1`, a **pairwise ranking reversal** is defined by `Δ_1 · Δ_2 < 0`. Equivalently, `sign(Δ_1) · δΔ < −|Δ_1|`: the shift must oppose the initial ordering and exceed the initial margin. This is a definitional restatement, not a theorem; we include it because it makes the **safety margin** `|Δ_1|` explicit — a single pair cannot reverse under any shift family `δΔ` whose magnitude bound `B` satisfies `B ≤ |Δ_1|`.

### 3.2 Decomposition identity (definition)
For a fixed mapping policy and reference, the metric admits the trivial product form `Δ(S) = b(S) · g(S)` with `g(S) := Δ(S)/b(S)` (assuming `b > 0`), where `b(S)` is the candidate-set positive base rate and `g(S)` is the base-rate-normalized discrimination gap. The finite product rule then gives an **exact** decomposition of `δΔ`:

`Δ_2 − Δ_1 = (b_2 − b_1) · g_1   +   b_2 · (g_2 − g_1)`.

- *base-rate term*: `(b_2 − b_1) · g_1`
- *calibration/discrimination term*: `b_2 · (g_2 − g_1)`

This is an algebraic identity, not a theorem. We caution explicitly that **certain sign-attribution claims from this decomposition are structurally pinned, not empirical**, whenever both base rates are positive (as in every dataset slice we examine):

- The "only-`b`-changes" counterfactual `Δ_1 · (b_2 g_1) < 0` simplifies to `(b_2/b_1) Δ_1² < 0`, which is **always false** for `b_1, b_2 > 0` and `Δ_1 ≠ 0`. So no observed reversal can ever be attributed to base-rate scaling alone — but this is a property of the decomposition under positive base rates, not a property of the data.
- The "only-`g`-changes" counterfactual `Δ_1 · (b_1 g_2) < 0` simplifies to `(b_1/b_2) Δ_1 Δ_2 < 0`, which is **always true** on reversal rows (where `Δ_1 Δ_2 < 0` by definition). So "discrimination change alone would have produced the reversal" holds 100% by construction.

The genuinely empirical content of the decomposition is the *magnitude* relationship between the two terms in reversal vs non-reversal rows (Section 5.2): although the directions are pinned, the magnitudes are not. The minimum-shift bound (Proposition 1 below) gives the necessary inequality that any reversal must satisfy, and the empirical question is whether the calibration-term magnitude systematically exceeds the base-rate-term magnitude in reversal rows.

### 3.3 Minimum-shift necessary condition (Proposition 1)

> **Proposition 1.** *For a shift family with bounded magnitude `|δΔ| ≤ B`, no pair with safety margin `|Δ_1| > B` can reverse. Furthermore, given the decomposition above, a necessary condition for reversal of a pair with margin `|Δ_1|` is*
>
> `|b_2 − b_1| · |g_1|   +   b_2 · |g_2 − g_1|   ≥   |Δ_1|.`
>
> *In particular, if g is stable under the shift (`|g_2 − g_1| ≤ ε_g`), then the base-rate change required to reverse the pair satisfies `|b_2 − b_1| ≥ (|Δ_1| − b_2 ε_g) / |g_1|`. If `b` is stable (`|b_2 − b_1| ≤ ε_b`), then the discrimination change required satisfies `|g_2 − g_1| ≥ (|Δ_1| − ε_b |g_1|) / b_2`.*
>
> *Proof.* The first claim is immediate from `|Δ_2 − Δ_1| ≤ B` and the reversal definition. The second follows from the decomposition and the triangle inequality applied to `δΔ`. The corollaries follow by isolating each term.

Proposition 1 is the closed-form "is my pair safe?" calculator. Combined with the empirical distributions of `|Δ_1|` and `|g_2 − g_1|` it yields per-pair safety bounds; the margin-threshold sweep in Section 5.6 visualizes the resulting safety regime by axis. Note that the corollaries are informative only when the right-hand sides are positive (e.g., the discrimination-stability corollary `|b_2 − b_1| ≥ (|Δ_1| − b_2 ε_g)/|g_1|` is informative only when `b_2 ε_g < |Δ_1|`; otherwise no constraint on `|b_2 − b_1|` is implied).

### 3.4 Reference-shift remark
For the immune reference-shift slice, base rates differ across references by absolute amount ≈ 4.7 × 10⁻⁴ (from ≈ 2 × 10⁻⁶ for `beeline_gsd` to ≈ 4.8 × 10⁻⁴ for `dorothea_trrust_union_immune`). Although the absolute range is non-trivial, the `b · g` decomposition is numerically unbalanced for reference shifts: empirically the mean magnitude ratio `|calibration term|/|base-rate term|` is ~10⁴ in both reversal and non-reversal rows (see `reference_shift_decomposition_summary.csv` from the prior pipeline). With one term dominating by four orders of magnitude regardless of reversal status, the decomposition offers no discrimination between mechanisms here. We do not apply the `b · g` decomposition to reference shifts; we report reversal counts only. A coverage-and-precision decomposition (`M = c · q` with `c` = reference-overlap fraction, `q` = within-overlap quality) is a natural alternative; we use it for mapping shifts (Section 5.5) where overlap varies materially.

## 4. Methods

### 4.1 Datasets and methods compared
Inputs come from the evaluation-bias workshop study [6]: tissue-stratified medians of AUPR, AUROC, and F1 across three Tabula Sapiens tissues (kidney, lung, immune) and three candidate-set definitions (`all_pairs`, `tf_sources`, `tf_sources_targets`). The methods compared are the six in Table A3 of that study: **genie3, grnboost2, pearson, spearman, scgpt_attention, random**. Reference-shift analysis uses the immune `score_eval_grn_baselines_immune` slice with three references (`hpn_dream`, `beeline_gsd`, `dorothea_trrust_union_immune`) and nine methods. Mapping-policy analysis uses four policies (`legacy_symbols`, `full_genes`, `crosswalk`, `omnipath_ref`) on the probe-prior subset.

All inputs are summary medians; replicate-level raw predictions are not available in this slice. Uncertainty therefore reflects between-pair structural variability, not within-method replicate noise (see §8).

### 4.2 Analyses
For each of the four protocol axes we compute:

1. **Pairwise reversal indicator** `1{Δ_1 · Δ_2 < 0}` per (method-pair, shift-cell).
2. **Cluster-bootstrap 95% CIs** with the method pair as the cluster, using 2,000 resamples of clusters; this addresses pair-level dependence that Wilson intervals assume away.
3. **Magnitude decomposition** (candidate shifts only): per reversal row, compute the calibration-vs-base-rate magnitude ratio and the Mann–Whitney comparison against non-reversal rows. We do not report sign-attribution counts because they are algebraically pinned under positive base rates (§3.2).
4. **Per-cell BH-FDR adjustment** on highlighted shift cells (two-sided test against rate = 0.5).
5. **Margin-magnitude threshold sweep**: reversal rate as a function of `τ`, the minimum of `|Δ_1|, |Δ_2|`, with cluster-bootstrap bands.
6. **Three null models** (candidate shift only): (a) joint method-relabel in c_2, (b) score-noise bootstrap with σ = 10% of within-cell median absolute margin, (c) rank permutation within (tissue × candidate) cell. (a) and (c) are the natural "is the rank signal real?" nulls; (b) targets the question "would these reversals have happened anyway under plausible AUPR noise?".
7. **Instability screening**: leave-one-tissue-out classifier evaluation, comparing the quantile heuristic (`predicted unstable iff |Δ_1| ≤ quantile_q of |δΔ|`) against a logistic regression learned on `(|Δ_1|, b_1, b_2 − b_1, |g_1|)` features. Report PR-AUC, not only operating-point precision/recall.
8. **Metric robustness**: repeat the tissue-shift analysis using AUPR, AUROC, and F1 (Table A1).

All numerical decomposition closure errors are < 1e−9 (verified in code; tolerance threshold checked at runtime). All seeds are fixed (project seed 42; permutation seed 42; bootstrap seed 42).

## 5. Results

A summary table of all four axes is given as Table 1.

### Table 1. Headline reversal rates by axis (cluster-bootstrap 95% CIs)

| Axis | n pairs | k reversals | Rate | Cluster CI | Metric | Dataset slice |
|---|---:|---:|---:|---|---|---|
| Candidate-set shift (cand-conditional) | 135 | 22 | 0.16 | [0.07, 0.27] | AUPR | A2/A3 (3 tissues × 3 cand × 6 methods) |
| Tissue shift (cand-conditional) | 135 | 26 | 0.19 | [0.09, 0.31] | AUPR | A3 (3 tissues × 3 cand × 6 methods) |
| Tissue shift (dedup, any candidate) | 45 | 17 | 0.38 | [0.18, 0.58] | AUPR | A3 (one row per method-pair × tissue-pair) |
| Tissue shift (dedup, all candidates) | 45 | 1 | 0.02 | [0.00, 0.07] | AUPR | A3 |
| Reference shift | 106 | 34 | 0.32 | [0.20, 0.43] | AUPR | immune GRN baselines, 9 methods, 3 refs |
| Mapping-policy shift | 165 | 0 | 0.00 | [0.00, 0.00] | F1 | probe priors, 4 policies |

Numerical precision is rounded to two decimals consistent with cluster-CI half-widths. The tissue-shift axis is reported with three rates because they answer different questions: the candidate-conditional 0.19 counts each method-pair × tissue-pair × candidate triple separately (and is therefore inflated by dependent observations); the dedup-any 0.38 counts a method-pair × tissue-pair as a reversal if it reverses under any candidate set; the dedup-all 0.02 counts a method-pair × tissue-pair only if it reverses under all candidate sets simultaneously. The honest single-number summary is the dedup-any rate, 0.38; we report all three for transparency.

### 5.1 Candidate-set shifts produce nontrivial reversal risk concentrated in near-tie pairs
**Evidence.** Candidate-set shifts produce 22/135 reversals (rate 0.16, cluster-bootstrap 95% CI [0.07, 0.27]). Per-cell BH-FDR-adjusted rates (`candidate_shift_cells_fdr.csv`) show that **none** of the high-reversal cells highlighted in earlier versions of this analysis (e.g., immune `all_pairs → tf_sources_targets` at 6/15 = 0.40) survive `q < 0.05` as significantly above 0.5; what survives is the opposite — cells significantly *more stable* than chance (kidney `tf_sources → tf_sources_targets` at 0/15, `q = 5 × 10⁻⁴`; lung `all_pairs → tf_sources` at 0/15, `q = 5 × 10⁻⁴`).

![Candidate-shift reversal rate by tissue × transition](figures/fig_candidate_heatmap.png)

**Inference.** Rank is not a method-intrinsic invariant; it is protocol-conditional. The marginal 16% rate is driven by the immune tissue and small-margin pairs (Section 5.6).

**Scientific implication.** A single leaderboard rank should not be used to justify biological claims without candidate-shift stability reporting, but the per-cell tail rates in the previous version of this analysis were not robust to multiple testing.

### 5.2 Calibration magnitude exceeds base-rate magnitude in every observed reversal (empirical)
**Structural note (what is *not* a finding).** As shown in Section 3.2, under positive base rates the decomposition `Δ = b · g` algebraically forces (i) the calibration term to have sign opposite to `Δ_1` in every reversal row, and (ii) the base-rate term to have sign aligned with `Δ_1`. These are properties of the decomposition, not of the data. We therefore do **not** report sign-attribution counts (0/22, 22/22, etc.) as findings; doing so was the central flaw of earlier versions of this analysis.

**Evidence (genuinely empirical).** What is *not* structurally pinned is the relative *magnitude* of the two terms. In our reversal rows we observe:

- `|calibration term| > |base-rate term|` in **22/22** reversal rows (rate 1.00), versus **56/113** in non-reversal rows (rate 0.50).
- Mean magnitude ratio `|cal|/|base-rate|` in reversal rows: **1.54**; median: **1.02**.
- Mean ratio in non-reversal rows: **0.83**; median: **1.00**.
- **Cluster-permutation Mann–Whitney U test** (5,000 permutations, restricted within method-pair clusters; one-sided "reversal-row ratios stochastically larger than non-reversal"): observed *U* = 2075, rank-biserial effect size *r* = 0.67 (large), permutation *p* < 2 × 10⁻⁴ (0/5000 null draws exceeded the observed *U*; null 95% interval [1426, 1907] vs observed 2075). For comparison, the standard iid MWU would yield *p* = 3.6 × 10⁻⁷ — but the iid assumption is invalid here because the 135 rows share method pairs across only 15 unique clusters, of which only 7 contain any reversal rows; the iid *p* is therefore too liberal and we report the cluster-restricted permutation *p* as the load-bearing significance statement.
- The necessary condition from Proposition 1 — specialized to our positive-base-rate setting as `|T_cal| − |T_br| > |Δ_1|` — is satisfied in 22/22 reversal rows (by construction, since this is what *defines* a reversal under the decomposition).

![Calibration vs base-rate magnitude ratio in reversal vs non-reversal rows](figures/fig_magnitude_ratio.png)

**Inference.** The empirical content of the decomposition is captured by the magnitude distribution, not by sign tallies. Reversal rows exhibit a calibration-magnitude advantage that is (a) universal in our data (22/22 with |cal| > |base-rate|) and (b) statistically much larger than in non-reversal rows. Because the magnitude inequality is *not* algebraically pinned by reversal status (a hypothetical reversal could in principle have |base-rate| > |cal| under negative base rates or different shift directions), this is a meaningful data finding rather than a re-statement of the decomposition.

**Hypothesis (biological).** Methods differ in their within-candidate-space discrimination of biologically plausible vs implausible edges; the magnitude evidence is consistent with discrimination shifts being the dominant *driver* of observed reversals in this slice, though sign causality is structural, not empirical.

**Scientific implication.** Benchmark interpretation should report the calibration-vs-base-rate magnitude ratio when discussing rank stability, not the sign-attribution counts that the decomposition algebraically pins.

### 5.3 Tissue shifts: deduplicated rate is much larger than the candidate-conditional rate suggests
**Evidence.** Treating each `(tissue_from, tissue_to, method_a, method_b)` triple as one observation (dedup), 17 of 45 method-pair × tissue-pair triples reverse under at least one candidate-set choice (rate 0.38, cluster CI [0.18, 0.58]). Only 1/45 reverse under *all* candidate-set choices (rate 0.02). The candidate-conditional rate of 0.19 reported in the previous version of this analysis lies between these two and inflates the underlying signal by counting each pair three times.

![Tissue-shift rate by candidate set (kept from prior pipeline)](figures/fig3_tissue_shift_reversal_heatmap.png)

**Inference.** Cross-tissue rank transportability is real but the magnitude depends sensitively on the reporting convention. The honest single-number summary is dedup-any = 0.38.

**Hypothesis (biological).** Candidate constraints amplify tissue-specific regulon mismatch by reducing background edge averaging; the 0.38 vs 0.02 gap shows most tissue-shift reversals are candidate-conditional, not universal.

**Scientific implication.** Cross-tissue mechanistic claims should report rank transportability under each candidate-set definition separately.

### 5.4 Reference shifts produce robust reversals across margin thresholds
**Evidence.** Reference-network shifts in the immune slice produce 34/106 reversals (rate 0.32, cluster CI [0.20, 0.43]). Per-cell BH-FDR: `dorothea_trrust_union_immune → hpn_dream` at rate 0.19 is significantly *below* 0.5 (`q = 7e−4`); the previously-highlighted `beeline_gsd → dorothea_trrust_union_immune` cell at 0.43 does **not** survive BH-FDR (`q = 0.40`) — its rate is not statistically distinguishable from 0.5 at this sample size. Importantly, the margin-magnitude sweep (§5.6) shows reference-shift reversals **persist** across all margin thresholds (rate stays in [0.22, 0.38] up to the 75th percentile of margins), while candidate-shift reversals vanish at high margins (0/34 at the 75th-percentile candidate threshold).

![Reference-shift reversal rate (kept from prior pipeline)](figures/fig4_reference_shift_reversal_bar.png)

**Inference.** Reference choice is a robust instability source: unlike candidate-shift instability, it is not concentrated in near-tie pairs.

**Scientific implication.** Single-reference "best method" claims are likely overconfident; multi-reference sensitivity should be standard.

### 5.5 Mapping-policy shifts changed coverage without reversing pairwise order
**Evidence.** Mapping-policy shifts show 0/165 pairwise reversals (Wilson upper 95% bound 2.3%, cluster bootstrap [0.00, 0.00]). Legacy-symbols → full-genes/crosswalk increases mean coverage by +0.86 (absolute) with near-zero mean F1 shift (−4e−5).

**Inference.** In this subset, mapping behaves as an approximately order-preserving transform despite large overlap gain.

**Hypothesis (biological).** The probe-prior subset may be less sensitive to alias ambiguity than broader evaluations with heterogeneous identifier conventions; this should not be generalized.

**Scientific implication.** Coverage changes still require reporting because they alter interpretability and comparability even when ordering is unchanged.

### 5.6 Margin magnitude controls reversal risk on the candidate axis but not the reference axis
**Evidence.** Figure `fig_margin_sweep` reports reversal rate as a function of the margin-magnitude floor `τ = min(|Δ_1|, |Δ_2|)`. For candidate shifts, rate drops from 0.16 (τ = 0, all 135 pairs) to 0.00 (τ = 75th-percentile margin, 34 pairs kept). For reference shifts, rate remains in [0.22, 0.38] up to the 75th-percentile margin. For tissue shifts, rate decays gradually from 0.19 to 0.09. For mapping shifts the rate is 0 throughout.

![Reversal rate vs margin threshold](figures/fig_margin_sweep.png)

**Inference.** Candidate-shift "instability" is overwhelmingly a phenomenon of near-tie method pairs; on pairs with margins above the median, candidate-shift reversal is exceedingly rare. Reference-shift instability is genuinely method-pair-structural and not explained by near-ties.

**Scientific implication.** Reporting reversal rates without a margin-magnitude regime is ambiguous; future stability reports should always include a margin sweep.

### 5.7 Reversal behavior is structured across three null models, but instability screening fails to beat baseline prevalence
**Evidence.** Three null models for the candidate-shift reversal rate are compared in figure `fig_nulls`:

| Null | Null mean (95% interval) | Observed |
|---|---|---:|
| Joint method-relabel in c_2 | 0.50 [0.37, 0.61] | 0.16 |
| Score-noise bootstrap (σ = 10% of median margin) | 0.29 [0.21, 0.37] | 0.16 |
| Rank permutation within (tissue × candidate) cell | 0.50 [0.37, 0.61] | 0.16 |

The score-noise null is a perturbation analysis rather than a true null: it adds Gaussian noise of magnitude `σ = 10% × (within-cell median pairwise margin)` to each observed score and recomputes the reversal rate. The reported value 0.29 is the *noise-inflated* rate; the observed 0.16 is the noise-free rate. Adding noise increases reversals (more pairs are perturbed across the sign-flip boundary), so 0.29 > 0.16 is mechanically expected; the substantive reading is that observed reversals are *robust* to ~10%-of-margin-scale score noise, i.e., the underlying ordering structure is tighter than that noise scale and the 16% reversal rate is not a noise artifact. The two relabel nulls confirm the rank signal is far from random.

![Observed rate vs three nulls](figures/fig_nulls.png)

For the instability-screening problem, the leave-one-tissue-out quantile heuristic peaks at precision 0.24, recall 0.64, F1 0.35 at threshold 0.25, and produces a PR-AUC of 0.18 — only marginally above the baseline reversal prevalence 0.16. A learned logistic-regression classifier on `(|Δ_1|, b_1, b_2 − b_1, |g_1|)` features yields PR-AUC 0.13, *below* baseline. This is an honest negative result: neither classifier provides actionable lift on this dataset slice.

![Instability screening curves](figures/fig_instability_screening.png)

**Inference.** Rank signal is real (well-separated from three independent nulls) but per-pair instability is not predictable from these features on this dataset; either richer features or replicate-level training signal would be needed.

### 5.8 Reversal risk depends materially on metric choice (AUROC is most stable)
**Evidence.** Repeating the tissue-shift analysis on Table A1 (no candidate stratification) with three metrics:

| Metric | n | k | Rate | Wilson 95% CI |
|---|---:|---:|---:|---|
| AUPR | 45 | 4 | 0.09 | [0.04, 0.21] |
| F1 | 45 | 4 | 0.09 | [0.04, 0.21] |
| AUROC | 40 | 0 | 0.00 | [0.00, 0.09] |

![Metric-robustness summary](figures/fig_metric_robustness.png)

**Inference.** AUROC produces no observed tissue-shift reversals in this dataset, while AUPR and F1 both produce 4 reversals (the same four pairs in each case). The choice of metric matters for the rank-stability conclusion.

**Hypothesis.** Because AUROC is invariant to monotone transformations of the score and AUPR is not, AUROC's stability reflects the underlying score-rank similarity across tissues; AUPR's instability reflects sensitivity to where the operating-rank-density lies relative to the positive base rate.

**Scientific implication.** Stability claims should be reported per metric; "method X is better on tissue Y" is metric-conditional and rank-conditional.

## 6. Biological Interpretation
The core biological lesson is that benchmarking protocol defines the biological question being asked. Candidate-space restrictions emphasize particular regulator-target submanifolds, reference changes reweight biological evidence classes, and tissue context changes the active regulatory program. Observed method rank mixes algorithmic quality with biological framing choices. We constrain our claims to protocol-conditional benchmarking behavior, not causal biological discovery; the hypothesis blocks in §5 are *conjectures* for follow-up, not tested findings of this paper.

## 7. Discussion
Three patterns persist across our analyses. (i) Among reversal rows, the calibration-term magnitude exceeds the base-rate-term magnitude in 22/22 cases (vs 56/113 in non-reversal rows; cluster-permutation MWU *p* < 2 × 10⁻⁴, rank-biserial *r* = 0.67), establishing on a magnitude basis (the only basis on which the `b · g` decomposition carries empirical content under positive base rates) that discrimination dominance accompanies the reversal phenomenon in this slice (§5.2). (ii) Candidate-shift instability is concentrated in near-tie pairs (§5.6); reference-shift instability is not. (iii) Metric choice materially affects the rank-stability conclusion: AUROC is fully stable across tissues in this slice while AUPR is not (§5.8).

We are deliberate about what is *not* a finding here. The "discrimination term opposes initial margin in 100% of reversals" sign-tally and the analogous "0/22 base-rate-only counterfactual" are algebraic consequences of the decomposition `Δ = b · g` under positive base rates and a fixed shift direction; they re-state the reversal definition rather than evidence about the mechanism (§3.2). Reporting them as findings, as earlier versions of this analysis did, conflates structure with data.

Our instability-screening results are honestly negative: neither a quantile heuristic nor a learned classifier outperforms baseline prevalence. We interpret this not as failure of the framework but as a sign that protocol-conditional reversal is, at this data size, a population-level phenomenon rather than a per-pair predictable one. Stronger screening would likely require replicate-level uncertainty estimates and a larger method × tissue × reference grid.

## 8. Limitations
1. Inputs are summary medians; raw per-replicate AUPR is not exposed by the source artifacts, so within-method uncertainty is not propagated into our CIs. Cluster bootstrap addresses dependence across pairs but not within-cell measurement noise.
2. Reference-shift analysis is centered on the immune baseline slice (one tissue, three references); the reference-shift conclusions should not be extrapolated to other tissues without re-running.
3. Mapping-policy results are limited to the probe-prior subset; the 0/165 finding does not imply mapping is irrelevant in general, only in this slice.
4. The minimum-shift bound (Prop. 1) is a necessary condition; we do not prove sufficiency.
5. Biological hypotheses in §5 are conjectures, not tested findings.

## 9. Data and Code Availability
All code, source data, regenerated tables, and figures used in this paper are released at the (author-anonymized) repository:

**https://anonymous.4open.science/r/grn-ranking-reversal-theory-DB8E/**

Repository layout: `data/` (Table A1/A2/A3 summary medians + the immune GRN baseline and four mapping-policy CSVs), `scripts/analysis.py` (self-contained pipeline), `results/tables/` and `results/figures/` (regenerated outputs), `paper/` (manuscript source and compiled PDF). Reproducing the analysis end-to-end requires Python 3.9+ and the packages listed in `requirements.txt`:

```
pip install -r requirements.txt
python scripts/analysis.py
```

All random seeds are fixed (project seed 42; permutation seed 42; bootstrap seed 42). Numerical decomposition closure is verified at runtime to within 1e-9. The repository's git history additionally preserves the pre-revision pipeline whose sign-attribution claims were retracted, for transparency.

## 10. Conclusion
Ranking reversal is a first-order reliability issue for GRN benchmarking. Cluster-bootstrap CIs that respect pair-level dependence, margin-magnitude sweeps, per-axis BH-FDR-adjusted cell rates, and a magnitude-based reading of the `b · g` decomposition (rather than the structurally pinned sign tallies) together replace the standard "rate ± Wilson interval" report with a more honest characterization of what is robust and what is not. The practical recommendation is direct: treat method rank as biologically interpretable evidence only after cross-axis stability is demonstrated, the reversal-prone margin regime is characterized, and the choice of metric is reported.

## References
[1] Cui H, Wang C, Maan H, et al. scGPT: toward building a foundation model for single-cell multi-omics using generative AI. Nature Methods. 2024.

[2] Theodoris CV, et al. Transfer learning enables predictions in network biology from large-scale single-cell atlases. Nature. 2023.

[3] Pratapa A, Jalihal AP, Law JN, Bharadwaj A, Murali TM. Benchmarking algorithms for gene regulatory network inference from single-cell transcriptomic data. Nature Methods. 2020.

[4] Aibar S, Gonzalez-Blas CB, Moerman T, et al. SCENIC: single-cell regulatory network inference and clustering. Nature Methods. 2017.

[5] Huynh-Thu VA, Irrthum A, Wehenkel L, Geurts P. Inferring regulatory networks from expression data using tree-based methods. PLoS ONE. 2010.

[6] Anonymous authors. Evaluation Bias and Symbol Mapping in GRN Benchmarks. Workshop manuscript. 2025. *(Author identity withheld for double-blind review.)*

[7] Tabula Sapiens Consortium. The Tabula Sapiens: a multiple-organ single-cell transcriptomic atlas of humans. Science. 2022.

[8] Holland CH, et al. DoRothEA as a resource of transcription factor activities from gene expression data. Nature Communications. 2020.

[9] Han H, et al. TRRUST v2: an expanded reference database of human and mouse transcriptional regulatory interactions. Nucleic Acids Research. 2018.

[10] Türei D, et al. Integrated intra- and intercellular signaling knowledge with OmniPath. Nature Methods. 2021.

[11] Post M. A Call for Clarity in Reporting BLEU Scores. WMT. 2018.

[12] Dehghani M, Tay Y, Gritsenko AA, et al. The Benchmark Lottery. arXiv:2107.07002. 2021.

[13] Demsar J. Statistical Comparisons of Classifiers over Multiple Data Sets. JMLR. 2006.

[14] Critchlow DE. Metric Methods for Analyzing Partially Ranked Data. Lecture Notes in Statistics 34. Springer. 1985.

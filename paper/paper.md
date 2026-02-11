# Ranking Reversal Theory Under Candidate-Set Shift in GRN Benchmarking

## Abstract
Gene regulatory network (GRN) benchmarks are often treated as evidence for biological mechanism quality, yet method ranking can change under plausible evaluation-protocol choices. We study this ranking-instability problem in a theory-first framework and evaluate it with low-compute analyses over existing single-cell GRN benchmark outputs. We derive exact reversal conditions and decomposition identities that separate base-rate effects from calibration/discrimination effects. Empirically, candidate-set shifts yield 22/135 pairwise reversals (16.3%, 95% CI 11.0-23.4), tissue shifts yield 26/135 (19.3%), and reference shifts in immune baseline evaluations yield 34/106 (32.1%). Mapping-policy shifts in a probe-prior subset induce large coverage changes but no pairwise order reversals (0/165; upper 95% bound 2.28%). A 5,000-permutation null shows observed candidate-shift reversals are far below random-order behavior (0.163 vs null mean 0.500), indicating partially stable but non-invariant structure. The central implication is scientific: benchmark rank should not be interpreted as biological evidence until stability across candidate, tissue, and reference axes is quantified.

## 1. Introduction
Mechanistic interpretability for biological foundation models increasingly relies on benchmark ranking to support claims about biological plausibility and downstream utility [1,2]. In GRN recovery, however, ranking can change substantially with protocol choices such as candidate-space restriction, symbol mapping policy, and reference-network selection [3-6].

This is not only a reporting issue. If ranking is unstable, biological decisions can flip: which regulators are prioritized for validation, which mechanistic narrative is emphasized, and which model is treated as scientifically credible. The field therefore needs explicit ranking-stability theory and diagnostics, not only larger metric tables.

We make three contributions. First, we formalize exact reversal conditions for pairwise method ordering under protocol shift. Second, we decompose margin shifts into base-rate and calibration/discrimination terms and show how this decomposition constrains interpretation. Third, we quantify reversal behavior across candidate, tissue, reference, and mapping-policy shifts and evaluate an instability-region diagnostic under leave-one-tissue-out testing.

## 2. Problem Setup
Let `M_m(S, pi, R)` be a metric for method `m` under candidate set `S`, mapping policy `pi`, and reference `R`. For methods `A` and `B`, define margin:

`Delta = M_A - M_B`.

For two settings (1 and 2), define `Delta_1`, `Delta_2`, and `dDelta = Delta_2 - Delta_1`. A ranking reversal occurs when:

`Delta_1 * Delta_2 < 0`.

## 3. Theory
### 3.1 Exact reversal criterion
**Proposition 1.** A reversal occurs iff:

`sign(Delta_1) * dDelta < -|Delta_1|`.

So a shift must oppose the initial ordering and exceed the initial margin.

### 3.2 Candidate-set decomposition
For fixed mapping policy and reference, write:

`Delta(S) = b(S) * g(S)`

where `b(S)` is candidate-set base rate and `g(S)` is base-rate-normalized discrimination gap.

For `S1 -> S2`:

`Delta_2 - Delta_1 = (b2 - b1) * g1 + b2 * (g2 - g1)`.

- Base-rate term: `(b2 - b1) * g1`
- Calibration/discrimination term: `b2 * (g2 - g1)`

**Corollary.** If `g2 = g1` and `b2 > 0`, base-rate scaling alone cannot reverse ordering.

### 3.3 Instability-region criterion
**Proposition 2.** If `|dDelta| <= B` for a shift family, then all pairs with `|Delta_1| <= B` lie in an instability region where reversal is possible.

This criterion is intended as a high-recall warning tool, not a high-precision classifier.

### 3.4 Mapping decomposition
For mapping shifts, write `M = c * q` with coverage `c` and coverage-adjusted quality `q`:

`M_2 - M_1 = (c2 - c1) * q1 + c2 * (q2 - q1)`.

This separates overlap expansion effects from quality changes.

## 4. Methods
### 4.1 Data sources and evaluation objects
We use previously generated benchmark summaries and score-evaluation outputs for:
- tissue-stratified method-by-candidate summaries,
- immune baseline method scores across multiple references,
- probe-prior evaluations across mapping-policy variants.

These artifacts represent kidney, lung, and immune contexts in a common GRN benchmarking stack built around curated references and baseline inference methods [3-6,9,10].

### 4.2 Analyses
We compute:
1. pairwise reversal rates with Wilson confidence intervals,
2. decomposition terms for candidate and reference shifts,
3. leave-one-tissue-out instability classification and quantile sweeps,
4. permutation null for candidate-shift reversal rate (5,000 permutations).

### 4.3 Statistical framing
All rate estimates include uncertainty intervals; decomposition identities are algebraic and checked numerically for closure error. We treat instability detection as a screening problem and report precision, recall, and specificity.

### 4.4 Interpretability validity checks
To avoid purely descriptive “storytelling,” we treat rank changes as valid evidence only when tied to controlled shift axes and explicit null models. Claims are constrained to protocol-conditional benchmarking behavior, not causal biological discovery [11-16].

## 5. Results
### 5.1 Candidate-set shifts produce nontrivial and tissue-heterogeneous reversal risk
**Evidence.** Candidate-set shifts produce 22/135 pairwise reversals (16.30%, 95% CI 11.02-23.44). Heterogeneity is large: immune all_pairs -> tf_sources_targets reaches 40% (6/15), while kidney tf_sources -> tf_sources_targets is 0/15.

![Candidate shift reversal heatmap](figures/fig1_candidate_shift_reversal_heatmap.png)

**Inference.** Rank is not a method-intrinsic invariant; it is protocol-conditional.

**Hypothesis (biological).** The immune context has stronger context-specific regulator usage and weaker reference completeness, increasing sensitivity of method ordering to candidate composition.

**Scientific implication.** A single leaderboard rank should not be used to justify biological claims without candidate-shift stability reporting.

### 5.2 Candidate-shift reversals are calibration/discrimination dominated
**Evidence.** In reversal rows, calibration term opposes initial margin in 100% of cases; base-rate term opposes in 0%. Mean `|calibration|/|base-rate|` for reversal rows is 1.54.

![Candidate decomposition scatter](figures/fig2_decomposition_scatter.png)

**Inference.** Base-rate inflation alone does not explain observed rank flips in this analysis.

**Hypothesis (biological).** Methods differ in within-candidate discrimination of biologically plausible vs implausible edges; these discrimination shifts, not only class-imbalance scaling, drive reversals.

**Scientific implication.** Benchmark interpretation should prioritize normalized discrimination stability in addition to absolute AUPR gains.

### 5.3 Tissue shifts become more unstable under constrained candidate spaces
**Evidence.** Tissue shifts yield 26/135 reversals (19.26%, 95% CI 13.50-26.72). By candidate set: all_pairs 4.44%, tf_sources 22.22%, tf_sources_targets 31.11%.

![Tissue shift reversal heatmap](figures/fig3_tissue_shift_reversal_heatmap.png)

**Inference.** Cross-tissue rank transportability degrades as candidate spaces become biologically curated.

**Hypothesis (biological).** Candidate constraints amplify tissue-specific regulon mismatch by reducing background edge averaging.

**Scientific implication.** Cross-tissue mechanistic claims should report rank transportability explicitly, especially for constrained candidate evaluations.

### 5.4 Reference shifts show the highest reversal rates in this study
**Evidence.** Reference shifts in immune baseline evaluations yield 34/106 reversals (32.08%, 95% CI 23.95-41.45). The beeline_gsd -> dorothea_trrust_union_immune shift shows 42.86% reversal.

![Reference shift reversal bar](figures/fig4_reference_shift_reversal_bar.png)

**Inference.** Reference choice is a dominant instability source in this slice.

**Hypothesis (biological).** Different references encode different biological evidence regimes (curated TF-target priors vs other signal classes), so method ordering can legitimately differ across them.

**Scientific implication.** Single-reference “best method” claims are likely overconfident; multi-reference sensitivity should be standard.

### 5.5 Mapping-policy shifts changed coverage without reversing order in this subset
**Evidence.** Mapping-policy shifts show 0/165 pairwise reversals (upper 95% bound 2.28%). Legacy-symbols -> full-genes/crosswalk increases mean coverage by +0.862 with very small mean F1 shift (-4.32e-05).

**Inference.** In this subset, mapping behaves approximately as an order-preserving transform despite large overlap gain.

**Hypothesis (biological).** The subset may be less sensitive to alias ambiguity than broader evaluations with heterogeneous identifier conventions.

**Scientific implication.** Coverage changes still require reporting because they alter interpretability and comparability even when ordering is unchanged.

### 5.6 Reversal behavior is structured, not random
**Evidence.** Under 5,000 permutations, observed candidate-shift reversal rate is 0.16296 vs null mean 0.50017 (null 2.5-97.5%: 0.385-0.615).

![Observed vs null](figures/fig5_observed_vs_null.png)

**Inference.** Rankings retain substantial shared structure across shifts, but with meaningful instability pockets.

**Hypothesis (biological).** Stable method pairs may share bias toward similar edge topologies or expression-driven relations; unstable pairs likely emphasize complementary biological signal families.

**Scientific implication.** Stability-aware method selection is feasible and more defensible than single-condition ranking.

### 5.7 Instability screening offers useful triage at moderate thresholds
**Evidence.** Leave-one-tissue-out instability quantile sweep peaks near quantile 0.25 with precision 0.237, recall 0.636, specificity 0.602, and F1 0.346.

![Instability quantile sweep](figures/fig6_instability_quantile_sweep.png)

**Inference.** Instability regions provide practical high-recall screening but are insufficient as stand-alone decision rules.

**Hypothesis (biological).** Adding replicate-level uncertainty and per-regulator stratification would improve specificity by separating generic protocol sensitivity from biologically coherent context effects.

**Scientific implication.** Instability score should be used as a gating signal before expensive biological validation.

## 6. Biological Interpretation
The core biological lesson is that benchmarking protocol defines the biological question being asked. Candidate-space restrictions emphasize particular regulator-target submanifolds, reference changes reweight biological evidence classes, and tissue context changes the active regulatory program. Therefore, observed method rank mixes algorithmic quality with biological framing choices. In this setting, biological interpretation should be conditional and explicitly tied to stability diagnostics, rather than treated as universally transferable.

## 7. Discussion
Our findings argue for a stability-first benchmarking culture in mechanistic interpretability for biology. The decomposition results indicate that rank flips are often linked to discrimination-shape changes, not only prevalence shifts. This matters because discrimination behavior is closer to what scientists care about when prioritizing mechanistic hypotheses.

## 8. Limitations
The present study relies on existing summary outputs rather than full replicate-level raw prediction matrices; reference-shift analysis is currently centered on immune baseline outputs; and biological hypotheses here are interpretation-level rather than causal proof. Future work should integrate perturbation-grounded validations and batch/donor-aware stability checks.

## 9. Data and Code Availability
All code, analysis scripts, figure-generation pipelines, and configuration files for this study are available at:

**https://github.com/Biodyn-AI/grn-ranking-reversal-theory**

The repository README provides instructions for reproducing all main figures and tables, along with dataset access guidance and environment setup details.

## 10. Conclusion
Ranking reversal is a first-order reliability issue for GRN benchmarking and mechanistic interpretation claims. A theory-first framework with explicit decomposition and instability diagnostics enables more honest and scientifically actionable interpretation. The practical recommendation is direct: treat method rank as biologically interpretable evidence only after cross-axis stability is demonstrated.

## References
[1] Cui H, Wang C, Maan H, et al. scGPT: toward building a foundation model for single-cell multi-omics using generative AI. Nature Methods. 2024.

[2] Theodoris CV, et al. Transfer learning enables predictions in network biology from large-scale single-cell atlases. Nature. 2023.

[3] Pratapa A, Jalihal AP, Law JN, Bharadwaj A, Murali TM. Benchmarking algorithms for gene regulatory network inference from single-cell transcriptomic data. Nature Methods. 2020.

[4] Aibar S, Gonzalez-Blas CB, Moerman T, et al. SCENIC: single-cell regulatory network inference and clustering. Nature Methods. 2017.

[5] Huynh-Thu VA, Irrthum A, Wehenkel L, Geurts P. Inferring regulatory networks from expression data using tree-based methods. PLoS ONE. 2010.

[6] Kendiukhov I, et al. Evaluation Bias and Symbol Mapping in GRN Benchmarks. Workshop manuscript. 2025.

[7] Tabula Sapiens Consortium. The Tabula Sapiens: a multiple-organ single-cell transcriptomic atlas of humans. Science. 2022.

[8] Holland CH, et al. DoRothEA as a resource of transcription factor activities from gene expression data. Nature Communications. 2020.

[9] Han H, et al. TRRUST v2: an expanded reference database of human and mouse transcriptional regulatory interactions. Nucleic Acids Research. 2018.

[10] Türei D, et al. Integrated intra- and intercellular signaling knowledge with OmniPath. Nature Methods. 2021.

[11] Olah C, Cammarata N, et al. Zoom In: An Introduction to Circuits. Distill. 2020.

[12] Geva M, Schuster R, Berant J, Levy O. Transformer Feed-Forward Layers Are Key-Value Memories. EMNLP. 2021.

[13] Meng K, Bau D, Andonian A, Belinkov Y. Locating and Editing Factual Associations in GPT. NeurIPS. 2022.

[14] Geiger A, Wu Z, Potts C, Icard T. Finding Causal Abstractions of Neural Networks. NeurIPS. 2021.

[15] Hewitt J, Liang P. Designing and Interpreting Probes with Control Tasks. EMNLP-IJCNLP. 2019.

[16] Belinkov Y. Probing Classifiers: Promises, Shortcomings, and Advances. Computational Linguistics. 2022.

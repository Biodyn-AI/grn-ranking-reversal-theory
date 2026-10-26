# Evaluation Bias and Symbol Mapping in GRN Benchmarks

## Status (2025-01-11)
- Layout fixes applied in LaTeX: centered heatmap figures via local overrides, tightened caption/float spacing, and split wide appendix tables; PDF rebuilt at `reports/evaluation_bias_protocol/workshop/eval_bias_paper_draft.pdf`.
- Main body remains 8 pages (references start on page 9).

## Next Steps
- Review the PDF in a viewer to confirm centering and caption spacing across pages.
- If any figure remains off-center, regenerate the source plots without large left margins (edge/gene coverage heatmaps, method-candidate heatmap).
- If the appendix is too long, trim or move less critical tables/figures to supplemental material.

## Conversation Summary
- Addressed layout issues in the ICLR-template PDF (non-centered figures and caption overlap).
- Identified mis-centered PNGs and created centered copies under `reports/evaluation_bias_protocol/workshop/figures/`, then prioritized them via `\graphicspath`.
- Adjusted LaTeX float spacing and table formatting, splitting wide appendix tables for readability; recompiled the PDF.

## Abstract
Evaluation of gene regulatory network (GRN) recovery is highly sensitive to protocol choices that are often underreported. We quantify the impact of gene-symbol normalization, gold-standard selection, candidate-set definition, and mapping policy on GRN metrics using Tabula Sapiens kidney, lung, and immune gene sets and three gold standards (TRRUST, DoRothEA curated, DoRothEA ChIP-seq). Symbol mapping alone shifts average edge coverage from 0.000 to 0.997 (range 0.994 to 0.999), while alternative mapping policies shift mean edge coverage by 4.1e-04. For scGPT attention, candidate-set restrictions change median AUPR from 2.5e-05 (all pairs) to 7.38e-03 (TF-source-target), a 295x increase; across all methods the median increase is 44x. GRNBoost2/GENIE3 baselines outperform the scGPT attention-derived network in TF-source-target evaluation (median AUPR 0.0255 and 0.0256 vs 0.00738), with Pearson/Spearman at 0.0204 and 0.0195. Bootstrap confidence intervals and structured noise perturbations show increasing variability as candidate sets shrink. We propose a reproducible evaluation protocol with explicit mapping policies, coverage reporting, and candidate-set metadata.

## Introduction
GRN recovery evaluations are commonly reported as a single number (e.g., AUPR) against a single gold standard. In practice, evaluations are dominated by gene symbol normalization rules, gold-standard compatibility with the gene universe, and the choice of candidate edges. These choices are frequently implicit, making cross-study comparisons unreliable. We quantify these effects and provide a reproducible evaluation suite for mechanistic interpretability pipelines.

## Related Work
Benchmarking GRN inference is an active area with multiple gold standards (TRRUST, DoRothEA, ChIP-seq compendia) and diverse model families (correlation, tree-based, probabilistic, and deep learning). Prior evaluations emphasize AUPR/AUROC but often under-report symbol normalization and candidate-set constraints, which can dominate observed performance. The scGPT family of models introduces transformer-based attention mechanisms for single-cell expression, motivating mechanistic interpretability studies that link attention weights to regulatory edges. Our work complements these efforts by isolating evaluation protocol decisions and standardizing reporting, rather than proposing a new inference algorithm.

## Methods
### Data
- scRNA-seq gene sets: Tabula Sapiens kidney, lung, and immune (H5AD files in `single_cell_mechinterp/data/raw/`). Each tissue provides ~60,606 raw genes; after mapping and filtering, ~60,339 unique mapped symbols remain.
- Predicted edges: scGPT attention-derived networks for kidney, lung, and immune, generated with `max_genes=1000` and `max_cells=2000` for all tissues and saved under `single_cell_mechinterp/outputs/eval_bias/scgpt/` (240k to 243k edges per tissue before candidate filtering).
- Gold standards: TRRUST human (`single_cell_mechinterp/external/networks/trrust_human.tsv`), DoRothEA curated (`single_cell_mechinterp/external/networks/dorothea_human.tsv`), and DoRothEA ChIP-seq (`single_cell_mechinterp/external/networks/dorothea_chipseq_human.tsv`). Gold-standard edge counts range from 8.4k (TRRUST) to 276k (DoRothEA curated).

### Preprocessing and Gene Set Construction
We preprocess each Tabula Sapiens tissue with the standard pipeline (`prepare-data`): minimum 200 genes per cell and 3 cells per gene, total-count normalization to 10,000, log1p transformation, and HVG selection with `hvg=5000`. The processed H5ADs live under `single_cell_mechinterp/outputs/atlas/{kidney,lung,immune}/` and serve both as the gene universe for evaluation and the input for baseline generation.

### scGPT Attention Extraction and Network Inference
We extract attention scores with `reduce_layers=true` and `reduce_heads=true`, so the attention tensor collapses into a 2D gene-by-gene matrix. To keep runs tractable on CPU we limit the dataset to `max_genes=1000` and `max_cells=2000` per tissue, use `batch_size=2`, and store scores as float16 with shared int32 counts. The final attention scores are averaged across counts and converted to a directed edge list via `NetworkConfig(top_k=50, threshold_percentile=95, remove_self=true)` without TF source/target masking. The resulting edges are saved under `single_cell_mechinterp/outputs/eval_bias/scgpt/{kidney,lung,immune}/inferred_edges.tsv`.

### Baseline Prediction Methods
We generate five additional prediction sets per tissue from processed Tabula Sapiens atlas datasets (`single_cell_mechinterp/outputs/atlas/*/tabula_sapiens_*_processed.h5ad`). For the tree-based baselines we subsample to 2,000 cells, restrict to 1,000 targets and 500 regulators (TFs drawn from DoRothEA), and use arboreto with reduced estimator counts (200 trees) for tractability, taking the top-50 regulators per target:
- Pearson correlation (absolute value), top-50 targets per source.
- Spearman correlation (absolute value), top-50 targets per source.
- GENIE3 (arboreto, random forest feature importances), top-50 regulators per target.
- GRNBoost2 (arboreto, gradient boosting feature importances), top-50 regulators per target.
- Random baseline (uniform random scores), top-50 targets per source.

Arboreto settings are `n_estimators=200` with `max_features=sqrt` for GENIE3 (RF) and `n_estimators=200`, `learning_rate=0.05`, `max_depth=3`, `subsample=0.8`, `max_features=0.3` for GRNBoost2 (GBM). All baseline settings are recorded in `single_cell_mechinterp/configs/eval_bias_baselines.yaml`.

Baseline edges are stored under `single_cell_mechinterp/outputs/eval_bias/baselines/` with per-tissue subdirectories.

### Symbol Mapping
We normalize symbols by uppercasing and stripping punctuation/whitespace, then apply HGNC alias and previous-symbol mappings (`single_cell_mechinterp/external/hgnc_complete_set.txt`). We additionally map Ensembl IDs to symbols via `single_cell_mechinterp/external/gene_info.csv`. Ambiguous aliases are resolved deterministically (lexicographic), and unmapped symbols are retained with their normalized form.

We evaluate three mapping policies: lexicographic resolution, drop ambiguous, and drop unmapped. Mapping outcomes are recorded in `single_cell_mechinterp/outputs/eval_bias/symbol_mapping_report.tsv` and summarized in `single_cell_mechinterp/outputs/eval_bias/tables/mapping_summary_by_type.csv`.

Each mapped token is tagged with a status (unchanged, alias_resolved, ambiguous_resolved, ambiguous_dropped, unmapped), enabling audit of how symbol normalization changes each context (predicted edges, gold standards, gene sets). We also emit a narrative log in `single_cell_mechinterp/outputs/eval_bias/symbol_mapping_log.md` that documents the mapping policies and skipped inputs.

### Candidate Sets
We evaluate three candidate-set protocols:
1. All expressed pairs (all genes x all genes).
2. TF-source only (sources constrained to gold-standard TFs).
3. TF-source + TF-target (sources and targets both constrained to gold-standard TFs/targets).

Median candidate sizes are 3.59e9 (all pairs), 4.35e7 (TF-source), and 1.94e6 (TF-source-target), with corresponding median base rates 3.20e-05, 2.01e-03, and 9.98e-03.

Candidate sets are constructed per gold standard using the mapped gene universe; TF-source sets restrict sources to gold-standard TFs, and TF-source-target sets additionally restrict targets to gold-standard targets. Because all three tissues share the same mapped gene-set size in this evaluation, candidate sizes and base rates are effectively tissue-invariant (Table 7).

### Metrics, Noise, and Bootstrap CIs
We compute precision, recall, F1, AUPR, AUROC, and precision/recall at K. Candidate-set size and base rate are reported per evaluation. Noise stability is evaluated via random edge deletion/addition at rates 0.05, 0.1, 0.2, and 0.3, as well as structured TF- and target-dropout perturbations. We compute bootstrap confidence intervals for AUPR/AUROC using 200 resamples of predicted edges.

Ranking metrics are computed from the sorted prediction scores with the candidate-set base rate appended to ensure full recall coverage when predictions do not span all true edges. Bootstrap CIs resample predicted edges with replacement while holding total positives/negatives fixed to the candidate set, yielding percentile intervals (2.5% and 97.5%). Noise perturbations remove a fraction of gold edges, then add an equal number of randomly sampled negatives; structured dropout removes all edges for a sampled subset of TF sources or targets, repeated with `structured_repeats=5`.

Metric definitions used throughout:
- Precision = TP / (TP + FP), Recall = TP / (TP + FN), F1 = 2 * (Precision * Recall) / (Precision + Recall).
- AUPR is the area under the precision-recall curve computed by trapezoidal integration over ranked predictions; AUROC analogously integrates TPR vs FPR.
- Base rate = (# true edges in candidate set) / (candidate set size).

### Evaluation Pipeline Details
Predicted edges are de-duplicated by keeping the highest score per source-target pair. For each gene set and gold standard, coverage is computed twice: once on normalized (uppercase/punctuation-stripped) symbols and once after full HGNC/Ensembl mapping. Candidate sets are built from the mapped gene universe and gold-standard TF/target lists; evaluation filters predicted edges to each candidate set and computes ranking metrics against the gold-standard edges restricted to the same gene universe.

### Random Seeds
All pipeline components use fixed seeds (project seed 42; bootstrap seed 42; noise seed 42) to ensure reproducibility across runs.

### Reproducibility
Primary commands:
- `python scripts/generate_eval_bias_baselines.py --config configs/eval_bias_baselines.yaml`
- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m src.cli --config configs/eval_bias_scgpt_kidney.yaml extract-attention --batch-size 2 --device cpu`
- `python -m src.cli --config configs/eval_bias_scgpt_kidney.yaml infer-network`
- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m src.cli --config configs/eval_bias_scgpt_lung.yaml extract-attention --batch-size 2 --device cpu`
- `python -m src.cli --config configs/eval_bias_scgpt_lung.yaml infer-network`
- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m src.cli --config configs/eval_bias_scgpt_immune.yaml extract-attention --batch-size 2 --device cpu`
- `python -m src.cli --config configs/eval_bias_scgpt_immune.yaml infer-network`
- `python scripts/run_eval_bias_protocol.py --config configs/eval_bias.yaml`
- `python scripts/generate_eval_bias_summary.py --input-dir outputs/eval_bias --output-dir outputs/eval_bias`

### Compute and Runtime Notes
All scGPT attention extractions were run on CPU (flash-attn not installed) with `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. In our run, attention extraction took ~17 to 25 minutes per tissue, baseline generation across three tissues took ~10 minutes, and the full evaluation protocol took ~35 to 40 minutes. These timings will vary with hardware and thread configuration.

## Results
### Symbol Mapping Drives Coverage
Without HGNC/Ensembl mapping, edge coverage is effectively zero for all gold standards. After mapping, average edge coverage is 0.997 (range 0.994 to 0.999), and average gene coverage is 0.997 (range 0.995 to 0.998). Mapping policy effects are small in aggregate (mean edge coverage range 4.1e-04 across policies), but are still large enough to matter when overlap is sparse (Tables 1 and 4; Figures 1, 2, and 3).

### Gold Standard Compatibility
Mapped edge coverage differs across gold standards: DoRothEA curated averages 0.9983, DoRothEA ChIP-seq averages 0.9989, and TRRUST averages 0.9949. These differences reflect divergent gene coverage and curation depth, and they propagate into candidate-set base rates and downstream metrics (Table 1; Figure 2).

### Candidate-Set Sensitivity
For scGPT attention, median AUPR increases from 2.5e-05 (all pairs) to 1.41e-03 (TF-source) and 7.38e-03 (TF-source-target), a 295x lift compared to all pairs. Aggregating across all methods, the median AUPR rises from 3.16e-04 to 1.40e-02 (44x). These gains coincide with large changes in candidate size and base rate (Figures 4, 5, and 6; `single_cell_mechinterp/outputs/eval_bias/tables/candidate_summary.csv`).

### Method Comparisons
Across all gold standards and mapping policies, GRNBoost2 and GENIE3 baselines achieve the highest median AUPR (0.0255 and 0.0225), followed by Spearman (0.00836), Pearson (0.00817), random (0.00223), and scGPT attention (0.00141). Under TF-source-target evaluation, GENIE3 and GRNBoost2 reach median AUPR 0.0256 and 0.0255, while Pearson and Spearman are 0.0204 and 0.0195, compared to 0.00738 for scGPT attention. This suggests that co-expression and tree-based baselines can outperform attention-derived edges under typical GRN evaluation protocols (Figures 7 and 8; Table 2).
Tissue-stratified summaries of method rankings and candidate-set effects are reported in Tables 6 to 8 and Figures 17 to 19.

### Tissue-Stratified Candidate-Set Effects
Candidate-set medians vary by tissue even with identical candidate sizes. Across all methods, all-pairs AUPR is lowest for immune (9.1e-05), moderate for lung (4.19e-04), and highest for kidney (6.24e-04). TF-source medians are 2.93e-03 (immune), 9.09e-03 (lung), and 8.59e-03 (kidney). TF-source-target medians rise to 1.02e-02 (immune), 1.95e-02 (lung), and 2.02e-02 (kidney). These tissue differences indicate that evaluation outcomes are not solely driven by candidate-set sizes; biological context and gold-standard overlap also matter (Table 7; Figure 18).

### Tissue-Stratified Method Performance
Across all candidate sets, kidney and lung support higher median AUPR than immune for every method. For example, median AUPR (all candidate sets) for GRNBoost2 is 0.0267 (kidney), 0.0255 (lung), and 0.00927 (immune); scGPT attention is 0.00110 (kidney), 0.00142 (lung), and 0.00124 (immune) (Table 6; Figure 17).

Under TF-source-target evaluation, method rankings differ by tissue. Kidney remains dominated by correlation baselines (Spearman 0.0373, Pearson 0.0339), followed by GRNBoost2 (0.0290) and GENIE3 (0.0285). Lung favors tree-based baselines (GENIE3 0.0256, GRNBoost2 0.0255), with Pearson/Spearman at 0.0204/0.0195. Immune compresses the range, with Spearman 0.0145, Pearson 0.0134, GRNBoost2 0.0105, GENIE3 0.00848, random 0.00889, and scGPT 0.00738 (Table 8; Figure 19).

### scGPT Tissue Profiles
scGPT attention remains sensitive to candidate-set definition across tissues. Median AUPR for scGPT under TF-source-target is 0.00499 (kidney), 0.00934 (lung), and 0.00738 (immune), compared to 1.8e-05 to 2.6e-05 for all-pairs. In the TF-source-target setting, scGPT remains below random for all three tissues (random median AUPR: 0.0117 kidney, 0.0123 lung, 0.00889 immune), highlighting the need for careful baseline comparisons.

### Mapping Policy Effects on AUPR
Mapping policy choices modestly shift aggregate performance. Median AUPR across all evaluations is 0.00679 for drop-ambiguous, 0.00663 for drop-unmapped, and 0.00643 for lexicographic mapping. While these shifts are small relative to candidate-set effects, they can be material in sparse-overlap regimes and should be reported alongside coverage statistics.

### Base Rate Effects
Base rate ranges from 2.02e-06 (all pairs, TRRUST) to 2.60e-02 (TF-source-target, DoRothEA curated). AUPR scales with base rate, producing near-linear trends in log space (Figure 6). This indicates that apparent performance gains can be explained by candidate-set restriction rather than model quality.

### Noise Stability
Random noise yields low AUPR variance for all pairs (mean std 5.3e-05) and TF-source sets (2.76e-04), with larger variance for TF-source-target (1.14e-03). Structured TF- and target-dropouts further increase variability (mean AUPR std 1.42e-03 and 6.49e-04, respectively), particularly for TF-source and TF-source-target candidate sets (Figure 11; `single_cell_mechinterp/outputs/eval_bias/noise_stability_summary.tsv`).

### Bootstrap Confidence Intervals
Median AUPR CI widths rise sharply as candidate sets shrink: 1.02e-04 (all pairs), 2.16e-03 (TF-source), and 7.86e-03 (TF-source-target). CI widths are also method-dependent, with correlation and tree-based baselines showing wider intervals than scGPT attention due to higher AUPR variability (Table 5; Figure 13).

## Discussion
Evaluation outcomes for GRN recovery are largely determined by protocol decisions. Symbol mapping alone can flip overlap from zero to near-complete, underscoring the need to version and report normalization policies. Candidate-set restrictions provide large metric gains that reflect class imbalance rather than underlying model improvements, and they increase sensitivity to label noise and bootstrap variability. Baseline co-expression methods outperform the attention-derived network under these protocols, highlighting the need to contextualize mechanistic claims against simple baselines.

Tissue-stratified results show that method rankings are not stable across contexts: kidney favors correlation baselines, lung favors tree-based baselines, and immune compresses the performance range with smaller separations between methods. This variability reinforces the need to report per-tissue or per-dataset metrics rather than single pooled numbers.

Mapping policy effects are modest in aggregate, but they can still shift AUPR by ~5% relative in low-overlap regimes. scGPT attention remains below random in TF-source-target evaluations across tissues, suggesting that attention-derived edges should be treated as hypothesis generators rather than reliable GRN predictors without additional validation.

### Recommended Protocol
- Version and cite symbol mapping sources (HGNC, Ensembl).
- Report mapping outcomes (alias-resolved, ambiguous-resolved, unmapped counts) and mapping policy.
- Report candidate-set sizes and base rates alongside metrics.
- Evaluate across at least two candidate sets (all pairs + TF-source-target).
- Include structured noise and bootstrap analyses to bound metric variability.
- Compare against simple baselines (Pearson/Spearman) and a tree-based baseline (GENIE3/GRNBoost2).

### Future Work
- Scale scGPT attention extraction to full gene sets and larger cell counts to test whether performance gaps persist.
- Add additional GRN inference methods (e.g., CLR/PIDC/SCENIC variants) to broaden the baseline panel.
- Incorporate perturbation-derived gold standards or condition-specific networks to test evaluation sensitivity to biological context.
- Explore alternative attention aggregation and thresholding schemes (e.g., per-head filters, percentile thresholds, TF-only attention).

## Limitations
- Gold standards are incomplete and biased toward well-studied TFs.
- scGPT attention edges are tissue-matched across kidney, lung, and immune with reduced budgets (`max_genes=1000`, `max_cells=2000`), which may dampen performance relative to full-scale runs.
- Tree-based baselines were run on a subset of targets and regulators (1,000 targets, 500 regulators) and 2,000 sampled cells for tractability.
- Arboreto GRNBoost2 here uses scikit-learn gradient boosting with reduced estimator counts, not the full-scale (or XGBoost-based) variant, which may understate baseline performance.
- We evaluate a single scGPT attention configuration; alternative attention aggregation schemes, thresholds, or candidate constraints may change the relative rankings.
- Noise simulations do not model tissue-specific regulatory context or temporal dynamics.

## Conclusion
Benchmark results for GRN recovery are highly sensitive to evaluation protocol design. Standardized symbol mapping, candidate-set reporting, and baseline comparisons are necessary to interpret and compare mechanistic interpretability claims. The provided evaluation suite enables reproducible and transparent benchmarking across datasets and gold standards.

## Data and Code Availability
All artifacts referenced in this draft are stored under `single_cell_mechinterp/outputs/eval_bias/`, with summary tables in `single_cell_mechinterp/outputs/eval_bias/tables/` and figures in `single_cell_mechinterp/outputs/eval_bias/figures/`. The evaluation pipeline code lives in `single_cell_mechinterp/src/eval/bias_protocol.py` and `single_cell_mechinterp/scripts/run_eval_bias_protocol.py`.
The LaTeX manuscript with all figures and tables is `reports/eval_bias_paper_draft.tex`.

### Artifact Inventory (Selected)
- Protocol comparison table: `single_cell_mechinterp/outputs/eval_bias/protocol_comparison.csv`
- Coverage tables: `single_cell_mechinterp/outputs/eval_bias/coverage_tables.tsv`
- Noise stability summary: `single_cell_mechinterp/outputs/eval_bias/noise_stability_summary.tsv`
- Symbol mapping report/log: `single_cell_mechinterp/outputs/eval_bias/symbol_mapping_report.tsv` and `single_cell_mechinterp/outputs/eval_bias/symbol_mapping_log.md`
- Baseline edge sets: `single_cell_mechinterp/outputs/eval_bias/baselines/{kidney,lung,immune}/*.tsv`
- scGPT edge sets: `single_cell_mechinterp/outputs/eval_bias/scgpt/{kidney,lung,immune}/inferred_edges.tsv`

`protocol_comparison.csv` includes per-evaluation metrics and metadata: precision/recall/F1, AUPR/AUROC, precision@K and recall@K (K = 50, 100, 200), predicted/true edge counts, candidate size, base rate, gene set, gold standard, candidate set, source/target counts, prediction method, mapping policy, and bootstrap CI bounds for AUPR/AUROC.

## Figures and Tables
- Table 1: Coverage summary (`single_cell_mechinterp/outputs/eval_bias/tables/coverage_summary.csv`)
- Table 2: Candidate-set summary (`single_cell_mechinterp/outputs/eval_bias/tables/candidate_summary.csv`)
- Table 3: Noise summary (`single_cell_mechinterp/outputs/eval_bias/tables/noise_summary_random.csv`)
- Table 4: Mapping summary by type (`single_cell_mechinterp/outputs/eval_bias/tables/mapping_summary_by_type.csv`)
- Table 5: Bootstrap CI summary (`single_cell_mechinterp/outputs/eval_bias/tables/bootstrap_summary.csv`)
- Table 6: Method summary by tissue (`single_cell_mechinterp/outputs/eval_bias/tables/method_summary_by_tissue.csv`)
- Table 7: Candidate-set summary by tissue (`single_cell_mechinterp/outputs/eval_bias/tables/candidate_summary_by_tissue.csv`)
- Table 8: Method x candidate set by tissue (`single_cell_mechinterp/outputs/eval_bias/tables/method_candidate_by_tissue.csv`)
- Figure 1: Edge coverage by mapping stage (`single_cell_mechinterp/outputs/eval_bias/figures/edge_coverage_mapping_stage.png`)
- Figure 2: Edge coverage heatmap (`single_cell_mechinterp/outputs/eval_bias/figures/edge_coverage_heatmap.png`)
- Figure 3: Edge coverage by mapping policy (`single_cell_mechinterp/outputs/eval_bias/figures/edge_coverage_by_policy.png`)
- Figure 4: Candidate-set AUPR distribution (`single_cell_mechinterp/outputs/eval_bias/figures/aupr_candidate_set_box.png`)
- Figure 5: Candidate size by protocol (`single_cell_mechinterp/outputs/eval_bias/figures/candidate_size_bar.png`)
- Figure 6: AUPR vs base rate (`single_cell_mechinterp/outputs/eval_bias/figures/aupr_base_rate_scatter.png`)
- Figure 7: Median AUPR by method (`single_cell_mechinterp/outputs/eval_bias/figures/aupr_by_method.png`)
- Figure 8: Method x candidate-set heatmap (`single_cell_mechinterp/outputs/eval_bias/figures/aupr_method_candidate_heatmap.png`)
- Figure 9: Median AUPR by mapping policy (`single_cell_mechinterp/outputs/eval_bias/figures/aupr_by_mapping_policy.png`)
- Figure 10: Median AUPR by gold standard (`single_cell_mechinterp/outputs/eval_bias/figures/aupr_by_gold_standard.png`)
- Figure 11: AUPR vs noise rate (`single_cell_mechinterp/outputs/eval_bias/figures/aupr_noise_random.png`)
- Figure 12: AUPR std vs noise rate (`single_cell_mechinterp/outputs/eval_bias/figures/aupr_noise_std_random.png`)
- Figure 13: AUPR CI width by method (`single_cell_mechinterp/outputs/eval_bias/figures/aupr_ci_width_by_method.png`)
- Figure 14: Gene coverage heatmap (`single_cell_mechinterp/outputs/eval_bias/figures/gene_coverage_heatmap.png`)
- Figure 15: AUPR vs candidate size (`single_cell_mechinterp/outputs/eval_bias/figures/aupr_candidate_size_scatter.png`)
- Figure 16: Mapping status by context type (`single_cell_mechinterp/outputs/eval_bias/figures/mapping_status_by_type.png`)
- Figure 17: Median AUPR by method and tissue (`single_cell_mechinterp/outputs/eval_bias/figures/aupr_by_method_tissue.png`)
- Figure 18: Median AUPR by candidate set and tissue (`single_cell_mechinterp/outputs/eval_bias/figures/aupr_by_candidate_set_tissue.png`)
- Figure 19: Method x tissue heatmap (TF-source-target) (`single_cell_mechinterp/outputs/eval_bias/figures/aupr_method_tissue_tf_sources_targets_heatmap.png`)
- Figure 20: Base rate by candidate set and gold standard (`single_cell_mechinterp/outputs/eval_bias/figures/base_rate_bar.png`)

## References
- Cui H, Wang C, Maan H, Pang K, Luo F, Wang B. 2023. scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI. bioRxiv.
- Han H, Cho JW, Lee S, Yun A, Kim H, Bae D, et al. 2018. TRRUST v2: an expanded reference database of human and mouse transcriptional regulatory interactions. Nucleic Acids Research.
- Garcia-Alonso L, Holland CH, Ibrahim MM, Turei D, Saez-Rodriguez J. 2019. DoRothEA: a resource of transcription factor activities and regulons for human and mouse. Genome Research.
- Tabula Sapiens Consortium. 2022. The Tabula Sapiens: a multiple-organ, single-cell transcriptomic atlas of humans. Science.
- Huynh-Thu VA, Irrthum A, Wehenkel L, Geurts P. 2010. Inferring regulatory networks from expression data using tree-based methods. PLoS ONE.
- Moerman T, Aibar S, Gonzalez-Blas CB, et al. 2019. GRNBoost2 and Arboreto: efficient and scalable inference of gene regulatory networks. Bioinformatics.

## Appendix: Tissue-Stratified Tables
### Table A1: Method summary by tissue (median metrics)
| gene_set | prediction_method | aupr_median | auroc_median | f1_median |
| --- | --- | --- | --- | --- |
| immune | grnboost2 | 0.009266 | 0.5016 | 0.004908 |
| immune | genie3 | 0.006433 | 0.5021 | 0.004037 |
| immune | pearson | 0.003418 | 0.5001 | 0.0004216 |
| immune | spearman | 0.003331 | 0.5001 | 0.0004095 |
| immune | random | 0.001482 | 0.5 | 0.000128 |
| immune | scgpt_attention | 0.001241 | 0.5 | 0.0001162 |
| kidney | grnboost2 | 0.02666 | 0.508 | 0.02916 |
| kidney | genie3 | 0.0258 | 0.5082 | 0.0288 |
| kidney | spearman | 0.008355 | 0.501 | 0.003312 |
| kidney | pearson | 0.008167 | 0.501 | 0.003122 |
| kidney | random | 0.002149 | 0.5001 | 0.0003313 |
| kidney | scgpt_attention | 0.001098 | 0.5 | 0 |
| lung | genie3 | 0.02562 | 0.5097 | 0.02916 |
| lung | grnboost2 | 0.02545 | 0.5095 | 0.02855 |
| lung | spearman | 0.01403 | 0.5008 | 0.003159 |
| lung | pearson | 0.01283 | 0.5008 | 0.00318 |
| lung | random | 0.003472 | 0.5001 | 0.0004398 |
| lung | scgpt_attention | 0.00142 | 0.5 | 0.000158 |

### Table A2: Candidate-set summary by tissue (median AUPR, base rate, candidate size)
| gene_set | candidate_set | aupr_median | base_rate_median | candidate_size_median |
| --- | --- | --- | --- | --- |
| immune | all_pairs | 9.102e-05 | 3.23e-05 | 3.593e+09 |
| immune | tf_sources | 0.002933 | 0.002006 | 4.346e+07 |
| immune | tf_sources_targets | 0.01024 | 0.009975 | 1.94e+06 |
| kidney | all_pairs | 0.0006238 | 3.23e-05 | 3.593e+09 |
| kidney | tf_sources | 0.008588 | 0.002006 | 4.346e+07 |
| kidney | tf_sources_targets | 0.02024 | 0.009975 | 1.94e+06 |
| lung | all_pairs | 0.0004189 | 3.23e-05 | 3.593e+09 |
| lung | tf_sources | 0.00909 | 0.002006 | 4.346e+07 |
| lung | tf_sources_targets | 0.01946 | 0.009975 | 1.94e+06 |

### Table A3: Method x candidate set by tissue (median AUPR)
| gene_set   | candidate_set      | prediction_method   |   aupr_median |
|:-----------|:-------------------|:--------------------|--------------:|
| immune     | all_pairs          | genie3              |   0.00190862  |
| immune     | all_pairs          | grnboost2           |   0.00293003  |
| immune     | all_pairs          | pearson             |   5.31693e-05 |
| immune     | all_pairs          | random              |   2.02806e-05 |
| immune     | all_pairs          | scgpt_attention     |   1.82146e-05 |
| immune     | all_pairs          | spearman            |   5.23449e-05 |
| immune     | tf_sources         | genie3              |   0.00295773  |
| immune     | tf_sources         | grnboost2           |   0.00395385  |
| immune     | tf_sources         | pearson             |   0.00341764  |
| immune     | tf_sources         | random              |   0.0014824   |
| immune     | tf_sources         | scgpt_attention     |   0.00124128  |
| immune     | tf_sources         | spearman            |   0.00333066  |
| immune     | tf_sources_targets | genie3              |   0.00847988  |
| immune     | tf_sources_targets | grnboost2           |   0.0104501   |
| immune     | tf_sources_targets | pearson             |   0.0134031   |
| immune     | tf_sources_targets | random              |   0.00889364  |
| immune     | tf_sources_targets | scgpt_attention     |   0.00737991  |
| immune     | tf_sources_targets | spearman            |   0.0144568   |
| kidney     | all_pairs          | genie3              |   0.0247065   |
| kidney     | all_pairs          | grnboost2           |   0.0255466   |
| kidney     | all_pairs          | pearson             |   0.000603619 |
| kidney     | all_pairs          | random              |   9.11127e-05 |
| kidney     | all_pairs          | scgpt_attention     |   1.82309e-05 |
| kidney     | all_pairs          | spearman            |   0.000643915 |
| kidney     | tf_sources         | genie3              |   0.0257991   |
| kidney     | tf_sources         | grnboost2           |   0.026664    |
| kidney     | tf_sources         | pearson             |   0.0277428   |
| kidney     | tf_sources         | random              |   0.00443225  |
| kidney     | tf_sources         | scgpt_attention     |   0.00109844  |
| kidney     | tf_sources         | spearman            |   0.0295816   |
| kidney     | tf_sources_targets | genie3              |   0.0285428   |
| kidney     | tf_sources_targets | grnboost2           |   0.0289758   |
| kidney     | tf_sources_targets | pearson             |   0.0338722   |
| kidney     | tf_sources_targets | random              |   0.0116695   |
| kidney     | tf_sources_targets | scgpt_attention     |   0.00498759  |
| kidney     | tf_sources_targets | spearman            |   0.0373398   |
| lung       | all_pairs          | genie3              |   0.025021    |
| lung       | all_pairs          | grnboost2           |   0.0250927   |
| lung       | all_pairs          | pearson             |   0.000349634 |
| lung       | all_pairs          | random              |   0.000102654 |
| lung       | all_pairs          | scgpt_attention     |   2.64444e-05 |
| lung       | all_pairs          | spearman            |   0.000345453 |
| lung       | tf_sources         | genie3              |   0.0261742   |
| lung       | tf_sources         | grnboost2           |   0.0264467   |
| lung       | tf_sources         | pearson             |   0.0144858   |
| lung       | tf_sources         | random              |   0.00457632  |
| lung       | tf_sources         | scgpt_attention     |   0.00141981  |
| lung       | tf_sources         | spearman            |   0.0143157   |
| lung       | tf_sources_targets | genie3              |   0.0256189   |
| lung       | tf_sources_targets | grnboost2           |   0.0254548   |
| lung       | tf_sources_targets | pearson             |   0.0203678   |
| lung       | tf_sources_targets | random              |   0.0122679   |
| lung       | tf_sources_targets | scgpt_attention     |   0.00933515  |
| lung       | tf_sources_targets | spearman            |   0.0194558   |

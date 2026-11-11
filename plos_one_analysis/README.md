# S1 Data: protocol-conditional ranking stability analyses

This package holds the data and code for the accompanying article (`manuscript.tex`) and its S1 Text. Its scripts regenerate every number and figure in both from the files in this package. For the zero-recovery replay they recompute the summaries from the stored draws; repeating the draws needs the export step described below. It contains:

- archived summary inputs and their provenance;
- analysis scripts;
- focused tests;
- all generated outputs, including the figures.

No network inference, foundation-model computation, or raw-data processing is performed; the only fitted models are the small logistic screens of the screening feasibility exercise. The inputs are summary-score tables and pairwise comparisons, not new biological samples. `inputs/input_provenance.json` lists each input file with its source and SHA-256 checksum. Two of its entries are marked as kept only in the author's working repository and are not included in this package: `immune_reference_panel/label_indices.csv` (see Zero-recovery replay) and `source_classifier_implementation.py` (see Source notes). `archive_manifest.json` lists every file in this package except itself, with its size and checksum, and names the inputs that are not included under `excluded_repository_only`.

## Reproduce

Create a Python 3.11 virtual environment and install the pinned packages:

```bash
pip install -r requirements.txt
```

Then run, from this folder and in this order:

```bash
python protocol_audit.py
python algebra_null.py
python extended_audit.py
python zero_recovery_replay.py summarize
python make_submission_figures.py
python -m pytest tests -q
```

`extended_audit.py` reads `outputs/protocol_metric_matched_rows.csv` and other `protocol_*` tables, so run it after `protocol_audit.py`; it takes a few minutes. `make_submission_figures.py` reads the outputs of the preceding steps. No data downloads are needed.

With the pinned versions in `requirements.txt` and `environment_versions.json`, the regenerated tables and JSON files in `outputs/` are byte-identical to the supplied ones, and so are the figure files when the Arial font is installed (both checked on macOS arm64). On other platforms the last digits of some fitted or resampled values can differ, and without Arial matplotlib falls back to DejaVu Sans, so the figure bytes differ.

## Zero-recovery replay

`outputs/extended_zero_recovery_*` contains all replay draws and summaries, and `python zero_recovery_replay.py summarize` recomputes the summary from the stored draws using only files in this package. Repeating the replay itself takes two further steps.

**1. `export`** runs the source study's score-evaluation pipeline once, with its configuration and a one-edge list, and captures the flattened label vectors of the immune reference panel:

```bash
python zero_recovery_replay.py export --repository <source>
```

Always pass `--repository`; its default, the folder two levels above this one, applies only in the author's working repository. Besides numpy and pandas, the export needs scanpy (imported by the source pipeline), h5py (to read the gene names of the processed immune subset) and PyYAML (to read the pipeline configuration). It was checked with Python 3.10.19, numpy 1.26.4, pandas 2.3.3, scanpy 1.10.4, anndata 0.11.4, h5py 3.15.1 and PyYAML 6.0.3. `<source>` is the absolute path of a folder that contains the paths below. The third column gives each path's location in the public source-study release (https://github.com/Biodyn-AI/grn-evaluation-bias, tag `release1`; doi:10.5281/zenodo.20639910); files marked "not in the release" are not redistributed there, and their sources and checksums follow the table.

| Path under `<source>` | Content | Location in the release |
|---|---|---|
| `network_inference/src/` | Source score-evaluation pipeline | `network_inference/src/` |
| `network_inference/configs/score_eval_grn_baselines_immune.yaml` | Pipeline configuration, identical to `inputs/source_reference_config.yaml` | `configs/score_eval_grn_baselines_immune.yaml` |
| `network_inference/data/hpn_dream_goldstandard.tsv` | HPN-DREAM gold standard (tab-separated, columns `source` and `target`) | not in the release |
| `network_inference/data/hpn_dream_symbol_map.tsv` | HPN-DREAM symbol map | `data/hpn_dream_symbol_map.tsv` |
| `network_inference/data/beeline_gsd_refnetwork.tsv` | BEELINE GSD reference network (tab-separated, columns `source` and `target`) | not in the release |
| `network_inference/data/beeline_gsd_symbol_map.tsv` | BEELINE GSD symbol map | `data/beeline_gsd_symbol_map.tsv` |
| `network_inference/data/dorothea_trrust_union_immune.tsv` | Immune DoRothEA/TRRUST union | `data/dorothea_trrust_union_immune.tsv` |
| `single_cell_mechinterp/src/__init__.py` and `single_cell_mechinterp/src/eval/` | Gene-symbol normalization and metric helpers imported by the pipeline | same paths |
| `single_cell_mechinterp/outputs/tabula_sapiens_immune_subset_hpn_processed.h5ad` | Processed Tabula Sapiens immune subset, which the source pipeline derives from the public atlas; only its gene names are read | not in the release |
| `single_cell_mechinterp/external/hgnc_complete_set.txt` | HGNC symbol table for alias mapping | not in the release |

The four files that are not in the release are:

- `hpn_dream_goldstandard.tsv`: the HPN-DREAM challenge gold standard (Synapse syn1720047), taken from a copy in SIF format (https://raw.githubusercontent.com/ninakersten/Masterthesis/master/Scoring/real%20data/Goldstandard.sif; SHA-256 `5e972f92e40ae9ae5d993e7024ceb3e9693823aed6bbf23ff4f2866b556120a4`) and converted to columns `source` and `target` by keeping the first and last field of each line (144 rows).
- `beeline_gsd_refnetwork.tsv`: the BEELINE GSD reference network, taken from `inputs/example/GSD/refNetwork.csv` of the BEELINE code repository (https://github.com/Murali-group/Beeline) and converted to columns `source` and `target` (86 rows, 79 distinct edges). Every curated GSD network of the BEELINE data record (doi:10.5281/zenodo.3701939; `BEELINE-data/inputs/Curated/GSD/*/refNetwork.csv` in `BEELINE-data.zip`) has the same distinct edges, and the `Gene1` and `Gene2` columns of one of them (`GSD-2000-1`), saved as `source` and `target`, gave the same export.
- `tabula_sapiens_immune_subset_hpn_processed.h5ad`: the source study's processed Tabula Sapiens immune subset. The export reads only its gene names, in file order, so a copy with the same gene names in the same order gives the same export.
- `hgnc_complete_set.txt`: a three-column HGNC table (`Approved symbol`, `Previous symbols`, `Alias symbols`; 50,093 rows) with file date 11 January 2026; the release records its checksum in `bcb2026/data/PROVENANCE.md`. HGNC tables of other dates or layouts can change the alias mapping (the first listed match of an alias wins), and with it the gene universe and the label vectors.

Their SHA-256 checksums, in the format that `shasum -a 256 -c` (or `sha256sum -c`) checks from `<source>`:

```
2020c0e7c8ca8d85001c3434c6ee0e1a0d7f36e6d1f2aba70cf153ca40eb4b13  network_inference/data/hpn_dream_goldstandard.tsv
5cb8428c5fda2610d080eb4d3e139972361857dca5270b91303b6178a55f2ae9  network_inference/data/beeline_gsd_refnetwork.tsv
991d3b7ff0f194aaec790a2b7ab57b5458fb1c58fa199758fad186d869d984c3  single_cell_mechinterp/outputs/tabula_sapiens_immune_subset_hpn_processed.h5ad
b9482a5247a4162017bd9731ac6300757b9b7511ec9a02f99bafd6b4a43a89fc  single_cell_mechinterp/external/hgnc_complete_set.txt
```

The export writes `inputs/immune_reference_panel/label_indices.csv` and overwrites the supplied `gene_universe.txt` and `export_check.json` in the same folder. It stops with an error unless, for every reference, the label-vector positives plus the reference's self-pairs equal the archived reference size in `inputs/score_eval_grn_baselines_immune.csv` (127, 48 and 11,583). A missing HGNC table also ends in this error, because the pipeline then silently skips HGNC alias mapping. The count check does not detect every change of the alias mapping, so compare the three files with their SHA-256 checksums in `inputs/input_provenance.json`, where the entry for `label_indices.csv` gives the checksum of the author's file. With this layout and the package versions above, all three match.

**2. `replay`** (a few minutes) rebuilds the label vectors from `inputs/immune_reference_panel/` and scores, with the source AUPR formula, ten replicates of nine seeded lists that recover no HPN-DREAM or BEELINE edge:

```bash
python zero_recovery_replay.py replay --replicates 10 --seed 42
```

Each list has the archived number of distinct listed pairs of one method: its recovered DoRothEA/TRRUST edges divided by its precision there or, for PIDC full, whose precision is zero, its number of listed rows. The replay writes `outputs/extended_zero_recovery_replay.csv` and `outputs/extended_zero_recovery_fidelity.json` and then runs `summarize`.

The reconstructed label indices derive from reference networks with share-alike, non-commercial, academic-use, or registered-access terms, so `label_indices.csv` is not included. `inputs/immune_reference_panel/` contains the gene universe and the export check (label-vector positives, self-pairs in the gene universe, and the pipeline AUPR of a one-edge list).

## Scripts and outputs

| Script | Output files | Contents |
|---|---|---|
| `protocol_audit.py` | `outputs/protocol_*` (CSV, JSON); `outputs/figures/protocol_*.png` | Transition-level reversal fractions; method-pair, transition-cell and tissue-cluster resampling; method and tissue deletion; margin sweeps; matched metric keys and exclusions; the source-reference inventory; the held-out screening audit |
| `algebra_null.py` | `outputs/algebra_*` (CSV, JSON); `outputs/figures/algebra_*` (PNG, TIFF) | Accounting controls; direction checks; the illustrative fixed-distribution prevalence example; the within-cell method-label permutation control; Gaussian score-perturbation sensitivities |
| `extended_audit.py` | `outputs/extended_*` except `extended_zero_recovery_*` | Recovered reference edges, endpoint informativeness and the tie-aware average precision (AP) of lists that recover nothing; base-rate-scaled margin sweeps; the noise yardstick (including the 1/√2 difference threshold); rounding-interval simulations and exact tests for the metric comparison; the release check and the analyses that use release values (see Source notes); method-level resampling, leave-two-methods-out, and the widths of all dependence summaries; screening permutation nulls, fixed binormal-score power, and power under three other designs (pair-shared noise, a trained screen, and a top-concentrated signal); held-out shift bounds; derived scale quantities |
| `zero_recovery_replay.py` | `outputs/extended_zero_recovery_*`; `export` writes `inputs/immune_reference_panel/` | The zero-recovery replay |
| `make_submission_figures.py` | `outputs/figures/Fig1`–`Fig7` (PNG, TIFF), `outputs/figures/S1_FigA.png`, `outputs/figures/S1_FigB.png` | Figs 1–7 and S1 Text Figs A and B; Fig 6 and S1 Text Fig A are made from figure files written by `algebra_null.py` and `protocol_audit.py` (see Figure files) |

In `outputs/extended_reference_endpoints.csv`, `base_rate` is the reported base rate (reference size, which counts self-pairs, over candidates), `label_vector_base_rate` is the label-vector positives of `inputs/immune_reference_panel/export_check.json` over candidates, and `tie_aware_ap_if_zero_recovery` is the exact tie-aware AP of a list that recovers no reference edge, which equals the label-vector base rate (empty where a list recovers an edge).

### Figure files

All figure files in `outputs/figures/` are 300-DPI images.

| File | Written by | Use |
|---|---|---|
| `Fig1.png`–`Fig7.png`, `Fig1.tif`–`Fig7.tif` | `make_submission_figures.py` | Figs 1–7; `Fig6.png` and `Fig6.tif` are RGB conversions of `algebra_label_noise_controls.png` |
| `S1_FigA.png`, `S1_FigB.png` | `make_submission_figures.py` | S1 Text Figs A and B; `S1_FigA.png` is a copy of `protocol_screening_audit.png` |
| `algebra_label_noise_controls.png`, `algebra_label_noise_controls.tif` | `algebra_null.py` | Source of Fig 6 |
| `protocol_screening_audit.png` | `protocol_audit.py` | Source of S1 Text Fig A |
| `algebra_structure.png`, `algebra_structure.tif` | `algebra_null.py` | Diagnostic plot, not cited in the article: accounting ratios of the observed reversals, and the positive-rate and sign-switch controls |
| `algebra_prevalence_only.png`, `algebra_prevalence_only.tif` | `algebra_null.py` | Diagnostic plot, not cited: ROC curves and population AUPR of the illustrative prevalence example |
| `protocol_axis_sensitivity.png` | `protocol_audit.py` | Diagnostic plot, not cited: reversal fractions with their method-pair resampling intervals, and the fractions after deleting one method (equal values stacked). The candidate-set and tissue fractions, which share one estimand, are drawn in one panel; the reference and mapping fractions are drawn in separate panels that are not comparable with it or with each other |
| `protocol_matched_metrics.png` | `protocol_audit.py` | Diagnostic plot, not cited: reversal counts and memberships on the 40 matched metric keys, also shown in Fig 7 |

The primary estimand is the fraction of eligible, non-tied method-pair/protocol-transition comparisons that reverse. Tissue any/all-candidate aggregation uses a different unit and is secondary. Only the candidate-set and tissue fractions share methods, metric and implementation, so the four fractions do not rank the axes. Percentile intervals are conditional resampling summaries: method pairs share methods, and the alternative groupings have few clusters. They do not quantify uncertainty over biological replicates or unobserved tissues.

## Source notes

Source Tables A1–A3 and the source report text are archived unedited in `inputs/source_workshop_tables.md`, so that its checksum matches the source. Its opening "Status", "Next Steps" and "Conversation Summary" sections are the source author's editorial notes; the report text begins at "## Abstract".

The immune reference-panel and mapping-panel score files are byte-identical to those in the public source-study release.

`inputs/source_release_method_summary_by_tissue.csv` is a verbatim copy of the unrounded tissue/method medians in that release (tag `release1`, file `single_cell_mechinterp/outputs/eval_bias/tables/method_summary_by_tissue.csv`, doi:10.5281/zenodo.20639910). The release belongs to a later version of the source protocol. `extended_audit.py` reports which printed Table A1 medians the release reproduces at four significant figures (`outputs/extended_release_value_check.csv`). A release value is used only for a scorer without stochastic fitting (Pearson, Spearman, seeded random scores, or scGPT attention; the tree ensembles GENIE3 and GRNBoost2 can differ between runs), and only in a tissue where the release reproduces every printed median of these four scorers in all three metrics. This holds for kidney and lung, which give 24 usable values, and fails for immune. In `outputs/extended_release_value_check.csv`, the column `deterministic_scorer` marks the four scorers, `tissue_validated` the tissues where all their medians are reproduced, and `usable` the 24 values. The usable values replace the printed medians in three analyses: the check of the five printed AUROC ties, one of which they break (`outputs/extended_metric_tie_resolution.csv`); a second rounding simulation in which they are fixed (`outputs/extended_metric_rounding_conditional_summary.csv` and `outputs/extended_metric_rounding_conditional_keys.csv`); and the matched comparison that includes the broken tie (41 keys; `resolved_matched_comparison` under the key `release_value_check` of `outputs/extended_summary.json`). Agreement at four significant figures supports, but cannot prove, that these values equal the archived unrounded medians, so these results are conditional on that equality. The archived input files are not modified.

The source report describes reference-specific candidate universes, whereas the saved configuration (`inputs/source_candidate_config.yaml`) defines TRRUST universes. This discrepancy is recorded in `inputs/provenance_limitations.txt`. The analyses make no claim about which construction produced the archived medians.

The comparison-level row files that the scripts read (`candidate_shift_pairwise_rows.csv`, `tissue_shift_pairwise_rows_candidate_conditional.csv`, `reference_shift_pairwise_rows.csv` and `mapping_policy_pairwise_rows_f1.csv` in `inputs/`) were written by an earlier exploratory script. That script is listed with its checksum in `inputs/input_provenance.json` (`source_classifier_implementation.py`) and kept only in the author's working repository. `protocol_audit.py` rebuilds every eligible comparison key and its reversal indicator from the source score tables, and stops if these archived rows differ. The exploratory script also wrote `tissue_shift_pairwise_rows_deduplicated.csv` (its tissue any/all-candidate aggregation, byte-identical to `outputs/protocol_tissue_candidate_aggregation.csv`, which `protocol_audit.py` recomputes), `original_revised_summary.json`, `margin_sweep_*`, `nulls_comparison.csv`, `instability_quantile_sweep.csv` and `instability_learned_*`; `instability_loo_*` come from another exploratory run (sources in `inputs/input_provenance.json`). These files, the hESC and hHep metric inventories (`inputs/hESC_metrics_inventory.csv` and `inputs/hHep_metrics_inventory.csv`, which `outputs/protocol_reference_inventory.csv` describes), the source study's summary script (`inputs/source_summary_implementation.py`) and its configurations (`inputs/source_candidate_config.yaml`, `inputs/source_reference_config.yaml`) are provenance records only: no script in this package executes or reads them, and every reported result is recomputed from the comparison rows and source tables. The summary script and the two configurations are identical to `single_cell_mechinterp/scripts/generate_eval_bias_summary.py`, `single_cell_mechinterp/configs/eval_bias.yaml` and `configs/score_eval_grn_baselines_immune.yaml` of the source release.

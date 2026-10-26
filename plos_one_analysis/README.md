# S1 Data: protocol-conditional ranking stability analyses

This package reproduces every number and figure in the accompanying article (`manuscript.tex`). It contains:

- archived summary inputs;
- analysis scripts;
- focused tests;
- all generated outputs.

No model training or data ingestion is performed. The inputs are summary-score tables and pairwise comparisons, not new biological samples. `inputs/input_provenance.json` lists each input file, its source, and its SHA-256 checksum. `archive_manifest.json` lists every file in this package with its size and checksum.

## Reproduce

Create a Python 3.11 virtual environment and install the pinned packages:

```bash
pip install -r requirements.txt
```

Then run, from this folder:

```bash
python protocol_audit.py
python algebra_null.py
python extended_audit.py
python zero_recovery_replay.py summarize
python make_submission_figures.py
python -m pytest tests -q
```

`extended_audit.py` reads `outputs/protocol_metric_matched_rows.csv`, so run it after `protocol_audit.py`. With the pinned versions in `requirements.txt` and `environment_versions.json`, the regenerated files in `outputs/` are byte-identical to the supplied ones. No data downloads are needed.

## Zero-recovery replay

`zero_recovery_replay.py` has two further steps:

- `export` runs the source study's score-evaluation pipeline once to reconstruct the immune reference panel's label vectors. It needs:
  - the source pipeline and configuration (`network_inference/` and `configs/score_eval_grn_baselines_immune.yaml` in the public source-study release, doi:10.5281/zenodo.20639910);
  - the HPN-DREAM, BEELINE and DoRothEA/TRRUST reference networks;
  - the processed Tabula Sapiens immune subset that the source pipeline derives from the public atlas.
- `replay` scores seeded lists that recover no reference edge, using the source AUPR formula.

The reconstructed label indices derive from reference networks with share-alike, non-commercial, academic-use or registered-access terms, so they are not included. `inputs/immune_reference_panel/` contains:

- the gene universe;
- the export check (label positives, self-pairs, and the pipeline AUPR of a one-edge list).

`outputs/extended_zero_recovery_*` contains all replay draws and summaries, and `zero_recovery_replay.py summarize` recomputes the summary from the stored draws.

## Scripts and outputs

| Script | Output files | Contents |
|---|---|---|
| `protocol_audit.py` | `outputs/protocol_*` | Transition-level reversal fractions; method-pair, transition-cell and tissue-cluster resampling; method and tissue deletion; margin sweeps; matched metric keys and exclusions; the source-reference inventory; the held-out screening audit |
| `algebra_null.py` | `outputs/algebra_*` | Accounting controls; direction checks; the illustrative fixed-distribution prevalence example; the within-cell method-label permutation control; Gaussian score-perturbation sensitivities |
| `extended_audit.py` | `outputs/extended_*` | Recovered reference edges and endpoint informativeness; base-rate-scaled margin sweeps; the noise yardstick; rounding-interval and exact tests for the metric comparison; method-level resampling and leave-two-methods-out; screening permutation nulls and power; held-out shift bounds |
| `zero_recovery_replay.py` | `outputs/extended_zero_recovery_*` | The zero-recovery replay |
| `make_submission_figures.py` | `outputs/figures/` | Figs 1–7 and S1 Text Figs A–B |

The primary estimand is the fraction of eligible, non-tied method-pair/protocol-transition comparisons that reverse. Tissue any/all-candidate aggregation uses a different unit and is secondary. Percentile intervals are conditional resampling summaries: method pairs share methods, and the alternative groupings have few clusters. They do not quantify uncertainty over biological replicates or unobserved tissues.

## Source notes

Source Tables A1–A3 and the source report text are archived unedited in `inputs/source_workshop_tables.md`, so that its checksum matches the source. Its opening "Status", "Next Steps" and "Conversation Summary" sections are the source authors' editorial notes; the report text begins at "## Abstract".

The immune reference-panel and mapping-panel score files are byte-identical to those in the public source-study release.

The source report describes reference-specific candidate universes, whereas the saved configuration (`inputs/source_candidate_config.yaml`) defines TRRUST universes. This discrepancy is recorded in `inputs/provenance_limitations.txt`. The analyses make no claim about which construction produced the archived medians.

`inputs/source_classifier_implementation.py` and the inputs `original_revised_summary.json`, `instability_*`, `margin_sweep_*` and `nulls_comparison.csv` are archived from earlier exploratory code that produced the comparison-level row files. They are provenance records only: the scripts in this package recompute every reported result from the comparison rows and source tables, and they do not execute or read those archived results.

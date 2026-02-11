# GRN Ranking Reversal Theory

This repository contains code and artifacts for the paper:
**Ranking Reversal Theory Under Candidate-Set Shift in GRN Benchmarking**.

## Contents
- `paper/`: manuscript source and compiled PDF.
- `scripts/`: analysis scripts for ranking-reversal decomposition and paper-grade result generation.
- `results/`: generated tables and figures used in the manuscript.

## Reproducibility
1. Install Python 3.10+ with dependencies: `pandas`, `numpy`, `matplotlib`.
2. Run:
   ```bash
   python scripts/build_paper_results.py \
     --report-markdown <path-to-eval-bias-paper-draft.md> \
     --score-eval-baseline <path-to-score_eval_grn_baselines_immune.csv> \
     --policy-dir <path-to-score_eval_probe_priors-outputs> \
     --out-dir results/generated
   ```
3. Compile paper:
   ```bash
   pandoc paper/paper.md -s -o paper/paper.pdf --pdf-engine=xelatex
   ```

The script outputs include reversal summaries, decomposition tables, permutation-null diagnostics, and publication figures.

## Data access
The analyses use previously generated benchmark outputs from the evaluation-bias protocol and immune baseline GRN evaluations. Source dataset access and generation details are documented in the manuscript Methods and linked references.

# GRN Ranking Reversal Theory

This repository contains code, source data, and regenerated artifacts for the paper:
**Ranking Reversal Theory Under Candidate-Set Shift in GRN Benchmarking**
(double-blind submission; author info withheld during review).

## Contents

- `paper/` — manuscript source (`paper.md`) and compiled PDF.
- `data/` — input CSVs: Table A1/A2/A3 summary medians from the source study plus
  the immune GRN-baseline score-evaluation CSV and four mapping-policy CSVs.
- `scripts/analysis.py` — self-contained pipeline that reads `data/` and writes
  every table and figure used in the manuscript.
- `results/tables/` — regenerated CSV outputs (cluster-bootstrap CIs, per-cell
  BH-FDR, null comparisons, margin sweeps, instability classifier, metric
  robustness).
- `results/figures/` — figures referenced from `paper/paper.md`.
- `results/summary.json` — single-file summary of all headline numbers.
- `requirements.txt` — pinned Python dependencies.

## Reproducing the analysis

Requires Python 3.9+ and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
python scripts/analysis.py
```

This regenerates everything under `results/` from scratch. Re-running overwrites
prior outputs. All seeds are fixed (project seed 42; permutation seed 42;
bootstrap seed 42). Numerical decomposition closure is verified at runtime to
within 1e-9.

## Headline numbers (cluster-bootstrap 95% CIs, verified against `results/summary.json`)

| Axis | n | k | Rate | Cluster CI |
|---|---:|---:|---:|---|
| Candidate-set shift | 135 | 22 | 0.16 | [0.07, 0.27] |
| Tissue shift (candidate-conditional) | 135 | 26 | 0.19 | [0.09, 0.31] |
| Tissue shift (dedup, any candidate) | 45 | 17 | 0.38 | [0.18, 0.58] |
| Tissue shift (dedup, all candidates) | 45 | 1 | 0.02 | [0.00, 0.07] |
| Reference shift (immune slice) | 106 | 34 | 0.32 | [0.20, 0.43] |
| Mapping-policy shift | 165 | 0 | 0.00 | [0.00, 0.00] |

Magnitude decomposition on the candidate axis:

- `|calibration term| > |base-rate term|` in 22/22 reversal rows vs 56/113 in non-reversal rows.
- **Cluster-permutation Mann–Whitney U** (5,000 permutations within method-pair
  clusters): observed U = 2075, rank-biserial r = 0.67 (large effect),
  permutation p < 2 × 10⁻⁴ (0/5000 null draws exceeded the observed U).
- For comparison, an iid MWU yields p = 3.6 × 10⁻⁷, but the iid assumption is
  invalid because the 135 rows share method pairs across only 15 unique
  clusters (7 with any reversals). The cluster-permutation p is the load-
  bearing significance statement.
- Mean magnitude ratio in reversal rows: 1.54; median: 1.02.
- Sign-tally counterfactuals (e.g. "0/22 reversals persist if only base rate
  changes") are **algebraically pinned under positive base rates** and are
  therefore not reported as findings — see paper §3.2 and §5.2.

## Data source

The summary medians (`table_a1/a2/a3_*.csv`) and the score-evaluation CSVs
under `data/` are previously generated benchmark outputs from the evaluation-
bias study cited in the manuscript Methods section. They are reproduced here
for self-containment of the analysis pipeline. See the manuscript for the
originating reference.

## Note on prior commit

A prior commit in this repository's history (`Initial release: …`) contained
the first-pass analysis whose sign-attribution claims were retracted as
algebraically pinned. The current commit replaces those outputs with the
revised analysis described in the manuscript. The buggy prior state remains in
git history for transparency.

## License

Code released under a permissive license; details will be specified after
author de-anonymization at acceptance.

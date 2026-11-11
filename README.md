# GRN Ranking Reversal Theory

Code and data for the article **"Ranking reversal theory under candidate-set shift in gene regulatory network benchmarking"** (PLOS ONE submission).

## Current analysis: `plos_one_analysis/`

`plos_one_analysis/` is the complete reproduction package for the article. It is identical to the article's S1 Data archive. It contains:

- the archived source summary tables and score files;
- comparison-level rows, eligibility and exclusions;
- the analysis scripts and focused tests;
- pinned requirements;
- every generated table and figure.

Reproduce every reported number and figure with Python 3.11:

```bash
cd plos_one_analysis
pip install -r requirements.txt
python protocol_audit.py
python algebra_null.py
python extended_audit.py
python zero_recovery_replay.py summarize
python make_submission_figures.py
python -m pytest tests -q
```

With the pinned versions, the regenerated `outputs/` are byte-identical to the committed ones (checked on macOS arm64; the figure files match only when the Arial font is installed). `plos_one_analysis/README.md` documents:

- each script and its outputs;
- the zero-recovery replay. Its `export` and `replay` steps also need the source study's pipeline, configuration and symbol maps (doi:10.5281/zenodo.20639910), the reference networks, an HGNC symbol table, and the processed Tabula Sapiens immune subset; `plos_one_analysis/README.md` lists each file with its source and checksum.

### Main findings

| Axis | Reversals | Notes |
|---|---|---|
| Candidate-set shifts | 22/135 (16.3%) | AUPR; same methods, metric and implementation as tissue shifts |
| Tissue shifts | 26/135 (19.3%) | AUPR; same methods, metric and implementation as candidate-set shifts |
| Identifier-mapping shifts | 0/165 | F1; three of four policies give identical scores and the fourth rescales them almost proportionally, so no comparison could reverse |
| Immune reference panel | 34/106 (32.1%) | Not a reference effect; see below |

Only the candidate-set and tissue fractions share an estimand; the four values do not rank the axes. The reference reversals are not evidence that reference choice matters more. Under HPN-DREAM and BEELINE no method recovers any reference edge, so their AUPRs reflect list length and tie order.

The product-rule decomposition is descriptive accounting: with increasing base rates, normalized-gap dominance in reversal rows is forced by algebra.

## Earlier exploratory version (historical)

The folders `paper/`, `data/`, `scripts/` and `results/` hold an earlier exploratory version of this analysis, kept for reference. Some of its interpretations are superseded by the article and by `plos_one_analysis/`:

- Its significance test of magnitude dominance does not stand, because that dominance is algebraically forced.
- Its reference-shift interpretation does not account for references at which no method recovers edges.

Do not use those folders to reproduce the article.

`posters/` contains a conference poster based on the earlier version.

## License

No license file has been added yet.

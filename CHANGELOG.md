# Changelog

## 2.0.0 (revision of manuscript 2016656)

### Fixed
- **Tied importance scores.** Features with exactly zero importance, which are
  common for L1-regularized logistic regression, were previously ordered by their
  column position. Top-k selections now contain only features with strictly
  positive importance (`stability.top_k_features`), consensus aggregation assigns
  average ranks to ties, and the Kuncheva index supports selections of unequal
  size. Results for Random Forest, Extremely Randomized Trees and univariate ANOVA
  are unchanged. Results for L1-regularized logistic regression and for the
  consensus panels changed, and the consensus panels are now comparable to, rather
  than better than, the best single method.

### Added
- `src/simulate.py`, a simulation study with known informative features placed at
  random column positions.
- `src/stability_vs_k.py`, stability and common-classifier accuracy at
  k = 10, 25, 50 and 100.
- Checkpoint and resume support in `robustness.py`, `hardening.py`,
  `stability_vs_k.py` and `simulate.py`.
- `scripts/make_figures.py` and `scripts/make_supplementary_tex.py`, which
  regenerate every figure and supplementary table from `results/`.
- `figures/` and `supplementary/` with the revised article figures and tables.

### Changed
- All files in `results/` were regenerated with the corrected code.

## 1.0.0

- Initial release accompanying the original submission.

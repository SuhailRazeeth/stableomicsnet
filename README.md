# StableOmicsNet

Code and derived results for the article

> **Stability Is Not Accuracy: A Comparative Study of Biomarker Reproducibility
> and Cross-Cohort Transfer in Classical and Network-Based Multi-Omics Models**
> M.M. Mohamed Mufassirin and M.S. Suhail Razeeth. *Under review (Frontiers, manuscript 2016656).*

StableOmicsNet runs one identical stability assessment on classical and
network-based multi-omics methods. It measures how reproducible the **biomarker
list** of each method is when the patient sample is perturbed, quantifies how far
methods agree on which molecules matter, tests whether within-cohort stability is
associated with **cross-cohort transfer**, and checks these relationships in a
**simulation with known ground truth**.

---

## Modules

| Module | Purpose |
|---|---|
| `src/stability.py` | Jaccard and chance-corrected Kuncheva indices, tie-safe top-k selection, consensus and stability-weighted rank aggregation |
| `src/methods.py` | Unified interface for the six method classes |
| `src/gcn_method.py` | Two-layer graph convolutional network over a patient-similarity graph (PyTorch, saliency importance) |
| `src/snf_method.py` | Similarity Network Fusion (NumPy, label propagation and spectral-embedding importance) |
| `src/experiment_stability.py` | Accuracy and stability under stratified subsampling (Tables 2 and 3, Figure 1) |
| `src/stability_vs_k.py` | Stability and common-classifier accuracy across panel sizes (Figure 2, Supplementary Table S5) |
| `src/crosscohort.py` | Panel selection and TCGA-BRCA to METABRIC transfer (Table 4, Figure 3) |
| `src/robustness.py` | Panel-size sweep, bootstrap intervals and paired significance tests (Table 5, Figure 4) |
| `src/hardening.py` | Both transfer directions with three downstream classifiers (Table 6, Figure 5) |
| `src/simulate.py` | Simulation study with known informative features (Figure 6, Supplementary Table S6) |
| `src/enrichment.py` | Pathway and Gene Ontology enrichment of consensus panels with Enrichr (Table 7, Figure 7) |
| `src/prepare_kipan.py` | Builds the KIPAN dataset from UCSC Xena |
| `scripts/make_figures.py` | Regenerates Figures 1 to 7 at 300 dpi from `results/` |
| `scripts/make_supplementary_tex.py` | Regenerates the Supplementary Tables S1 to S6 from `results/` |

## Installation

```bash
pip install -r requirements.txt
```

PyTorch is needed for the GCN and `gseapy` for enrichment. Both are optional, and
the framework registers the GCN only if PyTorch is available.

## Data

**No primary data are redistributed in this repository.** All datasets are
third-party and are either downloaded by the scripts or obtained from the
original providers.

| Dataset | Source | Obtained by |
|---|---|---|
| TCGA-BRCA (mRNA, methylation, miRNA) | Preprocessed distribution of MOGONET, <https://github.com/txWang/MOGONET> | `download_data.download_mogonet("BRCA")` |
| ROSMAP (mRNA, methylation, miRNA) | Preprocessed distribution of MOGONET, with original data from the Rush Alzheimer's Disease Center (RADC) under a Data Use Agreement | `download_data.download_mogonet("ROSMAP")` |
| KIPAN (mRNA, CNV, miRNA) | UCSC Xena, TCGA KIRC, KIRP and KICH cohorts | `prepare_kipan.build_kipan()` |
| METABRIC (mRNA, clinical) | cBioPortal data hub | `download_data.download_metabric()` |

> **ROSMAP note.** ROSMAP individual-level data are normally distributed under a
> Data Use Agreement through RADC and the AD Knowledge Portal. This work used the
> preprocessed matrices redistributed with MOGONET. Users should obtain ROSMAP
> access through the appropriate channel, and the data are not hosted here.

## Reproducing the article

```python
import sys; sys.path.append("src")

# 1. Data
from download_data import download_mogonet, download_metabric
download_mogonet("BRCA"); download_mogonet("ROSMAP"); download_metabric()
from prepare_kipan import build_kipan; build_kipan()

# 2. Stability and accuracy (Tables 2 and 3, Figure 1)
from experiment_stability import run_experiment
for ds in ("BRCA", "ROSMAP", "KIPAN"):
    run_experiment(ds, n_reps=20)

# 3. Panel sizes (Figure 2, Supplementary Table S5)
from stability_vs_k import run_stability_vs_k
run_stability_vs_k(["BRCA", "ROSMAP", "KIPAN"], ks=(10, 25, 50, 100), n_reps=20)

# 4. Cross-cohort transfer (Table 4, Figure 3)
from crosscohort import run_crosscohort
run_crosscohort(n_reps=20, top_k=50)

# 5. Transfer across panel sizes with paired significance (Table 5, Figure 4)
from robustness import run_robustness
run_robustness(ks=(10, 25, 50, 100), n_reps=20, B=1000)

# 6. Both directions and three classifiers (Table 6, Figure 5)
from hardening import run_hardening
run_hardening(top_k=25, n_reps=20, B=1000)

# 7. Simulation with known ground truth (Figure 6, Supplementary Table S6)
from simulate import run_simulation
run_simulation(n_sim=20, n_reps=10)

# 8. Enrichment of the consensus panels (Table 7, Figure 7)
from enrichment import enrich_panel, brca_consensus_symbols
enrich_panel(brca_consensus_symbols(), name="BRCA_consensus")
```

Every long analysis saves a checkpoint after each dataset, panel size or
direction, so an interrupted session (for example on Google Colab) resumes where
it stopped when the same call is repeated.

All outputs are written to `results/`, and the copies committed here are the
outputs reported in the article. Figures and supplementary tables are then
regenerated with

```bash
python scripts/make_figures.py
python scripts/make_supplementary_tex.py
```

### Handling of tied importance scores

Several methods assign an importance of exactly zero to many features (for
example, most coefficients of L1-regularized logistic regression). Top-k
selections therefore contain only features with strictly positive importance,
tied features receive their average rank in every consensus aggregation, and the
Kuncheva index uses its generalization to selections of unequal size. Ordering
tied features by column position would make results depend on the arbitrary
order of the input matrix. Release v1.0.0, which accompanied the original submission, did
not handle ties in this way (see `CHANGELOG.md`).

### Reproducibility

The four classical selectors and the stability-filtered consensus are reproduced
exactly across independent executions. The two **network models are not**,
because floating-point accumulation on GPUs and in multithreaded linear algebra
is not associative and the SNF importance is read from a near-degenerate spectral
embedding. Section 3.10 and Supplementary Table S4 of the article quantify this
effect. Small differences for the GCN, SNF and the equal-weight consensus should
therefore be expected on other hardware.

## Repository layout

```
src/            analysis framework, dataset download and KIPAN construction
scripts/        figure and supplementary-table generators
results/        result tables (CSV) and diagnostic plots reported in the article
figures/        Figures 1 to 7 of the article (300 dpi)
supplementary/  Supplementary Tables S1 to S6 (LaTeX)
```

Paths are resolved automatically. Set the `STABLEOMICSNET_DIR` environment
variable to choose where `data/` and `results/` are stored. Otherwise the code
uses the Google Drive folder when running in Colab and the repository root
elsewhere.

## License

The code is released under the MIT License (see `LICENSE`). Third-party datasets
remain under the terms of their respective providers.

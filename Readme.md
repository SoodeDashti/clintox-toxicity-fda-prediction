# Molecular Toxicity & FDA Approval Prediction (ClinTox)

An end-to-end machine learning system that predicts clinical trial toxicity risk and FDA approval likelihood for a molecule from its chemical structure alone — combining cheminformatics, multi-task classification, model interpretability, and a full deployment stack including a local LLM agent.

## What makes this project different

This is my second full ML portfolio project, after an earlier [diabetes risk prediction project](#) that covered the database → ML → FastAPI → Docker pipeline. This project intentionally goes further in two directions:

1. **A bioinformatics/clinical framing, not just a classifier.** Every step — from how molecules are cleaned, to how the train/test split is designed, to what the SHAP results mean — is interpreted the way a medicinal chemist or biologist would, not just reported as a metric.
2. **A local LLM agent with tool-calling**, layered on top of the same FastAPI service. Ask it a question in plain language ("Is ibuprofen likely to be toxic?") and it looks up the molecule via PubChem, runs it through the trained model, and answers — using the project's own model, not generic LLM knowledge.

## Table of Contents

- [Dataset](#dataset)
- [Data Cleaning](#data-cleaning)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Structural Alerts](#structural-alerts)
- [Chemical Space Visualization](#chemical-space-visualization)
- [Methodology Decisions](#methodology-decisions)
- [Model Results](#model-results)
- [Benchmark Comparison](#benchmark-comparison)
- [Model Interpretability (SHAP)](#model-interpretability-shap)
- [3D Molecular Structure Viewer](#3d-molecular-structure-viewer)
- [Architecture](#architecture)
- [Setup](#setup)
- [Limitations](#limitations)
- [References](#references)

## Dataset

Using the **ClinTox** dataset (1,484 molecules), part of the [MoleculeNet](https://moleculenet.org/) benchmark suite. Each molecule has a SMILES string and two binary labels: `FDA_APPROVED` and `CT_TOX` (failed clinical trials due to toxicity).

### Dataset Provenance

ClinTox combines molecules from two distinct sources:
- **Non-toxic/FDA-approved molecules** — from the [SWEETLEAD](https://simtk.org/projects/sweetlead) database of FDA-approved drugs.
- **Toxic molecules** — from [AACT](https://aact.ctti-clinicaltrials.org/) (Aggregate Analysis of ClinicalTrials.gov), specifically drugs that failed trials due to toxicity.

This explains a pattern found during EDA: **zero molecules** in the dataset are both non-toxic and not FDA-approved — every non-toxic molecule is also approved, because "non-toxic" is effectively defined as "currently an approved drug." This means the two labels are partly correlated *by dataset construction*, not purely by underlying chemistry — an important caveat when interpreting model performance, and one I verified directly rather than assuming.

## Data Cleaning

RDKit failed to parse 4 of 1,484 SMILES strings (0.27%):
- 1 organometallic (platinum-based) compound — RDKit doesn't fully support metal-coordinate bonds by default. Notably, platinum compounds are used in real chemotherapy drugs (e.g. cisplatin), so this is a known cheminformatics tooling limitation, not a data error.
- 3 molecules with kekulization errors, a common edge case in large chemical datasets.

These 4 were dropped, leaving 1,480 valid molecules.

**On missing-value handling:** rather than a generic imputation strategy, missingness in computed molecular descriptors here reflects a *parsing failure*, not a clinical signal — unlike, say, an unrecorded lab test in EHR data. So invalid molecules were excluded outright rather than imputed, and this reasoning is stated explicitly rather than assumed.

## Exploratory Data Analysis

**Class balance:** both labels are heavily imbalanced — `FDA_APPROVED` 93.6% positive, `CT_TOX` 7.6% positive. This shaped every downstream decision (class weighting, threshold selection, choice of evaluation metric).

**Physicochemical descriptors** (MolWt, LogP, TPSA, H-bond donors/acceptors, rotatable bonds) were computed via RDKit and show typical drug-like, right-skewed distributions.

## Structural Alerts

Molecules were screened against known toxicophore substructures (PAINS, epoxides, Michael acceptors, thioureas, aromatic amines) using RDKit's `FilterCatalog` and custom SMARTS patterns.

Most patterns showed no meaningful difference between toxic and non-toxic molecules — expected for PAINS, which flags assay-interference risk rather than clinical toxicity specifically. The exception: **aromatic amines**, present in 32.1% of toxic molecules vs. 12.7% of non-toxic ones (~2.5x enrichment). This aligns with established toxicology — aromatic amines are metabolically activated into reactive nitrenium ions that damage DNA, historically linked to bladder cancer in dye-industry workers exposed to them. This is a mechanistically grounded finding, not a spurious statistical correlation.

## Chemical Space Visualization

Morgan fingerprints (2048-bit) were reduced to 3D via PCA and plotted interactively (Plotly). Toxic molecules were found scattered throughout the non-toxic cluster rather than forming a distinct region — meaning toxicity here isn't driven by coarse whole-molecule similarity, consistent with toxicity often arising from a specific reactive substructure rather than overall shape. This result directly motivated the structural-alerts analysis above, rather than being a dead end.

## Methodology Decisions

Several choices here are deliberate and documented, not defaults:

- **Scaffold split, not random split.** Molecules were grouped by Murcko scaffold and split at the scaffold level (816 unique scaffolds → 1,256 train / 224 test), so structurally similar molecules never leak across the train/test boundary. MoleculeNet's official ClinTox protocol uses a random split, which is easier than it looks. Scaffold splitting better reflects real drug-discovery conditions, where the next molecule rarely resembles anything already tested.
- **Calibration, added after diagnosing a real problem.** An initial uncalibrated Random Forest showed signs of overconfidence (36% of test predictions had probability < 0.05 or > 0.95). `CalibratedClassifierCV` with isotonic scaling was tried first and made things *worse* (42% overconfident) — isotonic's non-parametric step function overfits with only ~30 minority-class samples per CV fold. Sigmoid (Platt) scaling, a more constrained parametric method, performed better and was kept.
- **Threshold tuning via Youden's J, not the default 0.5.** On this imbalanced data (6–8% positive class), the default classification threshold is close to meaningless — a model can look "accurate" while barely predicting the minority class. AUC-ROC (which is threshold-independent) was good from the start (0.84–0.85); the real fix was choosing an optimal threshold per task from the ROC curve, which brought balanced accuracy from ~0.52 to ~0.80–0.82.

## Model Results

A multi-task Random Forest (`MultiOutputClassifier` wrapping `CalibratedClassifierCV`, sigmoid calibration, `class_weight="balanced"`, 300 trees, `max_depth=15`, `min_samples_leaf=3`) trained on 2048-bit Morgan fingerprints, using the scaffold split above.

| Task | AUC-ROC | Balanced Accuracy (0.5 threshold) | Balanced Accuracy (optimal threshold) |
|---|---|---|---|
| FDA_APPROVED | 0.852 | 0.526 | **0.823** (threshold = 0.92) |
| CT_TOX | 0.841 | 0.521 | **0.804** (threshold = 0.10) |

**A concrete illustration of train-set memorization vs. genuine generalization:** aspirin appears in ClinTox itself labeled `FDA_APPROVED=0, CT_TOX=1` — despite being a well-known, safe, approved drug in reality. Querying the deployed model on aspirin reproduces this label with high confidence (99.7%), because aspirin was part of the *training set* — this is expected memorization, not a calibration failure. On a genuinely unseen molecule (caffeine, not in ClinTox at all), the same model outputs a moderate, well-calibrated probability (68.6% / 36.0%) rather than an extreme one. This distinction matters for interpreting any prediction from the deployed API: high confidence on a training-set molecule reflects a memorized label, not independent validation — and predictions in general reflect patterns in this dataset, not verified clinical ground truth.

## Benchmark Comparison

The best baseline reported in the original MoleculeNet paper ([Wu et al., 2018](https://doi.org/10.1039/C7SC02664A)) for ClinTox was **Weave** (a graph neural network), AUC-ROC = 0.832, using a standard random split. This project's Random Forest reached **AUC-ROC = 0.852 / 0.841** — matching or exceeding that baseline despite the stricter scaffold split used here, which should make the task harder, not easier.

More recent deep-learning approaches purpose-built for this task report substantially higher scores — [Li et al. (2022)](https://arxiv.org/abs/2204.06614) reached AUC-ROC up to 0.99 using a multi-task deep network with contrastive molecular explanations. That's a considerably more complex architecture, outside the scope of this project's classical ML approach, but a useful reference point for future extensions.

## Model Interpretability (SHAP)

SHAP was applied to the CT_TOX model to identify which molecular substructures drive toxicity predictions. 4 of the top 10 most important fingerprint bits corresponded to a single nitrogen atom in various environments, alongside aromatic ring fragments — broadly consistent with the aromatic-amine structural alert finding above, but revealing a *wider* nitrogen-related signal than any single hand-crafted rule captures. Of the 10 molecules pushed most strongly toward "toxic," 40% matched the aromatic-amine alert (vs. 32% dataset average for toxic molecules) — meaningful overlap, but not a complete one, suggesting SHAP captures a broader chemical signal than one SMARTS pattern alone.

Combining **structural alerts** (a mechanistic, rule-based explanation) with **SHAP** (a statistical, model-derived explanation), and checking where they agree and disagree, gives two independent lines of evidence pointing at the same underlying chemistry — rather than relying on either alone.

## 3D Molecular Structure Viewer

Example toxic and non-toxic molecules are rendered as real 3D structures (RDKit ETKDG embedding + MMFF optimization, `py3Dmol`), each labeled with molecular weight and LogP — connecting the descriptors used quantitatively in the EDA back to actual molecular geometry.

## Architecture

```
notebooks/       EDA, cleaning, modeling, SHAP, MLflow-tracked training
app/              FastAPI service: /predict, /predictions/history, /ask (agent)
dashboard/        Streamlit UI: SMILES prediction + natural-language agent chat
models/           Trained model artifact
data/             ClinTox dataset
analysis/         Saved figures
```

- **FastAPI + PostgreSQL** — every prediction is logged (SMILES, both probabilities, structural alert flag, timestamp) as a simple audit trail.
- **Docker Compose** — three services (`db`, `api`, `dashboard`), fully containerized and independently reproducible.
- **MLflow** — training runs are tracked with parameters, per-task metrics, and the model artifact, so different configurations (e.g. calibration method, tree depth) can be compared rather than overwritten.
- **Local agent (Ollama, `llama3.1:8b`)** — a tool-calling agent exposed via `/ask` and in the dashboard. It can call `lookup_pubchem_by_name` (real-time PubChem lookup by compound name) and `predict_toxicity_tool` (this project's own trained model) to answer natural-language questions, grounded in this project's actual predictions rather than the LLM's generic training knowledge.

## Setup

1. Download the ClinTox dataset and place it at `data/clintox.csv.gz`:
   `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz`
2. Copy `.env.example` to `.env` and set your own database credentials.
3. `docker-compose up --build`
4. API docs: `http://localhost:8001/docs` · Dashboard: `http://localhost:8502`
5. (Optional, for the agent) Install [Ollama](https://ollama.com), run `ollama pull llama3.1:8b`, and keep it running on the host machine.

> If ports 5433/8001/8502 conflict with something already running on your system, change the host-side ports in `docker-compose.yml`.

## Limitations

- **4 molecules (0.27%) excluded** due to RDKit parsing failures (1 organometallic, 3 kekulization errors) — a tooling limitation, not a data quality issue.
- **Label correlation is partly an artifact of dataset construction** (see Dataset Provenance) — the model may partly learn "toxic implies not approved" as a dataset pattern, not a purely chemical one.
- **Predictions on training-set molecules (e.g. aspirin) reflect memorized labels, not independent clinical validation.** Predictions in general reflect patterns in this specific training dataset, not verified clinical ground truth, and are not medical advice.
- **Scaffold split is stricter than MoleculeNet's official random-split protocol**, so reported metrics are not directly comparable to benchmark numbers using random split — though they still meet or exceed them here.

## References

- Wu, Z., Ramsundar, B., Feinberg, E.N., et al. (2018). MoleculeNet: A benchmark for molecular machine learning. *Chemical Science*, 9(2), 513-530. [https://doi.org/10.1039/C7SC02664A](https://doi.org/10.1039/C7SC02664A)
- Li, X., et al. (2022). Accurate Clinical Toxicity Prediction using Multi-task Deep Neural Nets and Contrastive Molecular Explanations. [https://arxiv.org/abs/2204.06614](https://arxiv.org/abs/2204.06614)
- [AACT (Aggregate Analysis of ClinicalTrials.gov) database](https://aact.ctti-clinicaltrials.org/)
- [SWEETLEAD: a Curated Database of FDA-Approved Drugs](https://simtk.org/projects/sweetlead)
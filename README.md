# Phase VI DOM fate machine learning

This repository contains the reproducible analysis used to classify assigned
FT-ICR MS molecular formulae from paired Phase VI PD/A reactor influent and
effluent samples into three operational DOM fate classes. It trains a random
forest classifier and an L2-regularized multinomial logistic-regression
baseline, evaluates both models once on a held-out test set, and calculates
SHAP values for the random forest.

## Interpretation boundary

The molecular formulae are analytical observations from one paired Phase VI
influent-effluent dataset. They are not independent reactor replicates. The
formula-level train-test split therefore provides internal validation only and
does not measure generalization across reactors, operating periods, or
independent sampling events. Assigned molecular formulae do not establish
molecular structure, relative intensity is not concentration, and the
operational labels do not by themselves prove biodegradation, production, or
chemical persistence.

## Operational DOM fate definitions

Relative intensity (RI) is recalculated independently within each raw sample:

```text
RI = intens / sum(intens)
```

The fold change for formulae detected in both samples is
`FC = RI_effluent / RI_influent`. Formulae are classified as follows:

| Detection and FC rule | DOM fate class |
|---|---|
| Detected only in the influent | Precursor |
| Detected only in the effluent | Product |
| Shared formula with FC < 0.5 | Precursor |
| Shared formula with 0.5 <= FC <= 2.0 | Resistant |
| Shared formula with FC > 2.0 | Product |

Nondetection is not replaced by zero.

## Candidate molecular descriptors

The initial predictor set contains 17 descriptors:

`O/C`, `MW`, `H`, `DBE-O`, `NOSC`, `S/C`, `O`, `S`, `C`, `H/C`,
`DBE`, `AImod`, `N`, `DBE/C`, `N/C`, `P/C`, and `P`.

`MW` is read from the raw `neu.m/z` column. `NOSC` is read from the corrected
source table and is required to agree across shared formulae. Element symbols
denote atom counts in the assigned molecular formula. Derived descriptors are
calculated from the corresponding formula properties.

## Analysis workflow

1. Validate the raw formula tables or a user-supplied processed dataset.
2. Recalculate RI from `intens` within each raw sample.
3. Assign precursor, product, and resistant labels using the rules above.
4. Partition formulae into stratified training and held-out test sets at a
   70:30 ratio using seed `20260920`.
5. Tune both classifiers by fivefold stratified cross-validation within the
   training set, using macro F1-score as the refitting criterion.
6. Fit median imputation, collinearity filtering, and model-specific scaling
   inside each training fold. The random forest uses unscaled predictors; the
   logistic-regression pipeline standardizes retained predictors.
7. Remove descriptor pairs with Pearson `r^2 > 0.80` using the prespecified
   chemical hierarchy and deterministic fallback rules.
8. Evaluate each tuned pipeline once on the held-out test set.
9. Calculate class-specific and overall mean absolute SHAP values for the
   held-out random-forest predictions.

All preprocessing and feature selection are fitted using training data only.
No synthetic oversampling or random undersampling is used. Both classifiers
use balanced class weights.

## Repository structure

```text
.
|-- data/
|   `-- README.md
|-- results/reference/
|-- src/analyze_phaseVI_dom_fate_ml.py
|-- CITATION.cff
|-- LICENSE
|-- README.md
`-- requirements.txt
```

## Installation

Python 3.12.5 was used for the reference analysis.

```bash
python -m venv .venv
```

Activate the environment and install the exact tested dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run with an analysis-ready processed dataset

Formula-level research data are not distributed in this public code
repository. Place an authorized analysis-ready dataset at the path supplied to
`--processed-input`, then run from the repository root:

```bash
python src/analyze_phaseVI_dom_fate_ml.py \
  --processed-input data/processed/phaseVI_dom_fate_dataset.csv \
  --processed-dir results/run/processed \
  --output-dir results/run
```

This command fits the random forest and multinomial logistic-regression
pipelines, evaluates the held-out test set, and calculates SHAP values.

## Run from paired raw workbooks

Raw workbooks are not distributed in this repository. To rebuild the processed
dataset from authorized local copies, run:

```bash
python src/analyze_phaseVI_dom_fate_ml.py \
  --influent /path/to/phaseVI_influent.xlsx \
  --effluent /path/to/phaseVI_effluent.xlsx \
  --processed-dir results/run/processed \
  --output-dir results/run
```

Each workbook must contain a sheet named `result` with these columns:
`formula`, `intens`, `neu.m/z`, `O/C`, `H/C`, `DBE`, `NOSC`, `AI_mod`,
`C`, `H`, `N`, `O`, `P`, `S`, `Cl`, and `Br`.

## Main outputs

The output directory contains:

- `phaseVI_dom_fate_ml_results.xlsx`: consolidated analysis tables;
- `model_comparison.csv`: cross-validation and held-out model metrics;
- `classification_report_*.csv`: class-specific precision, recall, and F1;
- `roc_pr_metrics.csv`: one-versus-rest ROC AUC and average precision;
- `selected_features.csv` and `removed_correlated_features.csv`;
- `cv_fold_feature_selection.csv`: fold-specific feature-selection audit;
- `shap_global_importance.csv` and `shap_class_importance.csv`;
- `shap_values.npz` and `shap_values_long.csv`;
- fitted model files, diagnostic figures, a run manifest, and a generated
  processing report.

The `results/reference` directory contains aggregate metrics,
feature-selection audits, and vector figures from the verified `v1.0.0` run.
Formula-level splits and predictions, large fitted models, raw SHAP arrays, and
the generated workbook are regenerated locally and are not versioned.

## Reference results

The final training-set filter retained 13 descriptors:

`O/C`, `MW`, `H`, `DBE-O`, `NOSC`, `S/C`, `O`, `C`, `H/C`, `DBE`,
`AImod`, `N/C`, and `P/C`.

It removed `S`, `N`, `DBE/C`, and `P`. Numerical reference results are stored
under `results/reference` and summarized in the generated analysis report.

## Data availability in this repository

The raw influent and effluent workbooks and the derived formula-level dataset
are excluded because they are unpublished project data. This public repository
contains the analysis code, environment specification, input schema, aggregate
reference results, and usage documentation. No molecular-formula records,
sample-level intensities, formula-level data partitions, held-out predictions,
credentials, personal data, or instrument acquisition files are included.

The MIT License applies to the software.

## Citation

Please cite the archived release described in `CITATION.cff`. A manuscript
citation can be added after publication.

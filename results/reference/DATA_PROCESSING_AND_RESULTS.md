# Phase VI DOM fate machine-learning analysis

## Data processing

1. The formula-level processed dataset used for the verified analysis was validated before model fitting and evaluation; the dataset is not distributed in this public repository.
2. The processed RI columns were generated from the raw workbooks as `intens / sum(intens)` within each sample. Their validated sums were 1 for the influent and 1 for the effluent.
3. The formula union contained 3,792 formulas: 1,859 shared, 920 influent-only, and 1,013 effluent-only formulas.
4. Shared formulas were classified from FC = RI_effluent / RI_influent. FC < 0.5 was assigned to precursors, 0.5 ≤ FC ≤ 2 to resistant formulas, and FC > 2 to products. Influent-only and effluent-only formulas were assigned to precursors and products, respectively; nondetection was not replaced by zero.
5. 17 prespecified descriptors were constructed. Measured neutral mass (`neu.m/z`) was used as MW, and NOSC was taken directly from the source table after invariant checking across shared formulas. Missing predictor values were imputed using medians fitted within the corresponding training fold (AImod: 1).
6. Molecular formulas were divided into training and held-out test sets at a 70:30 ratio using stratified sampling by DOM fate (seed = 20260920).
7. Within each training fold, descriptors with Pearson r² > 0.8 were filtered before model fitting. Median imputation, filtering, scaling and hyperparameter tuning were fitted using training data only.
8. Random forest and L2-regularized multinomial logistic regression were tuned by fivefold stratified cross-validation using macro-F1. No over- or undersampling was performed.
9. The fitted models were evaluated once on the held-out test set. SHAP values were calculated from the random forest for the held-out formulas.

## DOM fate counts

| DOM_fate | total | influent_only | shared | effluent_only | percentage_of_union |
|---|---|---|---|---|---|
| precursors | 1272 | 920 | 352 | 0 | 33.5443 |
| products | 1161 | 0 | 148 | 1013 | 30.6171 |
| resistant | 1359 | 0 | 1359 | 0 | 35.8386 |

The precursor total comprises 920 influent-only formulas and 352 shared formulas with FC < 0.5. The product total comprises 1,013 effluent-only formulas and 148 shared formulas with FC > 2. The 1,359 resistant formulas were detected in both samples with FC within the stated thresholds.

## Train/test composition

| split | DOM_fate | n |
|---|---|---|
| test | precursors | 382 |
| training | precursors | 890 |
| test | products | 348 |
| training | products | 813 |
| training | resistant | 951 |
| test | resistant | 408 |

## Collinearity filtering

- P was removed and P/C was retained (r² = 0.9794; elemental ratio retained over corresponding atom count).
- DBE/C was removed and H/C was retained (r² = 0.8888; higher mean absolute correlation with the remaining predictors).
- S was removed and S/C was retained (r² = 0.8683; elemental ratio retained over corresponding atom count).
- N was removed and N/C was retained (r² = 0.8680; elemental ratio retained over corresponding atom count).

The final training-set filter removed 4 descriptors (S, N, DBE/C, P) and retained 13 descriptors (O/C, MW, H, DBE-O, NOSC, S/C, O, C, H/C, DBE, AImod, N/C, P/C).

Training-fold selection audit:

- Fold 1: retained 13 descriptors; removed S, N, DBE/C, P.
- Fold 2: retained 13 descriptors; removed S, N, DBE/C, P.
- Fold 3: retained 13 descriptors; removed S, N, DBE/C, P.
- Fold 4: retained 13 descriptors; removed S, N, DBE/C, P.
- Fold 5: retained 13 descriptors; removed S, N, DBE/C, P.

## Model performance

| model | best_CV_macro_F1 | test_balanced_accuracy | test_macro_F1 | test_macro_ROC_AUC | test_macro_average_precision |
|---|---|---|---|---|---|
| random_forest | 0.7192 | 0.7555 | 0.7543 | 0.9009 | 0.8328 |
| logistic_regression | 0.6161 | 0.6046 | 0.6043 | 0.7796 | 0.6233 |

Best random-forest parameters: `{"model__max_depth": null, "model__max_features": "sqrt", "model__min_samples_leaf": 2, "model__n_estimators": 300}`

Best logistic-regression parameters: `{"model__C": 10.0}`

## SHAP ranking

Top descriptors by mean absolute SHAP value: NOSC (0.0933), N/C (0.0582), O/C (0.0461), H (0.0343), AImod (0.0267), C (0.0257), H/C (0.0181), DBE-O (0.0173), MW (0.0170), O (0.0153)

## Interpretation boundary

The analysis quantifies how well formula-derived descriptors discriminate the three DOM fate classes within the single paired Phase VI influent-effluent dataset. Molecular formulas are analytical observations rather than independent reactor replicates. The random train/test split therefore provides internal validation across formulas; it does not establish performance across independent reactors or sampling events. The precursor, product and resistant labels describe detection and relative-intensity patterns and do not by themselves prove biodegradation, production or chemical persistence.

## Reproduction

Run from the project root:

```bash
python src/analyze_phaseVI_dom_fate_ml.py   --processed-input data/processed/phaseVI_dom_fate_dataset.csv   --processed-dir results/run/processed   --output-dir results/run
```

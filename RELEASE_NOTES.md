# v1.0.0

Initial reproducible release of the Phase VI DOM molecular fate classification
workflow.

- validates paired influent and effluent FT-ICR MS formula tables;
- recalculates relative intensity within each sample;
- assigns precursor, product, and resistant operational classes;
- tunes random forest and multinomial logistic-regression pipelines;
- confines preprocessing and feature selection to training data;
- evaluates a stratified held-out test set; and
- calculates random-forest SHAP values and mean absolute feature importance.

The public release includes the analysis code, input schema, and aggregate
reference results. Raw and processed molecular-formula data, formula-level
partitions, and held-out predictions are excluded.

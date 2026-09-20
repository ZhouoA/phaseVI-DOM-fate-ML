# Reference results

This directory contains aggregate metrics, feature-selection audits, and
vector figures from the verified `v1.0.0` run. Formula-level data partitions,
held-out predictions, full fitted models, raw SHAP arrays, and the generated
workbook are excluded and can be regenerated with authorized input data.

The reference run used Python 3.12.5 and the exact package versions in
`requirements.txt`. It completed without warnings under:

```bash
python -W error src/analyze_phaseVI_dom_fate_ml.py \
  --processed-input data/processed/phaseVI_dom_fate_dataset.csv \
  --processed-dir results/reproduction_run/processed \
  --output-dir results/reproduction_run
```

The numerical results were checked against the current project analysis before
release.

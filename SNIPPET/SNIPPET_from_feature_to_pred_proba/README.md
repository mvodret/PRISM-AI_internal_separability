# SNIPPET from feature to pred_proba

Notebook (`snippet_from_feature_to_class.ipynb`) that takes the feature row produced by the previous step and returns
the predicted probability (`pred_proba`) and the final class (GG >= 3 or GG < 3), using an already trained and
serialized classifier. Run the notebook from this folder; the paths below are relative to it.

## Input

The notebook reads `df.csv`, the feature table **written by `../SNIPPET_from_aligned_images_to_features`**
(radiomics row + `psa_density`, with exactly the columns of the classifier). It does not create it.

| Mode | File read |
|---|---|
| Radiologist (manual) segmentation | `../SNIPPET_from_aligned_images_to_features/data/<SUBJ>/features/df.csv` |
| Automatic segmentation | `../SNIPPET_automatic_segmentation/<CASE_ID>/features/df.csv` |

In automatic mode `df.csv` has one row per lesion, and one prediction is printed per row.

## Included files

```
.
├── moe_model.py
├── snippet_from_feature_to_class.ipynb
└── data_report_GT/
    └── moe_classifier_trained.joblib      trained classifier (provided, not produced by any notebook)
```

- `moe_model.py`: implements `ScannerMixtureOfExpertsBootstrap`, a bootstrap Mixture-of-Experts by scanner. It must be
  importable to load the classifier.
- `data_report_GT/moe_classifier_trained.joblib`: bundle containing at least:
  - `model`: ready-to-use classifier (`predict_proba`).
  - `threshold_summary`: threshold summary (`median_threshold_f2_cv` is used in the notebook).
  - `feature_names`: the 9 features and their order (8 radiomic features from T2/ADC + `psa_density`). The
    feature-extraction notebook uses this list to select the columns of `df.csv`.

`data_report_GT/` is not written by any notebook: it contains the trained classifier and must stay where it is, since
the feature-extraction notebook also reads it.

## Output

For each case: the probability in `[0, 1]` of the positive class (GG >= 3), the operating threshold
(`median_threshold_f2_cv`) and the resulting class. Probability and threshold are printed together.

## Requirements

`pandas`, `numpy`, `scikit-learn` (the classifier was saved with 1.7.2; other versions give a warning),
`imbalanced-learn` and `joblib`.

```bash
pip install pandas numpy scikit-learn imbalanced-learn joblib
```

## Notes

- Keep the same feature set and order used in training: the columns of `df.csv` must match `bundle["feature_names"]`.
- The class in `moe_model.py` supports aggregated inference across multiple experts (mean probability).

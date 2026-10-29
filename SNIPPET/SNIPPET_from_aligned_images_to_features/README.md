# From aligned images to features

This folder contains a notebook (`snippet_from_aligned_images_to_features.ipynb`) that extracts radiomic features
(PyRadiomics, 2D largest lesion slice, T2 and ADC) from aligned images and lesion masks, adds the clinical
`psa_density` and writes a one-row feature table (`df.csv`) ready for the classifier. Run the notebook from this
folder; the paths below are relative to it.

## Two ways to provide the input

The notebook has two alternative input blocks. Run only one of them, then continue with the extraction cell.
This folder contains no input data: both blocks read from the sibling folders.
The `*.placeholder` files in those folders stand for the real files (same name without `.placeholder`).

| Block | `TYPE_IMAGE` | Where it reads the case from |
|---|---|---|
| Aligned images from DICOM (cells under "Block1") | `'standard'` | `../SNIPPET_from_dicom_to_aligned_images/data/<SUBJ>/` |
| Automatic segmentation (cells under "TRIAL WITH AUTOMATIC SEGMENTATION") | `'automatic_segmentation'` | `../SNIPPET_automatic_segmentation/<CASE_ID>/` |

The block you run must set `TYPE_IMAGE` accordingly: it decides where the final `radiomics.csv` and `df.csv` are
written (see Output).

## Inputs

**Standard** (manual lesion mask), from `../SNIPPET_from_dicom_to_aligned_images/data/<SUBJ>/` (images and mask are
outputs of that step; `psa_density.csv` must be provided by hand):

```
images/0000.nii.gz                ADC (on the T2 grid)
images/0001.nii.gz                T2
labels/0001.nii.gz                lesion mask
features/clinical/psa_density.csv column `psa_density` (provide it by hand)
```

**Automatic segmentation**, from `../SNIPPET_automatic_segmentation/<CASE_ID>/`:

```
images/<CASE_ID>_0000.nii.gz                    ADC
images/<CASE_ID>_0001.nii.gz                    T2
labels/<CASE_ID>_lesion001_vox<N>.nii.gz, …     one mask per lesion (produced by the segmentation notebook)
features/clinical/psa_density.csv               column `psa_density` (provide it by hand)
```

- The first subfolder found is processed (`subjs_aligned[0]`). In the automatic block `df` has one row per lesion
  mask, and `psa_density` is the first value of the CSV.
- **Channel convention: `0000` = ADC, `0001` = T2.**
- Also required: the trained classifier `../SNIPPET_from_feature_to_pred_proba/data_report_GT/moe_classifier_trained.joblib`
  (and `moe_model.py` from the same folder, needed to unpickle it). It is read only for its `feature_names`, to keep
  the features the classifier expects (radiomics columns + `psa_density`), in the right order. Requires `imbalanced-learn`
  and `scikit-learn` in the environment.

## Output

Everything is created by the notebook:

```
data/<SUBJ>/features/                                            (in this folder)
├── T2_<suffix>.csv, ADC_<suffix>.csv                            radiomic features per modality
├── mask_outlier_stats_T2_<suffix>.csv, …_ADC_<suffix>.csv       clipping / outlier statistics
└── extraction_quality_report.csv                                extraction status for T2 / ADC
radiomics_iterative_save_.../                                    per-subject logs, run_config.json and README_run.md
<case folder>/features/radiomics/radiomics.csv                   final radiomics table
<case folder>/features/df.csv                                    final table for the classifier (see the next point)
```

- `<case folder>` is `data/<SUBJ>` in `standard` mode and `../SNIPPET_automatic_segmentation/<CASE_ID>` in
  `automatic_segmentation` mode.
- **`df.csv` is the input of the next step** (`../SNIPPET_from_feature_to_pred_proba`): the radiomics row plus
  `psa_density`, restricted to the classifier's `feature_names`.
- In `automatic_segmentation` mode the intermediate results (`data/<SUBJ>/features/`) are still written in this folder.
- `data/` and `radiomics_iterative_save_.../` are not in the repository: the notebook creates them, and it overwrites an
  existing subject (`FORCE_OVERWRITE_SUBJECT = True`).

## Requirements

`pyradiomics` (`from radiomics import featureextractor`), `SimpleITK`, `nibabel`, `numpy`, `pandas` and `matplotlib`.
All of them are included in the `tf-gpu-env` environment (`../template_folder/tf-gpu-env.yml`), so the same environment can be used (no GPU is needed for this step).

## Notes

- The two input blocks reuse the same variable names (`subj`, `df`, `path`): do not run both in the same session
  without re-running the extraction cell.
- In the standard block, `np.sort(os.listdir(...))[1:]` drops the first entry (meant for `.DS_Store`): check it
  before running on another machine.
- The notebook has not been verified end to end after the repository reorganization; see the paths above if a
  folder is not found.

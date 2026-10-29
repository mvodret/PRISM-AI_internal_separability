# From DICOM to aligned images

This folder contains a notebook that converts each subject's T2 and ADC DICOM series to NIfTI and aligns the ADC
and lesion mask to the T2 image grid. Run the notebook from this folder; the paths below are relative to it.

## Inputs

```
raw_data/<SUBJ>/
├── T2/              DICOM series of the T2-weighted axial scan (required)
├── ADC/             DICOM series of the ADC map (required)
└── t2_mask.nii.gz   manual lesion mask drawn on the T2 (required for the complete aligned output)
```

- `<SUBJ>` is the subject folder name (for example, `PRISM 001`).
- **The repository contains no data.** `ADC/` and `T2/` hold only a `.gitkeep`: put the DICOM files there.
  `t2_mask.nii.gz.placeholder` stands for the mask: provide the real file named `t2_mask.nii.gz` (without the
  `.placeholder` suffix).
- `T2/` and `ADC/` must contain the corresponding DICOM series. If a folder contains multiple DICOM series, the
  notebook selects the one with the most files.
- `t2_mask.nii.gz` must be a NIfTI mask corresponding to the subject's T2 image, with the same volume shape. The
  notebook uses it to create `labels/0001.nii.gz` and to run the final resampling step. Without the mask, the
  DICOM-to-NIfTI conversions can run, but the final step skips the subject: the ADC output is not aligned to T2.

The notebook selects subjects from `raw_data/`; its example cell then narrows the selection to the first selected
subject. **Selection caveat:** the notebook currently drops the first sorted entry of `raw_data/` (using
`np.sort(os.listdir('raw_data'))[1:]`), apparently to skip `.DS_Store`. If `.DS_Store` is absent, this also skips the
first real subject. Check the `SUBJS` selection cell before running on another machine or dataset.

## Output

```
data/<SUBJ>/
├── images/
│   ├── 0000.nii.gz   ADC, resampled on the T2 grid
│   └── 0001.nii.gz   T2 converted from DICOM
├── labels/
│   └── 0001.nii.gz   lesion mask on the T2 grid; created only when t2_mask.nii.gz is provided
└── features/
    └── clinical/
        └── psa_density.csv   separate clinical input for downstream feature extraction; not generated here
```

The image channel convention is **`0000` = ADC, `0001` = T2** (matching the nnU-Net training configuration).
The label is named `0001` because it is defined on the T2 grid. `images/` and `labels/` are generated or overwritten
by the notebook; `features/clinical/psa_density.csv` is not. The notebook also displays quality-control plots.

## Clinical PSA data: not used here

`data/<SUBJ>/features/clinical/psa_density.csv` is **not read by this notebook** and is not created by it. It is
read by the later feature-extraction step, which uses its `psa_density` column. It is not needed to convert the
DICOMs; provide it when running that downstream step.

`data/` contains only `<SUBJ>/features/clinical/psa_density.csv.placeholder`, for the later standard feature-extraction step (provide the real `psa_density.csv`). The notebook creates `images/` and `labels/` there.

## Next steps

- **Radiomics / feature extraction:** `../SNIPPET_from_aligned_images_to_features/` reads the aligned images and
  labels from `data/<SUBJ>/` and uses the clinical `psa_density.csv`.
- **Automatic segmentation:** this notebook's output is not yet in the nnU-Net bundle layout. The images must first
  be copied/renamed to `SNIPPET_automatic_segmentation/<case>/images/<case>_0000.nii.gz` (ADC) and
  `<case>_0001.nii.gz` (T2), as described in that folder's documentation.

## Requirements

`SimpleITK`, `nibabel`, `numpy`, `antspyx` (`import ants`) and `matplotlib`.

## Notes

- No cropping, intensity normalization or bias-field correction is applied.

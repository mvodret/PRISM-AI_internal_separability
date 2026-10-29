# Prostate MRI: from DICOM to lesion classification

A set of Jupyter notebooks that take raw prostate MRI (T2 and ADC DICOM series) to a per-lesion prediction (probability and class), using radiomic features and a trained classifier.

Lesions can be provided in two ways:

- **Radiologist mask**: the lesion is segmented manually (`t2_mask.nii.gz`).
- **Automatic segmentation**: the lesion is segmented by a trained nnU-Net model.

## Roadmap

```
                 ┌──────────────────────────────────────────┐
                 │ 1. SNIPPET_from_dicom_to_aligned_images  │
                 │    DICOM + mask -> aligned NIfTI images  │
                 └───────────────┬──────────────────────────┘
                                 │
              radiologist mask   │   automatic segmentation
          ┌──────────────────────┴───────────────────────┐
          │                                              ▼
          │                         ┌──────────────────────────────────┐
          │                         │ 2. SNIPPET_automatic_segmentation│
          │                         │    images -> predicted lesion    │
          │                         └───────────────┬──────────────────┘
          ▼                                         ▼
 ┌────────────────────────────────────────────────────────────┐
 │ 3. SNIPPET_from_aligned_images_to_features                 │
 │    images + lesion mask -> radiomic features (df.csv)      │
 └─────────────────────────────┬──────────────────────────────┘
                               ▼
 ┌────────────────────────────────────────────────────────────┐
 │ 4. SNIPPET_from_feature_to_pred_proba                      │
 │    features -> probability and class                       │
 └────────────────────────────────────────────────────────────┘
```

| Step | Folder | Input | Output |
|------|--------|-------|--------|
| 1 | [SNIPPET_from_dicom_to_aligned_images](SNIPPET_from_dicom_to_aligned_images) | `raw_data/<SUBJ>/{T2,ADC,t2_mask.nii.gz}` | `data/<SUBJ>/{images,labels,features}` |
| 2 (optional) | [SNIPPET_automatic_segmentation](SNIPPET_automatic_segmentation) | `<CASE_ID>/images/` (T2 and ADC NIfTI) | predicted lesion mask for `<CASE_ID>` |
| 3 | [SNIPPET_from_aligned_images_to_features](SNIPPET_from_aligned_images_to_features) | step 1 output (`standard`) or step 2 output (`automatic_segmentation`) | `df.csv` with radiomic features |
| 4 | [SNIPPET_from_feature_to_pred_proba](SNIPPET_from_feature_to_pred_proba) | `df.csv` and `psa_density.csv` | probability and class per lesion |

### Path A: radiologist-drawn lesions

1 -> 3 (`standard` block) -> 4

### Path B: automatically segmented lesions

1 -> 2 -> 3 (`automatic_segmentation` block) -> 4

In path B the segmentation of step 2 replaces the radiologist mask. Step 1 is still needed to obtain aligned T2 and ADC images.

Each folder has its own README with the exact inputs, outputs and notebook cells to run.

## Repository layout

```
.
├── template_folder/                        # trained nnU-Net segmenter (used by step 2)
├── SNIPPET_from_dicom_to_aligned_images/   # step 1
├── SNIPPET_automatic_segmentation/         # step 2 (optional)
├── SNIPPET_from_aligned_images_to_features/# step 3
└── SNIPPET_from_feature_to_pred_proba/     # step 4 (includes the trained classifier)
```

The notebooks use **relative paths between sibling folders** (for example `../SNIPPET_automatic_segmentation` or `../../template_folder`). Do not rename or move the folders.

## Trained models

Both models are included:

- **Segmenter**: nnU-Net v2, in [template_folder](template_folder). The checkpoint (`checkpoint_best_global.pth`, about 234 MB) is stored with [Git LFS](https://git-lfs.com).
- **Classifier**: scanner mixture-of-experts, in [SNIPPET_from_feature_to_pred_proba/data_report_GT](SNIPPET_from_feature_to_pred_proba/data_report_GT). It also defines the 9 features used in step 3.

## Data

No patient data is included. Input files are replaced by empty `*.placeholder` files that show the expected names and locations. Remove the `.placeholder` suffix and put your own file there (for example `t2_mask.nii.gz.placeholder` -> `t2_mask.nii.gz`).

`psa_density.csv` is always provided manually: no notebook creates it.

## Environment (install this first)

Before running any notebook, create the conda environment defined in [template_folder/tf-gpu-env.yml](template_folder/tf-gpu-env.yml). It was exported from **Windows** and is meant to be used there:

- OS: Windows
- Python 3.10.14
- nnunetv2 2.6.4
- PyTorch 2.5.1 + CUDA 12.1 (`+cu121` wheels)
- pyradiomics, SimpleITK, nibabel, pandas, scikit-learn, imbalanced-learn

```bash
conda env create -f template_folder/tf-gpu-env.yml
conda activate tf-gpu-env
```

Step 2 (automatic segmentation) requires an NVIDIA GPU with CUDA support. On Linux or macOS the file may need adjustments (the `+cu121` wheels are CUDA-specific). Install [Git LFS](https://git-lfs.com) before cloning (`git lfs install`) so the checkpoint is downloaded.

## Notes

- The notebooks were edited after their last run. Re-run them in order before relying on their saved outputs.
- Some saved outputs contain local paths from the machine where they were run.

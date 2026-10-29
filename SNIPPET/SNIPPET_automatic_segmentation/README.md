# `SNIPPET_automatic_segmentation`

Example of automatic lesion segmentation with the trained nnU-Net model contained in
[`../template_folder`](../template_folder). Each subfolder is a **bundle**: one case with its images and the
notebook that runs the prediction (`run_predict_bundle.ipynb`, CUDA GPU). The included example is `ext0000/`;
copy its notebook to create new bundles.

## ⚠️ Do not change the folder structure

The notebook **does not contain the model**: it looks for it in `template_folder`, using the relative path
`<bundle>/../../template_folder/` (folders `nnUNet_results/` and `custom_trainers/`). If a folder is moved or
renamed, the notebook stops with `FileNotFoundError`. The layout must stay as follows:

```
<project>/
├── template_folder/            model, custom trainer, metadata, tf-gpu-env.yml
└── SNIPPET_automatic_segmentation/
    └── <CASE_ID>/              case bundle (e.g. ext0000), exactly two levels below <project>
```

- The folder must be named `template_folder` and sit next to `SNIPPET_automatic_segmentation`.
- Each bundle must be placed directly inside `SNIPPET_automatic_segmentation/`, not in deeper subfolders.
- Do not move or rename `nnUNet_results/` and `custom_trainers/` inside `template_folder`.

For the environment (Windows, NVIDIA GPU recommended) see the `template_folder` README.

## Input

```
SNIPPET_automatic_segmentation/<CASE_ID>/
├── images/
│   ├── <CASE_ID>_0000.nii.gz   ADC
│   └── <CASE_ID>_0001.nii.gz   T2
├── features/
│   └── clinical/
│       └── psa_density.csv     clinical input for the next step (not used by the segmentation)
└── run_predict_bundle.ipynb    prediction notebook (CUDA)
```

This is the **minimal bundle for the segmentation**: only `images/` and the notebook are needed. `features/clinical/psa_density.csv`
is an input of the following step (`../SNIPPET_from_aligned_images_to_features`), which reads it from the bundle:
place it there by hand.

- **The repository contains no data.** The `*.placeholder` files stand for the real files: provide them with the
  same name without the `.placeholder` suffix (e.g. `ext0000_0000.nii.gz`, `psa_density.csv`).
- **Channel order: `0000` = ADC, `0001` = T2** (as in the trained model).
- The two images must share the same grid/dimensions: they are the output of
  `../SNIPPET_from_dicom_to_aligned_images` (`images/0000` and `images/0001`), renamed with the `<CASE_ID>_` prefix.
- `images/` must contain **a single case**, unless `CASE_ID` is set in the notebook.
- Optional, only for evaluation (Dice, etc.): `labels/<CASE_ID>.nii.gz` (reference mask) and `metadata/` (copy
  `dataset.json` and `nnUNetPlans.json` from `template_folder/metadata/`). Without these files the evaluation is
  skipped and the prediction still works.

## How to run

1. Create the `<CASE_ID>/` folder with `images/` and copy `run_predict_bundle.ipynb` from `ext0000/` into it.
2. Open the notebook with the `tf-gpu-env` kernel and run the cells. If the bundle is not detected automatically,
   set `USER_BUNDLE_DIR` at the top of the notebook.
3. The notebook installs the custom trainer into the `nnunetv2` package, runs `nnUNetv2_predict`, splits the
   prediction into distinct lesions and shows control plots.

## Output

Everything is created by the notebook:

```
<CASE_ID>/
├── preds_3d_fullres_nnUNetTrainerBS4_CC_VOLAWARE_fold_all/
│   ├── <CASE_ID>.nii.gz        predicted segmentation (binary: lesion > 0)
│   ├── summary.json            only if the evaluation was run
│   └── dataset.json, plans.json, predict_from_raw_data_args.json   (written by nnU-Net)
├── labels/
│   └── <CASE_ID>_lesion001_vox<N>.nii.gz, …   one mask per connected component
│                                              (26-connectivity, minimum 40 voxels, sorted by volume)
└── _runtime/                   nnU-Net working folders (created by the notebook)
```

Note: the single-lesion masks are written to `labels/`, the same folder where the optional reference mask for
the evaluation must be placed.

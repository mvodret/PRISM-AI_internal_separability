# `template_folder`

This folder contains the **trained nnU-Net segmentation model** for lesions, used for automatic segmentation.
It contains no notebooks: the prediction notebook and a complete usage example are in
[`../SNIPPET_automatic_segmentation`](../SNIPPET_automatic_segmentation).

## Contents

- `nnUNet_results/`: trained model (`Dataset100_lesion`, trainer `nnUNetTrainerBS4_CC_VOLAWARE`, configuration
  `3d_fullres`), with the configuration and checkpoint (`fold_all/checkpoint_best_global.pth`) used for prediction.
- `custom_trainers/`: code of the custom trainer required by the model; the notebook installs it into `nnunetv2`.
- `metadata/`: `dataset.json` and `nnUNetPlans.json`. **The notebook does not read them from here**: for the
  (optional) evaluation, which also needs a reference mask, copy them to
  `SNIPPET_automatic_segmentation/<CASE_ID>/metadata/`.
- `tf-gpu-env.yml`: Conda environment definition.

Model input channels: **`0000` = ADC, `0001` = T2**.

## ⚠️ Do not move or rename this folder

The notebook in `SNIPPET_automatic_segmentation/<CASE_ID>/` looks for the model using the relative paths
`../../template_folder/nnUNet_results` and `../../template_folder/custom_trainers`. The folder must therefore be
named `template_folder` and sit next to `SNIPPET_automatic_segmentation`; `nnUNet_results/` and `custom_trainers/` must
not be moved or renamed either.

## Setting up the environment

The recommended environment is the one defined in `tf-gpu-env.yml`: **Windows**, Python 3.10, nnU-Net v2 (2.6.4)
and PyTorch 2.5.1 with CUDA 12.1. With Conda installed, from the `template_folder` directory:

```bash
conda env create -f tf-gpu-env.yml
conda activate tf-gpu-env
```

CUDA requires an NVIDIA GPU with drivers compatible with CUDA 12.1. On other operating systems the
Windows-specific packages (e.g. `pywin32`, `vc`, `tensorflow-intel`) may fail to install. In VS Code, select the
`tf-gpu-env` interpreter as the notebook kernel.

## Running a prediction

See the README in [`../SNIPPET_automatic_segmentation`](../SNIPPET_automatic_segmentation): it explains how to prepare a bundle
(`images/<CASE_ID>_0000.nii.gz` = ADC and `<CASE_ID>_0001.nii.gz` = T2), how to run `run_predict_bundle.ipynb`
(CUDA) and which files are produced. The `ext0000/` example contains the notebook, ready to copy.

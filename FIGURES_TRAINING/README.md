
# Figure documentation and model training

This folder contains documentation notebooks for the figures in the paper,
along with the training code for the classifier and the nnU-Net segmenter.

## Figure notebooks

- `FIG1_generation.ipynb` and `FIG2_generation.ipynb` document the code and
  outputs used to generate Figures 1 and 2.
- `FIG3_generation.ipynb` documents the Figure 3 analysis, including feature
  selection, classifier training, evaluation, and figure generation. The
  trained classifier is exported as a `joblib` bundle.
- `FIG4_generation.ipynb` documents the Figure 4 internal univariate
  separability procedure.

The notebooks retain outputs from the original analyses. See the accompanying
README files for the detailed methods and input requirements for Figures 3
and 4.

## nnU-Net segmenter training

[`trainer_nnunet/`](trainer_nnunet/) contains the custom nnU-Net trainer and
shell scripts used to train and evaluate the segmentation model. Its README
describes the expected data layout, environment, and execution details.

The datasets, trained checkpoints, prediction files, and generated figure
outputs are not included here.

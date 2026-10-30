
# Figure documentation and model training

This folder contains documentation notebooks for the figures in the paper,
along with the training code for the classifier and the nnU-Net segmenter.

## Figure notebooks

- [`fig1/FIG1_generation.ipynb`](fig1/FIG1_generation.ipynb) and
  [`fig2/FIG2_generation.ipynb`](fig2/FIG2_generation.ipynb) document the code
  and outputs used to generate Figures 1 and 2.
- [`fig3_training_classifier/FIG3_generation.ipynb`](fig3_training_classifier/FIG3_generation.ipynb)
  documents the Figure 3 analysis, including feature selection, classifier
  training, evaluation, and figure generation. The trained classifier is
  exported as a `joblib` bundle. See its
  [README](fig3_training_classifier/README_feature_selection_classification_FIG3.md)
  for the detailed method and inputs.
- [`fig4_internal_separability/FIG4_generation.ipynb`](fig4_internal_separability/FIG4_generation.ipynb)
  documents the Figure 4 internal univariate-separability procedure. See its
  [README](fig4_internal_separability/README_internal_separability_FIG4.md)
  for the detailed method and inputs.

## nnU-Net segmenter training

[`trainer_nnunet/`](trainer_nnunet/) contains the custom nnU-Net trainer and
shell scripts used to train and evaluate the segmentation model. Its README
describes the expected data layout, environment, and execution details.

The notebooks retain outputs from their original runs. The source datasets,
prediction files, trained model files, and standalone generated figure files
are not included in this folder.

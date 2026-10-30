# PRISM-AI internal separability

Code accompanying the paper on the internal separability of prostate lesions with radiomics and AI.

## Contents

| Folder | Description |
|--------|-------------|
| [SNIPPET](SNIPPET) | Pipeline from raw DICOM to per-lesion probability and class, with the trained segmenter and classifier. Start from its [README](SNIPPET/README.md). |
| [FIGURES_TRAINING](FIGURES_TRAINING) | Documentation notebooks for Figures 1–4, classifier-training documentation, univariate-separability procedure, and nnU-Net segmenter-training code. Start from its [README](FIGURES_TRAINING/README.md). |

## Figure documentation and training

The notebooks in `FIGURES_TRAINING` document how the paper figures are
produced; they are not the inference notebooks in `SNIPPET`.

- [`fig1/FIG1_generation.ipynb`](FIGURES_TRAINING/fig1/FIG1_generation.ipynb)
  and [`fig2/FIG2_generation.ipynb`](FIGURES_TRAINING/fig2/FIG2_generation.ipynb)
  document the generation of Figures 1 and 2.
- [`fig3_training_classifier/FIG3_generation.ipynb`](FIGURES_TRAINING/fig3_training_classifier/FIG3_generation.ipynb)
  documents the feature-selection and classifier training/evaluation workflow,
  including export of the trained classifier bundle.
- [`fig4_internal_separability/FIG4_generation.ipynb`](FIGURES_TRAINING/fig4_internal_separability/FIG4_generation.ipynb)
  documents the internal univariate-separability analysis.
- [`FIGURES_TRAINING/trainer_nnunet/`](FIGURES_TRAINING/trainer_nnunet/)
  contains the custom nnU-Net trainer and scripts for training and evaluating
  the segmentation model.

The `FIGURES_TRAINING` README links to the detailed FIG3 and FIG4 method and
input documentation. The notebooks retain outputs from their original runs;
the source datasets, prediction files, and standalone generated figure files
are not included.

## Git LFS

The segmenter checkpoint (`*.pth`, about 234 MB) is stored with [Git LFS](https://git-lfs.com). **Git LFS is required to download the full repository**: without it you only get a small pointer file instead of the model. Install it before cloning:

```bash
git lfs install
git clone <repo-url>
```

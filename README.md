# PRISM-AI internal separability

Code accompanying the paper on the internal separability of prostate lesions with radiomics and AI.

## Contents

| Folder | Description |
|--------|-------------|
| [SNIPPET](SNIPPET) | Pipeline from raw DICOM to per-lesion probability and class, with the trained segmenter and classifier. Start from its [README](SNIPPET/README.md). |
| [RESULTS](RESULTS) | Notebooks reproducing the figures of the paper (to be added). |

## Git LFS

The segmenter checkpoint (`*.pth`, about 234 MB) is stored with [Git LFS](https://git-lfs.com). **Git LFS is required to download the full repository**: without it you only get a small pointer file instead of the model. Install it before cloning:

```bash
git lfs install
git clone <repo-url>
```

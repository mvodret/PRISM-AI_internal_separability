# nnU-Net Training - Dataset 100

This folder documents the code used to train the nnU-Net segmentation model on
Dataset 100. It is intended for documentation and experiment traceability,
**not as an installable Python package**. It does not include the dataset,
checkpoints, or predictions.

## Files

- `nnUNetTrainerBS4_CC_VOLAWARE.py`: a custom trainer derived from nnU-Net's
  `nnUNetTrainer`. It sets `batch_size = 8`, uses a loss combining
  cross-entropy and Tversky loss, and computes `global_fg_dice` from aggregated
  validation TP, FP, and FN values. When the score improves, it saves
  `checkpoint_best_global.pth` and copies it to `checkpoint_final.pth`. It
  also implements early stopping (25-epoch warmup, patience of 25, and
  `min_delta = 1e-3`).
- `run_pipeline_100_all_train_new.sh`: trains using the `all` fold, then runs
  prediction and evaluation on the external cases. Prediction explicitly uses
  `checkpoint_best_global.pth`.
- `run_pipeline_100_new.sh`: trains folds `0` through `4`, predicts and
  evaluates the external cases for each fold, and then runs prediction and
  evaluation using the ensemble of all five folds.

The trainer class name contains `BS4`, but the batch size actually set by the
code is 8.

## Paths expected by the scripts

Both scripts set `PROJECT_DIR="$HOME/PrismaAI"` and expect the data and results
to already be available in this layout:

```text
$HOME/PrismaAI/
├── nnUNet_raw/
│   └── Dataset100_lesion/
│       └── dataset.json
├── nnUNet_preprocessed/
│   └── Dataset100_lesion/
│       └── nnUNetPlans.json
├── nnUNet_results/
└── external_val_leftout_100/
    ├── images/
    └── labels/
```

The raw and preprocessed datasets are used for training. The external images
and labels are used for prediction and evaluation. The scripts do not download
or prepare the data. Checkpoints and other training results are written under
`nnUNet_results`; external predictions and evaluation summaries are written
under `external_val_leftout_100`.

## Environment and execution

The scripts assume an environment that is already configured with nnU-Net v2,
the custom trainer available as
`nnunetv2.training.nnUNetTrainer.nnUNetTrainerBS4_CC_VOLAWARE`, and the commands
`nnUNetv2_train`, `nnUNetv2_predict`, and `nnUNetv2_evaluate_folder` available
on `PATH`. The trainer file in this folder is the source code for documenting
the implementation; the scripts **do not install or copy it** into the nnU-Net
package. The runtime environment must be configured separately so that the
import works.

The scripts check that required directories and files exist. They may skip
steps when checkpoints, complete predictions, or `summary.json` files already
exist. In particular, the `all`-fold script skips training if it finds either
`checkpoint_final.pth` or `checkpoint_best_global.pth` in the `fold_all`
output directory. Check these files before rerunning the script.

## Data and repository contents

The `$HOME/PrismaAI` path is machine-specific and may need to be adapted for a
different environment. 

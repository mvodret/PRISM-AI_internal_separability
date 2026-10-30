#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$HOME/PrismaAI"

export nnUNet_raw="$PROJECT_DIR/nnUNet_raw"
export nnUNet_preprocessed="$PROJECT_DIR/nnUNet_preprocessed"
export nnUNet_results="$PROJECT_DIR/nnUNet_results"

DATASET_ID=100
DATASET_NAME="Dataset100_lesion"
CONFIG="3d_fullres"
TRAINER="nnUNetTrainerBS4_CC_VOLAWARE"

EXT_BASE="$PROJECT_DIR/external_val_leftout_100"
EXT_IN="$EXT_BASE/images"
EXT_GT="$EXT_BASE/labels"

EXT_PRED_BASE="$EXT_BASE/preds_${CONFIG}_${TRAINER}"
OUT_ALL="$EXT_PRED_BASE/fold_all"
SUMMARY_ALL="$OUT_ALL/summary.json"

DATASET_JSON="$nnUNet_raw/$DATASET_NAME/dataset.json"
PLANS_JSON="$nnUNet_preprocessed/$DATASET_NAME/nnUNetPlans.json"

MODEL_BASE="$nnUNet_results/$DATASET_NAME/${TRAINER}__nnUNetPlans__${CONFIG}"
FOLD_ALL_DIR="$MODEL_BASE/fold_all"

log_section() {
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
    date
}

count_cases_from_images() {
    local folder="$1"
    find "$folder" -maxdepth 1 -type f -name "*.nii.gz" \
        | sed 's#.*/##' \
        | sed 's/_....\.nii\.gz$//' \
        | sort -u \
        | wc -l
}

count_cases_from_predictions() {
    local folder="$1"
    find "$folder" -maxdepth 1 -type f -name "*.nii.gz" \
        | sed 's#.*/##' \
        | sed 's/\.nii\.gz$//' \
        | sort -u \
        | wc -l
}

have_predictions_complete() {
    local in_folder="$1"
    local out_folder="$2"

    local n_in
    local n_out

    n_in=$(count_cases_from_images "$in_folder")
    n_out=$(count_cases_from_predictions "$out_folder")

    [ "$n_in" -gt 0 ] && [ "$n_out" -eq "$n_in" ]
}

log_section "PIPELINE START - TRAIN ALL"

echo "nnUNet_raw          = $nnUNet_raw"
echo "nnUNet_preprocessed = $nnUNet_preprocessed"
echo "nnUNet_results      = $nnUNet_results"
echo "EXT_IN              = $EXT_IN"
echo "EXT_GT              = $EXT_GT"
echo "OUT_ALL             = $OUT_ALL"
echo "FOLD_ALL_DIR        = $FOLD_ALL_DIR"

[ -d "$nnUNet_raw/$DATASET_NAME" ] || { echo "Missing raw dataset"; exit 1; }
[ -d "$nnUNet_preprocessed/$DATASET_NAME" ] || { echo "Missing preprocessed dataset"; exit 1; }
[ -d "$EXT_IN" ] || { echo "Missing external input folder"; exit 1; }
[ -d "$EXT_GT" ] || { echo "Missing external GT folder"; exit 1; }
[ -f "$DATASET_JSON" ] || { echo "Missing dataset.json"; exit 1; }
[ -f "$PLANS_JSON" ] || { echo "Missing nnUNetPlans.json"; exit 1; }

mkdir -p "$OUT_ALL"

echo
echo "=== TRAINER IMPORT TEST ==="
python -c "from nnunetv2.training.nnUNetTrainer.${TRAINER} import ${TRAINER}; print('trainer import ok')"

echo
echo "=== GPU INFO ==="
nvidia-smi || true

echo
echo "N external cases: $(count_cases_from_images "$EXT_IN")"

# =========================
# TRAIN ALL
# =========================
if [ -f "$FOLD_ALL_DIR/checkpoint_final.pth" ] || [ -f "$FOLD_ALL_DIR/checkpoint_best_global.pth" ]; then
    log_section "SKIP TRAINING FOLD ALL (checkpoint already exists)"
else
    log_section "TRAINING FOLD ALL"
    nnUNetv2_train "$DATASET_ID" "$CONFIG" all -tr "$TRAINER"
fi

# =========================
# EXTERNAL PREDICTION - ALL
# =========================
if have_predictions_complete "$EXT_IN" "$OUT_ALL"; then
    log_section "SKIP EXTERNAL PREDICTION - FOLD ALL (complete predictions already exist)"
    echo "Pred case count in $OUT_ALL: $(count_cases_from_predictions "$OUT_ALL")"
else
    log_section "EXTERNAL PREDICTION - FOLD ALL"
    echo "Pred case count before run in $OUT_ALL: $(count_cases_from_predictions "$OUT_ALL")"

    nnUNetv2_predict \
        -i "$EXT_IN" \
        -o "$OUT_ALL" \
        -d "$DATASET_ID" \
        -c "$CONFIG" \
        -tr "$TRAINER" \
        -f all \
        -chk checkpoint_best_global.pth

    echo "Pred case count after run in $OUT_ALL: $(count_cases_from_predictions "$OUT_ALL")"
fi

# =========================
# EXTERNAL EVALUATION - ALL
# =========================
if [ -f "$SUMMARY_ALL" ]; then
    log_section "SKIP EXTERNAL EVALUATION - FOLD ALL (summary.json already exists)"
    cat "$SUMMARY_ALL"
else
    log_section "EXTERNAL EVALUATION - FOLD ALL"

    nnUNetv2_evaluate_folder \
        "$EXT_GT" \
        "$OUT_ALL" \
        -djfile "$DATASET_JSON" \
        -pfile "$PLANS_JSON" \
        -o "$SUMMARY_ALL"

    if [ -f "$SUMMARY_ALL" ]; then
        echo "--- summary.json for fold_all ---"
        cat "$SUMMARY_ALL"
    else
        echo "WARNING: summary.json was not created"
    fi
fi

log_section "PIPELINE COMPLETED"
echo "Fold-all preds/eval in: $OUT_ALL"

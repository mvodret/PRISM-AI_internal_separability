#!/usr/bin/env bash
set -euo pipefail

# =========================
# USER SETTINGS
# =========================
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
DATASET_JSON="$nnUNet_raw/$DATASET_NAME/dataset.json"
PLANS_JSON="$nnUNet_preprocessed/$DATASET_NAME/nnUNetPlans.json"

FOLDS=(0 1 2 3 4)

MODEL_BASE="$nnUNet_results/$DATASET_NAME/${TRAINER}__nnUNetPlans__${CONFIG}"

# =========================
# HELPERS
# =========================
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

# =========================
# CHECKS
# =========================
log_section "PIPELINE START"

echo "=== PATHS ==="
echo "nnUNet_raw          = $nnUNet_raw"
echo "nnUNet_preprocessed = $nnUNet_preprocessed"
echo "nnUNet_results      = $nnUNet_results"
echo "DATASET_JSON        = $DATASET_JSON"
echo "PLANS_JSON          = $PLANS_JSON"
echo "EXT_IN              = $EXT_IN"
echo "EXT_GT              = $EXT_GT"
echo "EXT_PRED_BASE       = $EXT_PRED_BASE"
echo "MODEL_BASE          = $MODEL_BASE"
echo

[ -d "$nnUNet_raw/$DATASET_NAME" ] || { echo "Missing raw dataset: $nnUNet_raw/$DATASET_NAME"; exit 1; }
[ -d "$nnUNet_preprocessed/$DATASET_NAME" ] || { echo "Missing preprocessed dataset: $nnUNet_preprocessed/$DATASET_NAME"; exit 1; }
[ -d "$EXT_IN" ] || { echo "Missing external input folder: $EXT_IN"; exit 1; }
[ -d "$EXT_GT" ] || { echo "Missing external GT folder: $EXT_GT"; exit 1; }
[ -f "$DATASET_JSON" ] || { echo "Missing dataset.json: $DATASET_JSON"; exit 1; }
[ -f "$PLANS_JSON" ] || { echo "Missing nnUNetPlans.json: $PLANS_JSON"; exit 1; }

mkdir -p "$EXT_PRED_BASE"

echo "=== TRAINER IMPORT TEST ==="
python -c "from nnunetv2.training.nnUNetTrainer.${TRAINER} import ${TRAINER}; print('trainer import ok')"
echo

echo "=== GPU INFO ==="
nvidia-smi || true
echo

echo "=== EXTERNAL INPUT CASE COUNT ==="
echo "N external cases: $(count_cases_from_images "$EXT_IN")"
echo

# =========================
# TRAIN + EXTERNAL PRED + EVAL BY FOLD
# =========================
for FOLD in "${FOLDS[@]}"; do
    FOLD_DIR="$MODEL_BASE/fold_${FOLD}"
    OUT_FOLD="$EXT_PRED_BASE/fold_${FOLD}"
    SUMMARY_FOLD="$OUT_FOLD/summary.json"

    mkdir -p "$OUT_FOLD"

    # -------------------------
    # TRAIN
    # -------------------------
    if [ -f "$FOLD_DIR/checkpoint_final.pth" ]; then
        log_section "SKIP TRAINING FOLD $FOLD (checkpoint_final.pth already exists)"
    else
        log_section "TRAINING FOLD $FOLD"
        nnUNetv2_train "$DATASET_ID" "$CONFIG" "$FOLD" -tr "$TRAINER"
    fi

    # -------------------------
    # EXTERNAL PREDICTION
    # -------------------------
    if have_predictions_complete "$EXT_IN" "$OUT_FOLD"; then
        log_section "SKIP EXTERNAL PREDICTION - SINGLE FOLD $FOLD (complete predictions already exist)"
        echo "Pred case count in $OUT_FOLD: $(count_cases_from_predictions "$OUT_FOLD")"
    else
        log_section "EXTERNAL PREDICTION - SINGLE FOLD $FOLD"
        echo "Pred case count before run in $OUT_FOLD: $(count_cases_from_predictions "$OUT_FOLD")"
        nnUNetv2_predict \
            -i "$EXT_IN" \
            -o "$OUT_FOLD" \
            -d "$DATASET_ID" \
            -c "$CONFIG" \
            -tr "$TRAINER" \
            -f "$FOLD"
        echo "Pred case count after run in $OUT_FOLD: $(count_cases_from_predictions "$OUT_FOLD")"
    fi

    # -------------------------
    # EXTERNAL EVALUATION
    # -------------------------
    if [ -f "$SUMMARY_FOLD" ]; then
        log_section "SKIP EXTERNAL EVALUATION - SINGLE FOLD $FOLD (summary.json already exists)"
        echo "--- summary.json for fold $FOLD ---"
        cat "$SUMMARY_FOLD"
        echo
    else
        log_section "EXTERNAL EVALUATION - SINGLE FOLD $FOLD"
        nnUNetv2_evaluate_folder \
            "$EXT_GT" \
            "$OUT_FOLD" \
            -djfile "$DATASET_JSON" \
            -pfile "$PLANS_JSON" \
            -o "$SUMMARY_FOLD"

        if [ -f "$SUMMARY_FOLD" ]; then
            echo "--- summary.json for fold $FOLD ---"
            cat "$SUMMARY_FOLD"
            echo
        else
            echo "WARNING: summary.json was not created for fold $FOLD"
        fi
    fi
done

# =========================
# ENSEMBLE PRED + EVAL
# =========================
OUT_ENSEMBLE="$EXT_PRED_BASE/ensemble_0_1_2_3_4"
SUMMARY_ENSEMBLE="$OUT_ENSEMBLE/summary.json"
mkdir -p "$OUT_ENSEMBLE"

if have_predictions_complete "$EXT_IN" "$OUT_ENSEMBLE"; then
    log_section "SKIP EXTERNAL PREDICTION - ENSEMBLE (complete predictions already exist)"
    echo "Pred case count in $OUT_ENSEMBLE: $(count_cases_from_predictions "$OUT_ENSEMBLE")"
else
    log_section "EXTERNAL PREDICTION - ENSEMBLE FOLDS ${FOLDS[*]}"
    echo "Pred case count before run in $OUT_ENSEMBLE: $(count_cases_from_predictions "$OUT_ENSEMBLE")"
    nnUNetv2_predict \
        -i "$EXT_IN" \
        -o "$OUT_ENSEMBLE" \
        -d "$DATASET_ID" \
        -c "$CONFIG" \
        -tr "$TRAINER" \
        -f 0 1 2 3 4
    echo "Pred case count after run in $OUT_ENSEMBLE: $(count_cases_from_predictions "$OUT_ENSEMBLE")"
fi

if [ -f "$SUMMARY_ENSEMBLE" ]; then
    log_section "SKIP EXTERNAL EVALUATION - ENSEMBLE (summary.json already exists)"
    echo "--- summary.json for ensemble ---"
    cat "$SUMMARY_ENSEMBLE"
    echo
else
    log_section "EXTERNAL EVALUATION - ENSEMBLE"
    nnUNetv2_evaluate_folder \
        "$EXT_GT" \
        "$OUT_ENSEMBLE" \
        -djfile "$DATASET_JSON" \
        -pfile "$PLANS_JSON" \
        -o "$SUMMARY_ENSEMBLE"

    if [ -f "$SUMMARY_ENSEMBLE" ]; then
        echo "--- summary.json for ensemble ---"
        cat "$SUMMARY_ENSEMBLE"
        echo
    else
        echo "WARNING: summary.json was not created for ensemble"
    fi
fi

log_section "PIPELINE COMPLETED"
echo "Single-fold preds/eval in: $EXT_PRED_BASE/fold_*"
echo "Ensemble preds/eval in:    $OUT_ENSEMBLE"

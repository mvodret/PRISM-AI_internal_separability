# FIG 3: Classifier trained on ground-truth vs. detected lesions

`FIG3_generation.ipynb` is a reduced, documentation-oriented version of the source notebook (`0_FINAL_pipe_GT_from_GH_genera_joblib.ipynb`) for FIG 3. It retains the stored outputs of the original runs and the code that links the input feature tables to the final figure: data loading, feature selection, classifier construction and training, evaluation on the external centers, saved plot data, and the figure itself. The notebook's markdown is in English.

The notebook is meant to be **read**, not re-run top to bottom: stored outputs come from different executions of the source notebook (execution counts are not sequential).

## What the figure shows

FIG 3 compares a lesion-level classifier (target: Gleason Grade Group `GG >= 3`) trained and tested on either ground-truth (GT) or detected lesions.

- **Top two rows (left):** the 10 features with the largest absolute mean ensemble coefficient.
- **Top two rows (right):** ROC curves on the construction set, NOVARA and TORINO. Row 1: classifier trained on GT lesions (solid = tested on GT, dashed = tested on detected lesions). Row 2: classifier trained on detected lesions.
- **Bottom row:** AUC comparison (train GT/detected × test GT/detected) for NOVARA and TORINO.

AUCs are computed by integrating the saved ROC points (`np.trapz` over `fpr`/`tpr`). The AUC table is saved to `report_final/three_rows_auc_table.csv`; the figure to `report_final/three_rows_with_auc_heatmaps.pdf`.

## Pipeline overview

The pipeline runs **four times**, once per value of `DSET_TYPE`. Only that variable changes.

| `DSET_TYPE` | Construction centers (US-MRI, PI-CAI) | External centers (NOVARA, TORINO) |
|---|---|---|
| `GT` | GT lesions | GT lesions |
| `MIXED_GT_TRAIN` | GT lesions | detected lesions |
| `SURROGATE` | detected lesions | detected lesions |
| `MIXED_SURR_TRAIN` | detected lesions | GT lesions |

"Detected lesions" are the lesions found by the segmentation model, kept only when `dice > 0` against the GT lesion.

For each run, the notebook:

1. **Loads the data** (`import_gt_dataset`, `import_pred_dataset`) and builds the dataset for the selected `DSET_TYPE`.
2. **Selects features** using the construction centers only. Candidates are filtered by univariate AUC signal and by robustness flags (tail/median, log-range), ranked by a robust score, and reduced to `N_FEATURES = 10` by greedy decorrelation (`SEL2`, `CORR_THR = 0.7`). PSA density is added to the selected features (`USE_PSA = True`).
3. **Splits** the data into the construction set (US-MRI + PI-CAI) and the external sets (NOVARA, TORINO).
4. **Defines the classifier**: `StandardScaler` → `RFE` (all features kept) → `LogisticRegression`, with `C` tuned by Optuna.
5. **Trains** with nested cross-validation on the construction set (5 outer folds; inner Optuna search, up to 100 trials with early stopping), and collects cross-validated decision thresholds (Youden, prevalence, F2, sensitivity 90%, specificity 90%, top-left). The cross-validation table is saved to `final_pipe_train_on_gt_psa.csv`.
6. **Rebuilds the best pipeline** and fits the deployed model: a scanner-wise mixture of bootstrap experts (`ScannerMixtureOfExpertsBootstrap`, 20 bootstrap experts per scanner). Thresholds are recomputed on the full construction set (the top-left threshold, `THRESHOLD = THRESHOLDS[5]`, is used for external evaluation). Bootstrap coefficients are averaged to obtain the feature importance. The model is saved with `joblib`.
7. **Evaluates** on NOVARA and TORINO (AUC, accuracy, precision, recall, F1, confusion matrices) and writes `report_final/plot_data/{DSET_TYPE}_classification_*.csv`.

The final cell reads, for each of the four tags, `*_classification_roc_points.csv`, `*_classification_metrics.csv` and `*_classification_feature_importance.csv`, and draws FIG 3.

## Inputs and requirements

Run from the project folder containing the paths below. The GT loader selects the fourth entry of its `feature_tables_FINAL_2D` folder list (the `BIN_16` lesion feature-table variant).

Required GT inputs:

- `feature_tables_FINAL_2D/T2_lesion_iqr_zscore_stats2D_apply2D_lesionsrc_original__ADC_lesion_iqr_zscore_stats2D_apply2D_lesionsrc_original_BIN_16/` — the `images_clean_*.csv`, `radiomics_clean_*.csv` and `labels_clean_*.csv` files selected by the loader.
- `feature_tables/psa_density_all.csv`
- `feature_tables/valid_columns.csv`
- `labels_scanner_uptodate.csv`

Required segmentation-model prediction inputs (relative to the project folder):

- `../for_uunet/images_preds_NEW.csv`
- `../for_uunet/df_features_preds_NEW.csv`
- `../for_uunet/images_preds_TRAIN_NEW.csv`
- `../for_uunet/df_features_preds_TRAIN_NEW.csv`

Python packages: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn`, `optuna`, `joblib`, `nibabel`, `scipy`.

## Outputs

- `final_pipe_train_on_gt_psa.csv` — cross-validation results of the classifier search.
- `SNIPPET_from_feature_to_pred_proba/data_report_GT/moe_classifier_trained.joblib` — trained classifier bundle.
- `report_final/plot_data/{DSET_TYPE}_classification_*.csv` — per-run plot data (ROC points, metrics, feature importance, probabilities, counts, confusion matrices).
- `report_final/three_rows_with_auc_heatmaps.pdf` — FIG 3.
- `report_final/three_rows_auc_table.csv` — AUC for each (row, run, dataset) combination.

## Interpretation notes

- External centers are never used for feature selection, hyperparameter tuning or threshold selection; they are used only for evaluation.
- Stored outputs of the data-loading cell refer to the `GT` run. The evaluation cell's stored output ends with `Saved plot data with prefix: report_final/plot_data/MIXED_SURR_TRAIN_classification`, i.e. it comes from the last run executed. Results of the other runs are available only through the saved CSV files and the final figure.
- The classifier-search space in the notebook is reduced to a single configuration; the commented-out options in the source cell are earlier alternatives.
- The reference AUCs from the stored run of the final cell are: GT-trained model tested on GT — NOVARA 0.655, TORINO 0.864; GT-trained tested on detected — NOVARA 0.819, TORINO 0.885; detected-trained tested on detected — NOVARA 0.828, TORINO 0.882; detected-trained tested on GT — NOVARA 0.693, TORINO 0.858. Input versions and filtering can change these values.

---

## Supplementary Information: algorithm parameters

All values below are taken verbatim from the code in `FIG3_generation.ipynb`. The same parameters are used for the four `DSET_TYPE` runs (only the data source changes).

### 1. Data and target

| Parameter | Value |
|---|---|
| Construction centres | US-MRI + PI-CAI (index split 876 + 425) |
| External validation centres | Novara (77 cases), Torino (remaining cases) |
| Target | binary, ISUP grade group `GG >= 3` vs `GG <= 2` |
| Image features | T2 / ADC lesion radiomics (bin width setting `BIN_16`) + shape features |
| Clinical feature | PSA density (`USE_PSA = True`) |
| Segmentation-derived cases (SURROGATE) | only cases with `dice > 0` |

### 2. Feature analysis (construction set only)

| Step | Parameter | Value |
|---|---|---|
| Centre-shift diagnostic (bootstrap) | `B` / `MIN_CENTER_SAMPLES` / `RANDOM_STATE` | 200 / 50 / 42 |
| Scanner-level statistics | `MIN_SCANNER_SAMPLES` (scanners with fewer cases are ignored) | 20 |
| Scanner shift | per-scanner z-shift `(mean_scanner − mean_all) / std_all`; summarised as mean/max absolute shift | – |
| Univariate signal | `auc_signal = abs(AUC − 0.5)`; computed only if both classes have ≥ 5 cases (≥ 10 valid values) | – |
| Robustness score | `robust_score2 = auc_signal / (mean_abs_shift + 1e-6)` | – |
| Per-scanner effect direction | class-mean difference, requires ≥ 3 cases per class and ≥ 20 cases per scanner | – |
| "Ugly feature" flags | `abs_skew > 3`, `tail_ratio > 8`, `spread_ratio > 20`, `tail_median_ratio > 15`, `log_range > 1.5`, `corr_with_shape_volume > 0.85`, shape-like feature | – |
| Result | 8 scanners / 696 construction cases retained | – |

### 3. Feature selection (`FEATURE_SELECTION_MODE = "SEL2"`)

| Parameter | Value |
|---|---|
| Candidate filter | `auc_signal > 0.05`, `not flag_tailmedian`, `not flag_logrange` |
| Ranking | descending `robust_score2` |
| Decorrelation | greedy, keep a feature only if `|corr| <= CORR_THR = 0.7` with those already kept |
| Number of features (`N_FEATURES`) | 10 radiomic/shape features (+ `psa_density` → 11 columns) |
| Fallback | if fewer than `N_FEATURES` survive, the filters are relaxed (see code) |

### 4. Classifier (per expert)

| Step | Setting |
|---|---|
| 1. `StandardScaler` | default |
| 2. `RFE` (selector) | estimator `LogisticRegression(random_state=42, max_iter=1000)`, `n_features_to_select` = number of input columns (11), i.e. no further reduction |
| 3. `LogisticRegression` (classifier) | L2 penalty, `max_iter=1000`, `class_weight=None`, `random_state=42` |
| Class imbalance handling | none |
| Tuned hyper-parameter | `C` only |

### 5. Nested cross-validation and hyper-parameter search

| Parameter | Value |
|---|---|
| Outer CV (`n_splits_outer`) | 5, `StratifiedKFold(shuffle=True, random_state=43)` |
| Inner CV (`n_splits_inner`) | 5 (reduced if the minority class is smaller), stratified |
| Search | Optuna, `direction="maximize"`, `TPESampler(seed=42)` |
| Objective | mean inner-CV ROC AUC |
| Search space | `C ∈ [1e-2, 1e2]`, log-uniform |
| Budget per outer fold | `n_trials = 100`, `timeout = 30 s`, `n_jobs_optuna = 1` |
| Early stopping | `patience = 10` trials, `min_delta = 1e-3` |
| Consensus of per-fold `C` | geometric median (median in log space) |
| Reported CV result | AUC 0.733 ± 0.047; selected `C = 0.3149` |
| Selection of final configuration | table sorted by ascending `std_auc_cv`; `K_BEST_COMBINATION = 0` |

### 6. Decision thresholds

Computed from out-of-fold scores (CV) and then recomputed on the whole construction set for the final model.

| Name | Definition |
|---|---|
| Youden | maximises `TPR − FPR` |
| Top-left | minimises distance to `(FPR, TPR) = (0, 1)` |
| Sens90 | lowest FPR with `TPR >= 0.90` |
| Spec90 | highest TPR with `FPR <= 0.10` |
| Prevalence | positive-class prevalence of the construction set |
| F2 | maximises F-beta, `beta = 2`, on the precision–recall curve |
| **Used for FIG. 3** | `THRESHOLD = median_threshold_topleft_cv` (top-left), value 0.3844 on the construction set |

### 7. Final model: scanner mixture of experts (`ScannerMixtureOfExpertsBootstrap`)

| Parameter | Value |
|---|---|
| `min_samples` (minimum cases per scanner to get an expert) | 50 |
| `n_bootstrap` (experts per scanner) | 20 |
| `bootstrap_size` | 70 cases per bootstrap (35 per class) |
| `balanced_bootstrap` | `True` (sampling with replacement within each class) |
| `random_state` | 42 |
| `min_minority_for_smote` | 6 (inactive: no sampler in the pipeline) |
| Valid scanners / experts | 4 scanners (E11 n=202, RUMC n=154, D13 n=100, ZGT n=67) → 80 experts |
| Within-scanner aggregation | mean probability across the 20 bootstrap experts |
| Across-scanner aggregation | weighted mean, weights ∝ `sqrt(n_scanner)` |
| Feature importance | mean of the logistic coefficients across all experts (non-selected features = 0) |
| Final model file | `data_report_GT/moe_classifier_trained.joblib` |

### 8. Evaluation and plotting

| Parameter | Value |
|---|---|
| Validation sets | combined, Novara, Torino (never used for training or selection) |
| Metrics | AUC, accuracy, precision, recall, F1, confusion matrix (at the top-left threshold) |
| FIG. 3 heatmaps | `TOP_N = 10` features per panel |
| AUC in FIG. 3 | trapezoidal integration of the saved ROC points (`np.trapz`) |
| Output tables | `report_final/plot_data/{DSET_TYPE}_classification_*.csv`, `report_final/three_rows_auc_table.csv` |

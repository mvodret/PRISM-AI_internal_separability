# FIG 4: Internal separability

`FIG4_generation.ipynb` is a reduced, standalone version of the source notebook for reproducing FIG 4. It retains the figure's stored output and the code required to load its inputs, calculate feature-level separability, and render the figure. The notebook's markdown is in English.

## What the figure measures

The analysis compares three cohorts:

1. **All GT lesions**: the ground-truth radiomic feature table.
2. **Matched GT lesions**: GT lesions selected by `case_id` values present in the combined surrogate prediction labels, before applying the detection filter. The GT feature rows are aligned using `Subject` (or the original index-column fallback in the source code).
3. **Detected lesions**: surrogate radiomic rows with `dice > 0`. This filter is applied only to this cohort, after the matched-GT cohort is built.

The comparison is performed separately for the **TORINO** and **NOVARA** centers. Center names are normalized before analysis.

For each radiomic feature and center, the code:


- Defines the positive class as `GG >= 3` and the negative class as `GG <= 2`; rows with `GG <= -1` are excluded.
- Requires at least five valid observations in **each** class. If either class has fewer than five observations, that feature/center AUC is recorded as missing.
- Calculates the ROC AUC, then converts it to the direction-independent separability signal
  `AUC_f = abs(AUC - 0.5)`.

Thus, `AUC_f` measures distance from chance-level separability, regardless of which class has the higher feature values. It ranges from 0 (chance-level) to 0.5 (perfect separation); it is not a signed measure of prediction direction.

The right-hand panel calculates the signed difference
`Delta AUC_f = Detected - Matched GT`.
Features are ranked by their mean delta within each modality. The displayed delta panel contains the top 10 ADC, top 10 T2, and top 8 Shape features. The left panel compares signal across all three cohorts for the ADC, T2, and Shape features included by the source plotting code.

## Inputs and requirements

Run the notebook with the working directory set to the project folder containing the input paths below. The GT loader selects the fourth entry in its configured `feature_tables_FINAL_2D` folder list (the `BIN_16` lesion feature-table variant).

Required GT inputs:

- `feature_tables_FINAL_2D/T2_lesion_iqr_zscore_stats2D_apply2D_lesionsrc_original__ADC_lesion_iqr_zscore_stats2D_apply2D_lesionsrc_original_BIN_16/` — the `images_clean_*.csv`, `radiomics_clean_*.csv`, and `labels_clean_*.csv` files selected by the loader.
- `feature_tables/psa_density_all.csv`
- `feature_tables/valid_columns.csv`
- `labels_scanner_uptodate.csv`

Required surrogate/prediction inputs, relative to the project folder:

- `../for_uunet/images_preds_NEW.csv`
- `../for_uunet/df_features_preds_NEW.csv`
- `../for_uunet/images_preds_TRAIN_NEW.csv`
- `../for_uunet/df_features_preds_TRAIN_NEW.csv`

The notebook uses Python packages `numpy`, `pandas`, `matplotlib`, `seaborn`, and `scikit-learn`. 

## Interpretation notes

- This is a univariate, within-cohort separability comparison. It is not a held-out predictive-performance estimate.
- The source notebook's retained FIG 4 output reported 1,462 all-GT lesions, 897 matched-GT lesions, and 917 detected lesions. These are reference counts from that saved run; input versions and filtering can change them.


---

## Supplementary Information: algorithm parameters

All values are taken from the code in `FIG4_generation.ipynb`.

| Parameter | Value |
|---|---|
| Positive class | ISUP grade group `GG >= 3` (negative: `0 <= GG <= 2`; cases with `GG = -1` excluded) |
| Centres (`CENTER_ORDER`) | TORINO, NOVARA (centre names are normalised before use) |
| Cohorts | All GT; Matched GT (GT lesions whose `case_id` appears in the combined surrogate prediction labels, before Dice filtering); Detected (nnU-Net lesions with `dice > 0`) |
| Univariate separability | `AUC_f = abs(ROC AUC − 0.5)` per feature and centre |
| Minimum class size (`MIN_CLASS_PER_CENTER`) | 5 per class and per centre, otherwise the feature is skipped (NaN) |
| Delta panel | `Detected − Matched GT` of `AUC_f` |
| Features shown in the delta panel | top 10 ADC (`TOP_ADC_DELTA`), top 10 T2 (`TOP_T2_DELTA`), top 8 Shape (`TOP_SHAPE_DELTA`), ranked by the delta |
| Colour scales | Left heatmaps: `Reds`, shared `vmin = 0` and `vmax` equal to the global maximum across the three cohorts; delta panel: `Purples`, `vmin = 0` and `vmax` equal to the maximum plotted delta (or 1.0 if that maximum is not positive) |
| Figure | `figsize = (5, 6)`, font size 10, output `report_final/feature_separability_gt_matched_detected_delta.pdf` |
| Cohort sizes (reference) | 1462 (All GT) / 897 (Matched GT) / 917 (Detected) |

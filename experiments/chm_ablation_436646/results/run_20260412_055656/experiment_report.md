# CHM Ablation Report

- Tile ID: 436646
- Years: 2018, 2020, 2022, 2024
- Dry run: False
- Generated: 2026-04-12T08:44:34

## Labels

- 2018: /workspace/output/model_search_v3_academic_leakage26/prepared/labels_curated_v2/436646_2018_madal_chm_max_hag_20cm_labels.csv
- 2020: /workspace/output/model_search_v3_academic_leakage26/prepared/labels_curated_v2/436646_2020_madal_chm_max_hag_20cm_labels.csv
- 2022: /workspace/output/model_search_v3_academic_leakage26/prepared/labels_curated_v2/436646_2022_madal_chm_max_hag_20cm_labels.csv
- 2024: /workspace/output/model_search_v3_academic_leakage26/prepared/labels_curated_v2/436646_2024_madal_chm_max_hag_20cm_labels.csv

## Selected Models

- Total selected: 5

| source | experiment_id | model_name | mean_cv_f1 | checkpoint |
|---|---|---|---:|---|
| v3 | s1_convnext_small_full_ce_mixup_40_5_3b1439db | convnext_small | 0.988164 | /workspace/output/model_search_v3_academic_leakage26/checkpoints/s1_convnext_small_full_ce_mixup_40_5_3b1439db/fold1.pt |
| v3 | s1_efficientnet_b2_full_ce_mixup_40_5_7a5bb15c | efficientnet_b2 | 0.983008 | /workspace/output/model_search_v3_academic_leakage26/checkpoints/s1_efficientnet_b2_full_ce_mixup_40_5_7a5bb15c/fold1.pt |
| v2 | s2_convnext_tiny_full_ce_mixup_tta_1_ep_60_focused_3fb56d79 | convnext_tiny | 0.923282 | /workspace/output/model_search_v2/checkpoints/s2_convnext_tiny_full_ce_mixup_tta_1_ep_60_focused_3fb56d79/fold1.pt |
| v2 | s2_convnext_small_full_ce_mixup_swa_tta_0_ep_60_focused_df4d4976 | convnext_small | 0.921680 | /workspace/output/model_search_v2/checkpoints/s2_convnext_small_full_ce_mixup_swa_tta_0_ep_60_focused_df4d4976/fold1.pt |
| v2 | s2_convnext_small_full_focal_mixup_tta_0_ep_60_focused_3e06b9c6 | convnext_small | 0.920638 | /workspace/output/model_search_v2/checkpoints/s2_convnext_small_full_focal_mixup_tta_0_ep_60_focused_3e06b9c6/fold1.pt |

## CHM Discovery

- Total CHMs discovered: 380

## Top CHMs (Mean F1 Across Selected Models)

| rank | year | chm_name | mean_f1 | mean_recall | mean_precision | n_models |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 2018 | 436646_2018_madal_chm_max_hag_20cm.tif | 1.0000 | 1.0000 | 1.0000 | 5 |
| 2 | 2018 | 436646_2018_madal_chm_max_hag_20cm_last2.tif | 1.0000 | 1.0000 | 1.0000 | 5 |
| 3 | 2018 | 436646_2018_madal_classified_myria3d_chm_max_hag_20cm_all.tif | 1.0000 | 1.0000 | 1.0000 | 5 |
| 4 | 2018 | 436646_2018_madal_classified_myria3d_chm_max_hag_20cm_last2.tif | 1.0000 | 1.0000 | 1.0000 | 5 |
| 5 | 2018 | 436646_2018_madal_reclassified_pdal_chm_max_hag_20cm_all.tif | 1.0000 | 1.0000 | 1.0000 | 5 |
| 6 | 2018 | 436646_2018_madal_reclassified_pdal_chm_max_hag_20cm_last2.tif | 1.0000 | 1.0000 | 1.0000 | 5 |
| 7 | 2022 | 436646_2022_madal_chm_max_hag_20cm.tif | 1.0000 | 1.0000 | 1.0000 | 5 |
| 8 | 2024 | 436646_2024_madal_chm_max_hag_20cm.tif | 1.0000 | 1.0000 | 1.0000 | 5 |
| 9 | 2020 | 436646_2020_madal_chm_max_hag_20cm.tif | 0.9988 | 1.0000 | 0.9976 | 5 |
| 10 | 2018 | 436646_2018_madal_chm_max_hag_20cm_last.tif | 0.9953 | 0.9909 | 1.0000 | 5 |
| 11 | 2018 | 436646_2018_madal_classified_myria3d_chm_max_hag_20cm_last.tif | 0.9953 | 0.9909 | 1.0000 | 5 |
| 12 | 2018 | 436646_2018_madal_reclassified_pdal_chm_max_hag_20cm_last.tif | 0.9953 | 0.9909 | 1.0000 | 5 |
| 13 | 2022 | 436646_2022_madal_exp_return_chm13_median_ground_last_20cm.tif | 0.9602 | 0.9264 | 0.9970 | 5 |
| 14 | 2024 | 436646_2024_madal_exp_return_chm13_median_ground_last_20cm.tif | 0.9593 | 0.9222 | 1.0000 | 5 |
| 15 | 2020 | 436646_2020_madal_exp_return_chm13_median_ground_all_20cm.tif | 0.9405 | 0.9073 | 0.9766 | 5 |
| 16 | 2020 | 436646_2020_madal_exp_return_chm13_median_ground_last2_20cm.tif | 0.9405 | 0.9073 | 0.9766 | 5 |
| 17 | 2022 | 436646_2022_madal_exp_return_chm13_median_ground_all_20cm.tif | 0.9312 | 0.8778 | 0.9924 | 5 |
| 18 | 2022 | 436646_2022_madal_exp_return_chm13_median_ground_last2_20cm.tif | 0.9312 | 0.8778 | 0.9924 | 5 |
| 19 | 2020 | 436646_2020_madal_exp_return_chm13_median_ground_last_20cm.tif | 0.9235 | 0.8732 | 0.9815 | 5 |
| 20 | 2022 | 436646_2022_madal_exp_return_chm13_lowest_ground_last_20cm.tif | 0.9234 | 0.8583 | 1.0000 | 5 |

## Top Variants (Across Years)

| rank | variant_id | mean_f1 | mean_recall | n_years | n_model_chm_cases |
|---:|---|---:|---:|---:|---:|
| 1 | madal_chm_max_hag_20cm_last2 | 1.0000 | 1.0000 | 1 | 5 |
| 2 | madal_classified_myria3d_chm_max_hag_20cm_all | 1.0000 | 1.0000 | 1 | 5 |
| 3 | madal_classified_myria3d_chm_max_hag_20cm_last2 | 1.0000 | 1.0000 | 1 | 5 |
| 4 | madal_reclassified_pdal_chm_max_hag_20cm_all | 1.0000 | 1.0000 | 1 | 5 |
| 5 | madal_reclassified_pdal_chm_max_hag_20cm_last2 | 1.0000 | 1.0000 | 1 | 5 |
| 6 | madal_chm_max_hag_20cm_last | 0.9953 | 0.9909 | 1 | 5 |
| 7 | madal_classified_myria3d_chm_max_hag_20cm_last | 0.9953 | 0.9909 | 1 | 5 |
| 8 | madal_reclassified_pdal_chm_max_hag_20cm_last | 0.9953 | 0.9909 | 1 | 5 |
| 9 | madal_exp_return_chm13_lowest_ground_last_20cm | 0.8671 | 0.7810 | 4 | 45 |
| 10 | madal_exp_return_chm13_lowest_ground_all_20cm | 0.8670 | 0.7822 | 4 | 20 |
| 11 | madal_exp_return_chm13_lowest_ground_last2_20cm | 0.8670 | 0.7822 | 4 | 20 |
| 12 | madal_exp_return_chm13_median_ground_all_20cm | 0.8270 | 0.7637 | 4 | 25 |
| 13 | madal_exp_return_chm13_median_ground_last2_20cm | 0.8270 | 0.7637 | 4 | 25 |
| 14 | madal_exp_return_chm13_median_ground_last_20cm | 0.8148 | 0.7607 | 4 | 25 |
| 15 | madal_exp_intensity_04_13_lowest_ground_all_20cm | 0.8044 | 0.7528 | 4 | 20 |
| 16 | madal_exp_intensity_04_13_lowest_ground_last2_20cm | 0.8044 | 0.7528 | 4 | 20 |
| 17 | madal_exp_intensity_04_13_median_ground_last2_20cm | 0.7948 | 0.7382 | 4 | 20 |
| 18 | madal_exp_intensity_04_13_median_ground_last_20cm | 0.7489 | 0.7015 | 4 | 20 |
| 19 | madal_exp_intensity_04_13_lowest_ground_last_20cm | 0.7484 | 0.7023 | 4 | 20 |
| 20 | madal_exp_intensity_04_13_median_ground_all_20cm | 0.6832 | 0.6379 | 4 | 25 |

## Model Robustness Across CHMs

| source | experiment_id | model_name | mean_f1 | mean_recall | mean_precision | n_chms |
|---|---|---|---:|---:|---:|---:|
| v3 | s1_convnext_small_full_ce_mixup_40_5_3b1439db | convnext_small | 0.3822 | 0.3286 | 0.6834 | 380 |
| v2 | s2_convnext_small_full_focal_mixup_tta_0_ep_60_focused_3e06b9c6 | convnext_small | 0.3334 | 0.3026 | 0.5185 | 380 |
| v3 | s1_efficientnet_b2_full_ce_mixup_40_5_7a5bb15c | efficientnet_b2 | 0.3160 | 0.2744 | 0.5451 | 380 |
| v2 | s2_convnext_tiny_full_ce_mixup_tta_1_ep_60_focused_3fb56d79 | convnext_tiny | 0.2684 | 0.2267 | 0.4488 | 380 |
| v2 | s2_convnext_small_full_ce_mixup_swa_tta_0_ep_60_focused_df4d4976 | convnext_small | 0.2344 | 0.2003 | 0.4341 | 380 |


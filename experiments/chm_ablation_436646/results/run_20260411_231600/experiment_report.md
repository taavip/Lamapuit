# CHM Ablation Report

- Tile ID: 436646
- Years: 2018, 2020, 2022, 2024
- Dry run: False
- Generated: 2026-04-11T23:16:23

## Labels

- 2018: /workspace/output/model_search_v3_academic_leakage26/prepared/labels_curated_v2/436646_2018_madal_chm_max_hag_20cm_labels.csv
- 2020: /workspace/output/model_search_v3_academic_leakage26/prepared/labels_curated_v2/436646_2020_madal_chm_max_hag_20cm_labels.csv
- 2022: /workspace/output/model_search_v3_academic_leakage26/prepared/labels_curated_v2/436646_2022_madal_chm_max_hag_20cm_labels.csv
- 2024: /workspace/output/model_search_v3_academic_leakage26/prepared/labels_curated_v2/436646_2024_madal_chm_max_hag_20cm_labels.csv

## Selected Models

- Total selected: 3

| source | experiment_id | model_name | mean_cv_f1 | checkpoint |
|---|---|---|---:|---|
| v3 | s1_convnext_small_full_ce_mixup_40_5_3b1439db | convnext_small | 0.988164 | /workspace/output/model_search_v3_academic_leakage26/checkpoints/s1_convnext_small_full_ce_mixup_40_5_3b1439db/fold1.pt |
| v3 | s1_efficientnet_b2_full_ce_mixup_40_5_7a5bb15c | efficientnet_b2 | 0.983008 | /workspace/output/model_search_v3_academic_leakage26/checkpoints/s1_efficientnet_b2_full_ce_mixup_40_5_7a5bb15c/fold1.pt |
| v2 | s2_convnext_tiny_full_ce_mixup_tta_1_ep_60_focused_3fb56d79 | convnext_tiny | 0.923282 | /workspace/output/model_search_v2/checkpoints/s2_convnext_tiny_full_ce_mixup_tta_1_ep_60_focused_3fb56d79/fold1.pt |

## CHM Discovery

- Total CHMs discovered: 1

## Top CHMs (Mean F1 Across Selected Models)

| rank | year | chm_name | mean_f1 | mean_recall | mean_precision | n_models |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 2018 | 436646_2018_madal_cdw_features_lowest_all_all_20cm.tif | 0.0278 | 0.0152 | 0.1667 | 3 |

## Top Variants (Across Years)

| rank | variant_id | mean_f1 | mean_recall | n_years | n_model_chm_cases |
|---:|---|---:|---:|---:|---:|
| 1 | madal_cdw_features_lowest_all_all_20cm | 0.0278 | 0.0152 | 1 | 3 |

## Model Robustness Across CHMs

| source | experiment_id | model_name | mean_f1 | mean_recall | mean_precision | n_chms |
|---|---|---|---:|---:|---:|---:|
| v3 | s1_convnext_small_full_ce_mixup_40_5_3b1439db | convnext_small | 0.0833 | 0.0455 | 0.5000 | 1 |
| v2 | s2_convnext_tiny_full_ce_mixup_tta_1_ep_60_focused_3fb56d79 | convnext_tiny | 0.0000 | 0.0000 | 0.0000 | 1 |
| v3 | s1_efficientnet_b2_full_ce_mixup_40_5_7a5bb15c | efficientnet_b2 | 0.0000 | 0.0000 | 0.0000 | 1 |


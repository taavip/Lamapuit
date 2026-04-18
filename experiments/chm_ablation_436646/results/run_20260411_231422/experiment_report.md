# CHM Ablation Report

- Tile ID: 436646
- Years: 2018, 2020, 2022, 2024
- Dry run: True
- Generated: 2026-04-11T23:14:23

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
| v3 | s1_deep_cnn_attn_dropout_tuned_full_ce_mixup_60_5_5766c021 | deep_cnn_attn_dropout_tuned | 0.948660 | /workspace/output/model_search_v3_academic_leakage26/checkpoints/s1_deep_cnn_attn_dropout_tuned_full_ce_mixup_60_5_5766c021/fold1.pt |
| v3 | s1_maxvit_small_full_ce_mixup_3_2_b75ef730 | maxvit_small | 0.934922 | /workspace/output/model_search_v3_academic_leakage26/checkpoints/s1_maxvit_small_full_ce_mixup_3_2_b75ef730/fold1.pt |
| v2 | s2_convnext_tiny_full_ce_mixup_tta_1_ep_60_focused_3fb56d79 | convnext_tiny | 0.923282 | /workspace/output/model_search_v2/checkpoints/s2_convnext_tiny_full_ce_mixup_tta_1_ep_60_focused_3fb56d79/fold1.pt |

## CHM Discovery

- Total CHMs discovered: 12

Dry run completed. Metrics are not computed in dry-run mode.


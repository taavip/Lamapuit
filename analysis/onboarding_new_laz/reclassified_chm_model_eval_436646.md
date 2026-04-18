# Reclassified LAS->CHM Detection Benchmark (436646_2018_madal)

- Labels: output/onboarding_labels_v2_drop13/436646_2018_madal_chm_max_hag_20cm_labels.csv
- Sources included: manual/reviewed
- Metric focus: recall (CDW detection rate), plus F1/precision/AUC

## Improvement vs Baseline (Per Model)

| model | baseline_recall | best_new_recall | recall_delta | baseline_f1 | best_new_f1 | best_new_chm |
|---|---:|---:|---:|---:|---:|---|
| cnn_seed42 | 0.6389 | 0.6389 | +0.0000 | 0.5529 | 0.5897 | 436646_2018_madal_classified_myria3d_chm_max_hag_20cm_last.tif |
| cnn_seed43 | 0.5944 | 0.6722 | +0.0778 | 0.5737 | 0.6385 | 436646_2018_madal_classified_myria3d_chm_max_hag_20cm_last.tif |
| cnn_seed44 | 0.5722 | 0.6278 | +0.0556 | 0.6358 | 0.6647 | 436646_2018_madal_classified_myria3d_chm_max_hag_20cm_last.tif |
| effnet_b2 | 0.6111 | 0.6222 | +0.0111 | 0.5584 | 0.5818 | 436646_2018_madal_reclassified_pdal_chm_max_hag_20cm_last.tif |
| ensemble_model | 0.5944 | 0.6167 | +0.0222 | 0.6524 | 0.6491 | 436646_2018_madal_classified_myria3d_chm_max_hag_20cm_last.tif |

## Full Metrics

| model | chm_group | chm | thr | n | acc | prec | recall | f1 | auc | tp | fp | tn | fn |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ensemble_model | baseline | 436646_2018_madal_chm_max_hag_20cm.tif | 0.66 | 630 | 0.8190 | 0.7230 | 0.5944 | 0.6524 | 0.8631 | 107 | 41 | 409 | 73 |
| cnn_seed42 | baseline | 436646_2018_madal_chm_max_hag_20cm.tif | 0.50 | 630 | 0.7048 | 0.4873 | 0.6389 | 0.5529 | 0.7930 | 115 | 121 | 329 | 65 |
| cnn_seed43 | baseline | 436646_2018_madal_chm_max_hag_20cm.tif | 0.50 | 630 | 0.7476 | 0.5544 | 0.5944 | 0.5737 | 0.8127 | 107 | 86 | 364 | 73 |
| cnn_seed44 | baseline | 436646_2018_madal_chm_max_hag_20cm.tif | 0.50 | 630 | 0.8127 | 0.7153 | 0.5722 | 0.6358 | 0.8404 | 103 | 41 | 409 | 77 |
| effnet_b2 | baseline | 436646_2018_madal_chm_max_hag_20cm.tif | 0.50 | 630 | 0.7238 | 0.5140 | 0.6111 | 0.5584 | 0.7903 | 110 | 104 | 346 | 70 |
| ensemble_model | myria3d | 436646_2018_madal_classified_myria3d_chm_max_hag_20cm_all.tif | 0.66 | 630 | 0.8190 | 0.7230 | 0.5944 | 0.6524 | 0.8628 | 107 | 41 | 409 | 73 |
| cnn_seed42 | myria3d | 436646_2018_madal_classified_myria3d_chm_max_hag_20cm_all.tif | 0.50 | 630 | 0.7016 | 0.4831 | 0.6333 | 0.5481 | 0.7912 | 114 | 122 | 328 | 66 |
| cnn_seed43 | myria3d | 436646_2018_madal_classified_myria3d_chm_max_hag_20cm_all.tif | 0.50 | 630 | 0.7476 | 0.5544 | 0.5944 | 0.5737 | 0.8125 | 107 | 86 | 364 | 73 |
| cnn_seed44 | myria3d | 436646_2018_madal_classified_myria3d_chm_max_hag_20cm_all.tif | 0.50 | 630 | 0.8127 | 0.7153 | 0.5722 | 0.6358 | 0.8404 | 103 | 41 | 409 | 77 |
| effnet_b2 | myria3d | 436646_2018_madal_classified_myria3d_chm_max_hag_20cm_all.tif | 0.50 | 630 | 0.7238 | 0.5140 | 0.6111 | 0.5584 | 0.7900 | 110 | 104 | 346 | 70 |
| ensemble_model | myria3d | 436646_2018_madal_classified_myria3d_chm_max_hag_20cm_last.tif | 0.66 | 630 | 0.8095 | 0.6852 | 0.6167 | 0.6491 | 0.8510 | 111 | 51 | 399 | 69 |
| cnn_seed42 | myria3d | 436646_2018_madal_classified_myria3d_chm_max_hag_20cm_last.tif | 0.50 | 630 | 0.7460 | 0.5476 | 0.6389 | 0.5897 | 0.7990 | 115 | 95 | 355 | 65 |
| cnn_seed43 | myria3d | 436646_2018_madal_classified_myria3d_chm_max_hag_20cm_last.tif | 0.50 | 630 | 0.7825 | 0.6080 | 0.6722 | 0.6385 | 0.8263 | 121 | 78 | 372 | 59 |
| cnn_seed44 | myria3d | 436646_2018_madal_classified_myria3d_chm_max_hag_20cm_last.tif | 0.50 | 630 | 0.8190 | 0.7063 | 0.6278 | 0.6647 | 0.8197 | 113 | 47 | 403 | 67 |
| effnet_b2 | myria3d | 436646_2018_madal_classified_myria3d_chm_max_hag_20cm_last.tif | 0.50 | 630 | 0.7429 | 0.5441 | 0.6167 | 0.5781 | 0.8133 | 111 | 93 | 357 | 69 |
| ensemble_model | myria3d | 436646_2018_madal_classified_myria3d_chm_max_hag_20cm_last2.tif | 0.66 | 630 | 0.8190 | 0.7230 | 0.5944 | 0.6524 | 0.8628 | 107 | 41 | 409 | 73 |
| cnn_seed42 | myria3d | 436646_2018_madal_classified_myria3d_chm_max_hag_20cm_last2.tif | 0.50 | 630 | 0.7016 | 0.4831 | 0.6333 | 0.5481 | 0.7912 | 114 | 122 | 328 | 66 |
| cnn_seed43 | myria3d | 436646_2018_madal_classified_myria3d_chm_max_hag_20cm_last2.tif | 0.50 | 630 | 0.7476 | 0.5544 | 0.5944 | 0.5737 | 0.8125 | 107 | 86 | 364 | 73 |
| cnn_seed44 | myria3d | 436646_2018_madal_classified_myria3d_chm_max_hag_20cm_last2.tif | 0.50 | 630 | 0.8127 | 0.7153 | 0.5722 | 0.6358 | 0.8404 | 103 | 41 | 409 | 77 |
| effnet_b2 | myria3d | 436646_2018_madal_classified_myria3d_chm_max_hag_20cm_last2.tif | 0.50 | 630 | 0.7238 | 0.5140 | 0.6111 | 0.5584 | 0.7900 | 110 | 104 | 346 | 70 |
| ensemble_model | pdal | 436646_2018_madal_reclassified_pdal_chm_max_hag_20cm_all.tif | 0.66 | 630 | 0.8190 | 0.7200 | 0.6000 | 0.6545 | 0.8620 | 108 | 42 | 408 | 72 |
| cnn_seed42 | pdal | 436646_2018_madal_reclassified_pdal_chm_max_hag_20cm_all.tif | 0.50 | 630 | 0.7032 | 0.4852 | 0.6389 | 0.5516 | 0.7893 | 115 | 122 | 328 | 65 |
| cnn_seed43 | pdal | 436646_2018_madal_reclassified_pdal_chm_max_hag_20cm_all.tif | 0.50 | 630 | 0.7508 | 0.5602 | 0.5944 | 0.5768 | 0.8108 | 107 | 84 | 366 | 73 |
| cnn_seed44 | pdal | 436646_2018_madal_reclassified_pdal_chm_max_hag_20cm_all.tif | 0.50 | 630 | 0.8111 | 0.7133 | 0.5667 | 0.6316 | 0.8389 | 102 | 41 | 409 | 78 |
| effnet_b2 | pdal | 436646_2018_madal_reclassified_pdal_chm_max_hag_20cm_all.tif | 0.50 | 630 | 0.7222 | 0.5115 | 0.6167 | 0.5592 | 0.7887 | 111 | 106 | 344 | 69 |
| ensemble_model | pdal | 436646_2018_madal_reclassified_pdal_chm_max_hag_20cm_last.tif | 0.66 | 630 | 0.8095 | 0.6852 | 0.6167 | 0.6491 | 0.8503 | 111 | 51 | 399 | 69 |
| cnn_seed42 | pdal | 436646_2018_madal_reclassified_pdal_chm_max_hag_20cm_last.tif | 0.50 | 630 | 0.7460 | 0.5476 | 0.6389 | 0.5897 | 0.7974 | 115 | 95 | 355 | 65 |
| cnn_seed43 | pdal | 436646_2018_madal_reclassified_pdal_chm_max_hag_20cm_last.tif | 0.50 | 630 | 0.7825 | 0.6080 | 0.6722 | 0.6385 | 0.8245 | 121 | 78 | 372 | 59 |
| cnn_seed44 | pdal | 436646_2018_madal_reclassified_pdal_chm_max_hag_20cm_last.tif | 0.50 | 630 | 0.8175 | 0.7044 | 0.6222 | 0.6608 | 0.8184 | 112 | 47 | 403 | 68 |
| effnet_b2 | pdal | 436646_2018_madal_reclassified_pdal_chm_max_hag_20cm_last.tif | 0.50 | 630 | 0.7444 | 0.5463 | 0.6222 | 0.5818 | 0.8129 | 112 | 93 | 357 | 68 |
| ensemble_model | pdal | 436646_2018_madal_reclassified_pdal_chm_max_hag_20cm_last2.tif | 0.66 | 630 | 0.8190 | 0.7200 | 0.6000 | 0.6545 | 0.8620 | 108 | 42 | 408 | 72 |
| cnn_seed42 | pdal | 436646_2018_madal_reclassified_pdal_chm_max_hag_20cm_last2.tif | 0.50 | 630 | 0.7032 | 0.4852 | 0.6389 | 0.5516 | 0.7893 | 115 | 122 | 328 | 65 |
| cnn_seed43 | pdal | 436646_2018_madal_reclassified_pdal_chm_max_hag_20cm_last2.tif | 0.50 | 630 | 0.7508 | 0.5602 | 0.5944 | 0.5768 | 0.8108 | 107 | 84 | 366 | 73 |
| cnn_seed44 | pdal | 436646_2018_madal_reclassified_pdal_chm_max_hag_20cm_last2.tif | 0.50 | 630 | 0.8111 | 0.7133 | 0.5667 | 0.6316 | 0.8389 | 102 | 41 | 409 | 78 |
| effnet_b2 | pdal | 436646_2018_madal_reclassified_pdal_chm_max_hag_20cm_last2.tif | 0.50 | 630 | 0.7222 | 0.5115 | 0.6167 | 0.5592 | 0.7887 | 111 | 106 | 344 | 69 |
| ensemble_model | rf_best_open_source | 436646_2018_madal_reclassified_rf_chm_max_hag_20cm_all.tif | 0.66 | 630 | 0.6492 | 0.0444 | 0.0111 | 0.0178 | 0.4692 | 2 | 43 | 407 | 178 |
| cnn_seed42 | rf_best_open_source | 436646_2018_madal_reclassified_rf_chm_max_hag_20cm_all.tif | 0.50 | 630 | 0.6381 | 0.1667 | 0.0667 | 0.0952 | 0.4046 | 12 | 60 | 390 | 168 |
| cnn_seed43 | rf_best_open_source | 436646_2018_madal_reclassified_rf_chm_max_hag_20cm_all.tif | 0.50 | 630 | 0.6238 | 0.1200 | 0.0500 | 0.0706 | 0.3957 | 9 | 66 | 384 | 171 |
| cnn_seed44 | rf_best_open_source | 436646_2018_madal_reclassified_rf_chm_max_hag_20cm_all.tif | 0.50 | 630 | 0.6556 | 0.1224 | 0.0333 | 0.0524 | 0.4894 | 6 | 43 | 407 | 174 |
| effnet_b2 | rf_best_open_source | 436646_2018_madal_reclassified_rf_chm_max_hag_20cm_all.tif | 0.50 | 630 | 0.5952 | 0.1795 | 0.1167 | 0.1414 | 0.4870 | 21 | 96 | 354 | 159 |
| ensemble_model | rf_best_open_source | 436646_2018_madal_reclassified_rf_chm_max_hag_20cm_last.tif | 0.66 | 630 | 0.6746 | 0.0690 | 0.0111 | 0.0191 | 0.5682 | 2 | 27 | 423 | 178 |
| cnn_seed42 | rf_best_open_source | 436646_2018_madal_reclassified_rf_chm_max_hag_20cm_last.tif | 0.50 | 630 | 0.6810 | 0.1818 | 0.0333 | 0.0563 | 0.5638 | 6 | 27 | 423 | 174 |
| cnn_seed43 | rf_best_open_source | 436646_2018_madal_reclassified_rf_chm_max_hag_20cm_last.tif | 0.50 | 630 | 0.6619 | 0.1489 | 0.0389 | 0.0617 | 0.4886 | 7 | 40 | 410 | 173 |
| cnn_seed44 | rf_best_open_source | 436646_2018_madal_reclassified_rf_chm_max_hag_20cm_last.tif | 0.50 | 630 | 0.6683 | 0.0857 | 0.0167 | 0.0279 | 0.5356 | 3 | 32 | 418 | 177 |
| effnet_b2 | rf_best_open_source | 436646_2018_madal_reclassified_rf_chm_max_hag_20cm_last.tif | 0.50 | 630 | 0.6619 | 0.2000 | 0.0611 | 0.0936 | 0.5479 | 11 | 44 | 406 | 169 |
| ensemble_model | rf_best_open_source | 436646_2018_madal_reclassified_rf_chm_max_hag_20cm_last2.tif | 0.66 | 630 | 0.6476 | 0.0435 | 0.0111 | 0.0177 | 0.4716 | 2 | 44 | 406 | 178 |
| cnn_seed42 | rf_best_open_source | 436646_2018_madal_reclassified_rf_chm_max_hag_20cm_last2.tif | 0.50 | 630 | 0.6333 | 0.1408 | 0.0556 | 0.0797 | 0.4077 | 10 | 61 | 389 | 170 |
| cnn_seed43 | rf_best_open_source | 436646_2018_madal_reclassified_rf_chm_max_hag_20cm_last2.tif | 0.50 | 630 | 0.6222 | 0.1081 | 0.0444 | 0.0630 | 0.3968 | 8 | 66 | 384 | 172 |
| cnn_seed44 | rf_best_open_source | 436646_2018_madal_reclassified_rf_chm_max_hag_20cm_last2.tif | 0.50 | 630 | 0.6540 | 0.1200 | 0.0333 | 0.0522 | 0.4858 | 6 | 44 | 406 | 174 |
| effnet_b2 | rf_best_open_source | 436646_2018_madal_reclassified_rf_chm_max_hag_20cm_last2.tif | 0.50 | 630 | 0.5921 | 0.1709 | 0.1111 | 0.1347 | 0.4945 | 20 | 97 | 353 | 160 |

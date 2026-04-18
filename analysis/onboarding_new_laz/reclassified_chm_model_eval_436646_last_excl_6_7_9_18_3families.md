# Reclassified LAS->CHM Detection Benchmark (436646_2018_madal)

- Labels: output/onboarding_labels_v2_drop13/436646_2018_madal_chm_max_hag_20cm_labels.csv
- Sources included: manual/reviewed
- Metric focus: recall (CDW detection rate), plus F1/precision/AUC

## Improvement vs Baseline (Per Model)

| model | baseline_recall | best_new_recall | recall_delta | baseline_f1 | best_new_f1 | best_new_chm |
|---|---:|---:|---:|---:|---:|---|
| cnn_seed43 | 0.5944 | 0.1722 | -0.4222 | 0.5737 | 0.2394 | 436646_2018_madal_exp_return_chm13_lowest_ground_last_20cm.tif |
| effnet_b2 | 0.6111 | 0.0889 | -0.5222 | 0.5584 | 0.1488 | 436646_2018_madal_exp_return_chm13_median_all_last_20cm.tif |
| ensemble_model | 0.5944 | 0.0722 | -0.5222 | 0.6524 | 0.1313 | 436646_2018_madal_exp_return_chm13_lowest_ground_last_20cm.tif |

## Full Metrics

| model | chm_group | chm | thr | n | acc | prec | recall | f1 | auc | tp | fp | tn | fn |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ensemble_model | baseline | 436646_2018_madal_chm_max_hag_20cm.tif | 0.66 | 630 | 0.8190 | 0.7230 | 0.5944 | 0.6524 | 0.8631 | 107 | 41 | 409 | 73 |
| cnn_seed43 | baseline | 436646_2018_madal_chm_max_hag_20cm.tif | 0.50 | 630 | 0.7476 | 0.5544 | 0.5944 | 0.5737 | 0.8127 | 107 | 86 | 364 | 73 |
| effnet_b2 | baseline | 436646_2018_madal_chm_max_hag_20cm.tif | 0.50 | 630 | 0.7238 | 0.5140 | 0.6111 | 0.5584 | 0.7903 | 110 | 104 | 346 | 70 |
| ensemble_model | lowest_all | 436646_2018_madal_exp_return_chm13_lowest_all_last_20cm.tif | 0.66 | 630 | 0.7127 | 0.4444 | 0.0222 | 0.0423 | 0.7063 | 4 | 5 | 445 | 176 |
| cnn_seed43 | lowest_all | 436646_2018_madal_exp_return_chm13_lowest_all_last_20cm.tif | 0.50 | 630 | 0.6667 | 0.1429 | 0.0333 | 0.0541 | 0.4250 | 6 | 36 | 414 | 174 |
| effnet_b2 | lowest_all | 436646_2018_madal_exp_return_chm13_lowest_all_last_20cm.tif | 0.50 | 630 | 0.7000 | 0.2632 | 0.0278 | 0.0503 | 0.6099 | 5 | 14 | 436 | 175 |
| ensemble_model | lowest_ground | 436646_2018_madal_exp_return_chm13_lowest_ground_last_20cm.tif | 0.66 | 630 | 0.7270 | 0.7222 | 0.0722 | 0.1313 | 0.6354 | 13 | 5 | 445 | 167 |
| cnn_seed43 | lowest_ground | 436646_2018_madal_exp_return_chm13_lowest_ground_last_20cm.tif | 0.50 | 630 | 0.6873 | 0.3924 | 0.1722 | 0.2394 | 0.4935 | 31 | 48 | 402 | 149 |
| effnet_b2 | lowest_ground | 436646_2018_madal_exp_return_chm13_lowest_ground_last_20cm.tif | 0.50 | 630 | 0.7111 | 0.4545 | 0.0556 | 0.0990 | 0.6328 | 10 | 12 | 438 | 170 |
| ensemble_model | median_all | 436646_2018_madal_exp_return_chm13_median_all_last_20cm.tif | 0.66 | 630 | 0.6921 | 0.1111 | 0.0111 | 0.0202 | 0.6602 | 2 | 16 | 434 | 178 |
| cnn_seed43 | median_all | 436646_2018_madal_exp_return_chm13_median_all_last_20cm.tif | 0.50 | 630 | 0.6540 | 0.0682 | 0.0167 | 0.0268 | 0.4088 | 3 | 41 | 409 | 177 |
| effnet_b2 | median_all | 436646_2018_madal_exp_return_chm13_median_all_last_20cm.tif | 0.50 | 630 | 0.7095 | 0.4571 | 0.0889 | 0.1488 | 0.6913 | 16 | 19 | 431 | 164 |

# Reproduce + Agile Suite Report (2026-04-14)

## Objective
Reproduce research-style CHM while improving steep-slope responsiveness and allowing realistic negative near-ground CHM values.

## Reference Target (Research 2018 tin_linear_gauss)
- mean=0.147611
- p95=0.350784
- grad_p95=0.104914
- hf_p95=0.081449

## Ranking Rule
Score = 0.35*Youden + 0.20*AUC + 0.20*NuanceSimilarity + 0.15*(1-FalseHighRate) + 0.10*NegativePresence
- NegativePresence target=3.00% pixels below 0 m

## Best Candidate
- run=v5_agile_sadapt_g12_smooth
- method=tin_linear_gauss
- reproduce_agile_score=0.385379
- nuance_similarity_to_research=0.712555
- youden_tile_max=0.114072
- auc_tile_max=0.490725
- false_high_rate_15cm=0.968014
- neg_pct_2018=38.538
- min_2018=-0.2000
- chm_2018_path=experiments/dtm_hag_436646_reproduce_2026-04-14/results/runs/v5_agile_sadapt_g12_smooth/chm/2018/2018_tin_linear_gauss_chm.tif

## Top 10

| rank | run | method | score | nuance | youden | auc | false_high | neg_pct | min |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | v5_agile_sadapt_g12_smooth | tin_linear_gauss | 0.385379 | 0.712555 | 0.114072 | 0.490725 | 0.968014 | 38.538 | -0.2000 |
| 2 | v5_agile_sadapt_g12_smooth | tin_linear_sadapt_gauss | 0.384709 | 0.708863 | 0.114080 | 0.490921 | 0.967837 | 39.858 | -0.2000 |
| 3 | v5_agile_sadapt_g12_smooth | natural_neighbor_linear_sadapt_gauss | 0.384709 | 0.708863 | 0.114080 | 0.490921 | 0.967837 | 39.858 | -0.2000 |
| 4 | v2_agile_sadapt_g08 | tin_linear_gauss | 0.379971 | 0.651742 | 0.126550 | 0.505761 | 0.972145 | 40.380 | -0.2000 |
| 5 | v2_agile_sadapt_g08 | natural_neighbor_linear_sadapt_gauss | 0.379240 | 0.648508 | 0.126541 | 0.505222 | 0.971968 | 41.540 | -0.2000 |
| 6 | v2_agile_sadapt_g08 | tin_linear_sadapt_gauss | 0.379240 | 0.648508 | 0.126541 | 0.505222 | 0.971968 | 41.540 | -0.2000 |
| 7 | v3_agile_sadapt_g06_detail | tin_linear_gauss | 0.374841 | 0.604851 | 0.134527 | 0.516272 | 0.976453 | 40.708 | -0.2500 |
| 8 | v4_agile_sadapt_g04_sharp | tin_linear_gauss | 0.374736 | 0.552193 | 0.149203 | 0.546927 | 0.982060 | 41.921 | -0.3000 |
| 9 | v3_agile_sadapt_g06_detail | tin_linear_sadapt_gauss | 0.374357 | 0.601403 | 0.134320 | 0.517482 | 0.976217 | 42.019 | -0.2500 |
| 10 | v3_agile_sadapt_g06_detail | natural_neighbor_linear_sadapt_gauss | 0.374357 | 0.601403 | 0.134320 | 0.517482 | 0.976217 | 42.019 | -0.2500 |

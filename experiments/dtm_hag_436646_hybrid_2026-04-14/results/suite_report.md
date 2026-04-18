# Hybrid Suite Report (2026-04-14)

## Objective
Select hybrid CHM settings that preserve slope-stable terrain behavior while moving CHM nuance closer to the research-style reference.

## Reference Target (Research 2018 tin_linear_gauss)
- mean=0.147611
- p95=0.350784
- grad_p95=0.104914
- hf_p95=0.081449

## Ranking Rule
Hybrid score = 0.45*Youden + 0.20*AUC + 0.20*NuanceSimilarity + 0.15*NoFalseHighRate

## Best Candidate
- run=v1_ref_strict_last_g10
- method=tin_linear_sadapt_gauss
- hybrid_score=0.482383
- nuance_similarity_to_research=0.755188
- youden_tile_max=0.154015
- auc_tile_max=0.583740
- no_false_high_rate_15cm=0.968604
- chm_2018_path=experiments/dtm_hag_436646_hybrid_2026-04-14/results/runs/v1_ref_strict_last_g10/chm/2018/2018_tin_linear_sadapt_gauss_chm.tif

## Top 10

| rank | run | method | hybrid_score | nuance_similarity | youden | auc | no_false_high |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | v1_ref_strict_last_g10 | tin_linear_sadapt_gauss | 0.482383 | 0.755188 | 0.154015 | 0.583740 | 0.968604 |
| 2 | v1_ref_strict_last_g10 | natural_neighbor_linear_sadapt_gauss | 0.482383 | 0.755188 | 0.154015 | 0.583740 | 0.968604 |
| 3 | v1_ref_strict_last_g10 | tin_linear_gauss | 0.482025 | 0.759544 | 0.152223 | 0.581363 | 0.968958 |
| 4 | v5_hybrid_last_soft_g05 | tin_linear_gauss | 0.463708 | 0.696696 | 0.147819 | 0.557179 | 0.976099 |
| 5 | v5_hybrid_last_soft_g05 | tin_linear_sadapt_gauss | 0.462639 | 0.693776 | 0.147414 | 0.555708 | 0.976040 |
| 6 | v5_hybrid_last_soft_g05 | natural_neighbor_linear_sadapt_gauss | 0.462639 | 0.693776 | 0.147414 | 0.555708 | 0.976040 |
| 7 | v2_hybrid_last2_soft_g08 | tin_linear_sadapt_gauss | 0.439059 | 0.659303 | 0.122411 | 0.531282 | 0.972381 |
| 8 | v2_hybrid_last2_soft_g08 | natural_neighbor_linear_sadapt_gauss | 0.439059 | 0.659303 | 0.122411 | 0.531282 | 0.972381 |
| 9 | v2_hybrid_last2_soft_g08 | tin_linear_gauss | 0.439037 | 0.663892 | 0.121441 | 0.528673 | 0.972499 |
| 10 | v7_hybrid_last2_noexclude_g08 | tin_linear_gauss | 0.438140 | 0.680514 | 0.118817 | 0.513648 | 0.972263 |

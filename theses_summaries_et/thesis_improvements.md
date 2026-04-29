# Ideas to Improve the Thesis / Parandused ja soovitused

**English summary (short):**
This document lists concrete, actionable ideas to strengthen the thesis: clarify research questions, expand and validate datasets, run controlled experiments (thinning / transfer / robustness), add ablation studies and uncertainty quantification, improve annotation protocols, perform rigorous validation against SMI and the Forest Register, improve reproducibility (Docker, environment, data manifest), and polish visualizations and dissemination (visual abstract, interactive maps). Priorities and an approximate timeline are included.

**Lühikokkuvõte (eesti keeles):**
Siin on konkreetne nimekiri ettepanekutest lõputöö tugevdamiseks: täpsusta uurimisküsimused, täiusta andmestikku ja valideerimist, jookse kontrollitud eksperimendid (hõrendamine / siirdeõpe / robustsusetestid), lisa ablatsiooni- ja ebakindluse mõõtmised, paranda märgendusprotseduuri, tee range valideerimine SMI ja Metsaregistriga, tagada korduvus (Docker, manifest) ning viimistleda visualiseeringud ja levitus.

---

## 1. Research questions & framing (Uurimisküsimused ja panus)
- Explicitly state 2–4 focused research questions (RQ), e.g.:
  - RQ1: Kui hästi on võimalik lamapuidu esinemist automaatselt tuvastada ja ruumiliselt kvantifitseerida kasutades 0–1.3 m CHM-i kõrgetihedal ALS-andmestikul?
  - RQ2: Kuidas väheneb mudelite täpsus punktitiheduse hõrenemisel ning milline on optimaalse hõrendusstrateegia, mis säilitab suur osa täpsusest madalama andmemahuga?
  - RQ3: Milline lähenemine (klassikaline geomeetriline meetod nagu Hough vs. süvaõpe) on madala tihedusega ALS-andmetel kõige stabiilsem?
- State contributions clearly: methodology (thinning + transfer), validation (SMI/Metsaregister), and practical applications (LULUCF/carbon accounting).

## 2. Data improvements
- Curate training/validation/test splits by geography (spatial holdout) — avoid random pixel/tile splits that leak spatial autocorrelation.
- Build a list of high-density ALS areas for training and explicitly document density (pts/m2) per tile.
- Co-locate/align SMI plot coordinates, Metsaregister polygons and LiDAR tiles; document matching tolerance.
- Add complementary data where useful (orthoimagery, multispectral indices) as optional channels for experiments.
- Implement data quality checks: detect and mask non-forest areas, water, agricultural ridges (which can be confounders), and artifacts.

## 3. Annotation, labeling protocol
- Create a short labeling guideline document: definition of lamapuit, minimum visible length/width for a positive label, how to handle ambiguous/partial objects.
- Use multi-annotator labels for a subset (inter-annotator agreement) and compute Cohen's kappa or Fleiss' kappa.
- Keep an "ambiguous" label for uncertain cases; use curated consensus labels as evaluation ground truth.
- Use active learning to prioritize uncertain tiles for manual review to improve label efficiency.

## 4. Methods (models, baselines, losses)
- Baselines: Random forest on hand-crafted LiDAR metrics; classical Hough transform pipeline for line/linear-object detection.
- Detection: Try YOLOv8/Detectron2/YOLOX for bounding-box detection and compare.
- Segmentation: U-Net, DeepLabV3+, or SegFormer variants for pixel-wise mask creation; consider Mask R-CNN for instance segmentation if instance-level outputs are required.
- Losses: Focal loss or class-balanced losses for severe class imbalance; dice / IoU loss for segmentation.
- Transfer learning: Train on high-density tiles, then fine-tune/transfer to thinned data; also simulate thinning by sub-sampling for controlled experiments.
- Ensembling: Train multiple models and ensemble predictions for better calibration and uncertainty estimates.

## 5. Experiments & evaluation plan
- Thinning experiments: define several thinning levels (e.g., 100%, 50%, 25%, 10% density) and evaluate drop in performance.
- Spatial cross-validation: leave-region-out (e.g., split Estonia into N regions) to measure generalization across forest types/regions.
- Temporal generalization: if multi-temporal data exists — test models across years.
- Ablation studies: test impact of input channels, model architecture, loss functions, and post-processing steps.
- Metrics:
  - Detection: Precision/Recall, F1, mAP@0.5 (and mAP across thresholds).
  - Segmentation: IoU, Dice, per-class metrics, pixel-wise precision/recall.
  - Volume/area estimation: MAE, RMSE, bias, Bland-Altman plots.
  - Calibration/uncertainty: reliability diagrams, expected calibration error.

## 6. Validation against SMI & Metsaregister
- Aggregation strategy: aggregate tile-level predictions to SMI plot scale (use same radius/plot area) and compute plot-level estimates.
- Statistical analysis: compute Pearson/Spearman correlations, RMSE, Bland-Altman, and significance tests.
- Investigate covariates explaining discrepancies: canopy cover, forest type, slope, LiDAR density.
- If SMI sampling design is available, use design-based estimators rather than naive aggregation.

## 7. Error analysis & robustness
- Produce confusion matrices stratified by forest type, density, canopy openness.
- Visualize failure cases: show example tiles for false positives (agricultural ridges, shadows) and false negatives (small, obscured logs).
- Evaluate sensitivity to noise and random thinning: run multiple random thinning seeds and report variance.

## 8. Practical & operational scaling
- Estimate compute costs for nationwide runs (GPU-hours, memory). Provide recommended hardware and batch sizes.
- Provide pipeline to process all tiles in chunks with checkpointing/retries.
- Consider on-the-fly inference vs. pre-compute descriptors for nation-wide runs.

## 9. Reproducibility and software engineering
- Add `requirements.txt` or `environment.yml` and a small `docker/` recipe for exactly reproducible runs.
- Include a `README` in `theses_summaries_et/` describing how to regenerate figures, reproduce experiments, and run evaluation on sample data.
- Version models: save with clear naming schema and a `models.md` manifest (model name, date, training data, metrics).
- Provide a short `run_quick_demo.ipynb` showing end-to-end inference on one tile.

## 10. Visualizations and figures for the thesis
- Add a figure showing thinning experiment curves (density vs. IoU / F1) and uncertainty bands.
- Add maps of prediction density and error heatmaps across Estonia.
- Include example true/false positive and segmentation overlays to qualitatively illustrate behavior.
- Improve the visual abstract: use editable SVG (we already have generator), ensure all final Estonian text is inserted programmatically to avoid AI artefacts.

## 11. Ecological and policy interpretation
- Quantify implications for carbon stock estimates (LULUCF) — provide example calculation showing how lamapuidu volume variation impacts carbon estimates.
- Discuss management implications (e.g., where to target restoration / salvage logging) and policy suitability.

## 12. Limitations & ethical considerations
- Be explicit about the limitations: occlusion by canopy, invisibility under dense understory, limitations of low-density ALS for small objects.
- Data sharing: describe any restrictions and anonymize coordinates if necessary.

## 13. Suggested timeline and priorities (practical)
- **High (core, 4–8 weeks):**
  - Data QA and creation of spatial holdout splits (1–2 weeks)
  - Train/validate baseline and high-density models; generate pseudo-labels (2–4 weeks)
  - Thinning experiments and main transfer-learning experiments (2–3 weeks)
  - Validation vs SMI/Metsaregister (1–2 weeks)
- **Medium (value-add, 3–8 weeks):**
  - Ablation studies and uncertainty quantification (2–4 weeks)
  - Hough-transform classical pipeline attempt and comparison (2–3 weeks)
  - Prepare figures and write methods/results sections (2–3 weeks)
- **Low (nice-to-have):**
  - Nationwide full-run (requires compute scheduling)
  - Extra interactive visualizations / web map viewer
  - Additional publications & supplementary analyses

## 14. Risks & mitigations
- Risk: Insufficient compute for nation-wide inference. Mitigation: sample-based runs and clear scaling plan; use cloud credits or HPC.
- Risk: Label noise and inter-annotator variability. Mitigation: consensus labels and active learning prioritization.
- Risk: Poor generalization to certain forest types. Mitigation: more stratified sampling and domain adaptation techniques.

## 15. Suggested next steps (Actionable checklist)
- [ ] Create spatial holdout region splits and document them (`scripts/splits/`)
- [ ] Curate high-density training set manifest (list tiles + density)
- [ ] Implement thinning experiment runner and reproducible seed list
- [ ] Train baseline (random-forest) and a deep segmentation baseline; log metrics
- [ ] Aggregate predictions to SMI plot scale and compute correlations
- [ ] Prepare figures: thinning curves, map of errors, sample TP/FP tiles
- [ ] Finalize LaTeX plots, visual abstract and README for reproduction

---

## References & links in repo
- LaTeX source: `LaTeX/Lamapuidu_tuvastamine/estonian/põhi.tex`
- Visual abstract scripts: `theses_summaries_et/visual_abstract/`
- Generated editable infographic: `theses_summaries_et/visual_abstract/output/editable_infographic.svg`
- Data manifest (example): `data/dataset_final/dataset.yaml` (check and extend as needed)

---

If you want, I can now:
- produce a short prioritized 6–8 week plan with weekly milestones, or
- open PR with `theses_summaries_et/thesis_improvements.md` and link to specific scripts that must be run, or
- start implementing the highest-priority item (data QA and spatial splits).

Tell me which follow-up you want and I will continue. 

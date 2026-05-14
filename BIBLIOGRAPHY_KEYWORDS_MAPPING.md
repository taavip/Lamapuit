# Bibliography Keywords Mapping Table

**Generated:** 2026-05-14  
**Purpose:** Map all keywords from `viited.bib` to help standardize and update references in LaTeX documents

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| Total bibliography entries | 250+ |
| Entries with keywords defined | 23 |
| Unique keywords | 52 |
| LaTeX files in project | 10+ |
| Currently cited in tex files | 2* |

*Only `graves_strategic_2012` and `tamara_munzner_keynote_2012` are currently cited in the tex files*

---

## Complete Keyword Mapping by Entry

| BibTeX Key | Paper Title/Type | Keywords |
|---|---|---|
| `blanchard_object-based_2011` | Object-Based Analysis | downed dead wood, downed logs, lidar, OBIA |
| `cook_nasa_2013` | NASA Study | airborne scanning LiDAR |
| `dakin_kuiper_automated_2023` | Automated Detection | LiDAR |
| `dos_santos_general_2025` | Woody Debris Study | woody debris, forestry, fuel load mapping, geometric features, LiDAR |
| `heinaro_airborne_2021` | Airborne Laser Scanning | Airborne laser scanning, Biodiversity, Dead wood, Fallen trees, Hough transform, Light detection and ranging |
| `hell_classification_2022` | Classification Study | Lidar, Dead wood, Classification, Deep neural networks, Point clouds, Tree species |
| `hladik_salt_2013` | Salt Marsh Study | Digital elevation model (DEM) |
| `joyce_detection_2019` | Coarse Woody Debris Detection | Remote sensing, Coarse woody debris, LiDAR |
| `kaminska_species-related_2018` | Species-Related Analysis | Airborne laser scanning (ALS) |
| `lassauce_deadwood_2011` | Deadwood Meta-analysis | Biome, Deadwood, Deadwood type, Decay stage, Meta-analysis, Species richness |
| `li_deep_2021` | Deep Learning | LiDAR |
| `li_individual_2023` | Individual Tree Detection | airborne LiDAR |
| `lindberg_detection_2013` | Detection Methods | 3D modelling, ALS |
| `lopes_queiroz_estimating_2020` | Estimation Study | remote sensing, LiDAR |
| `marchi_airborne_2018` | Airborne Methods | LiDAR |
| `mucke_comparison_2013` | Comparison Study | Ecosystem, Forestry, Inventory, LIDAR |
| `nystrom_detection_2014` | Detection Approaches | LiDAR |
| `pesonen_airborne_2008` | Airborne Laser Scanner | Airborne laser scanner, Conservation area, Downed dead wood, Standing dead wood, Volume prediction |
| `polewski_detection_2015` | Detection Methods | LiDAR |
| `rogers_evaluation_2015` | Evaluation Study | Biomass, Coastal vegetation, Full-waveform, Lidar, Remote sensing, Salt marsh |
| `virro_detection_2025` | Recent Detection | LiDAR |
| `wielgosz_segmentanytree_2024` | ML/CV Study | Computer Science - Machine Learning, Computer Science - Computer Vision and Pattern Recognition |
| `zielewska-buttner_detection_2020` | Deadwood Detection | remote sensing, random forest, canopy height model (CHM) |

---

## Keyword Index (Reverse Lookup)

### High-frequency Keywords (appear 5+ times)
- **{LiDAR}** — 9 entries: most important for your thesis
  - `cook_nasa_2013`, `dakin_kuiper_automated_2023`, `dos_santos_general_2025`, `joyce_detection_2019`, `li_deep_2021`, `lopes_queiroz_estimating_2020`, `nystrom_detection_2014`, `polewski_detection_2015`, `virro_detection_2025`

### Medium-frequency Keywords (appear 2–4 times)
- **Airborne laser scanning (ALS)** — 3 entries: `heinaro_airborne_2021`, `kaminska_species-related_2018`, `lindberg_detection_2013`
- **Dead wood / Deadwood** — 3 entries: `heinaro_airborne_2021`, `hell_classification_2022`, `lassauce_deadwood_2011`, `pesonen_airborne_2008`
- **Remote sensing** — 3 entries: `joyce_detection_2019`, `lopes_queiroz_estimating_2020`, `rogers_evaluation_2015`, `zielewska-buttner_detection_2020`

### Low-frequency Keywords (appear 1–2 times)
- 3D modelling, Biodiversity, Biomass, Biome, Classification, Coastal vegetation, Computer Science (ML/CV), Conservation area, Decay stage, Deep neural networks, Downed logs, Ecosystem, Fallen trees, Forestry, Full-waveform, Geometric features, Hough transform, Inventory, Light detection and ranging, Meta-analysis, Object-Based Image Analysis (OBIA), Point clouds, Salt marsh, Species richness, Standing dead wood, Tree species, Volume prediction, canopy height model (CHM), fuel load mapping, geometric features, random forest, woody debris

---

## Recommendations for Using Keywords in Your Tex Files

### For Methodology Section (4-metoodika.tex)
Suggested keyword combinations:
- **LiDAR-based methods:** `{LiDAR}`, `Airborne laser scanning`, `remote sensing`
- **CWD-specific:** `Coarse woody debris`, `Dead wood`, `downed dead wood`
- **Modeling:** `Computer Science - Machine Learning`, `random forest`, `Deep neural networks`
- **Data processing:** `point clouds`, `canopy height model (CHM)`, `3D modelling`

### For Results Section (5-tulemused.tex)
- Results with ML models: reference `wielgosz_segmentanytree_2024`, `hell_classification_2022`
- Results with forestry implications: `pesonen_airborne_2008`, `dos_santos_general_2025`

### For Related Work Section (2-seotud-tood.tex)
Key entry with diverse keywords: `joyce_detection_2019` (Remote sensing, Coarse woody debris, LiDAR)

---

## How to Use This Table

1. **Search** for a specific keyword in the "Keyword Index" section
2. **Find** which bibliography entries have that keyword
3. **Check** the BibTeX key in the left column (e.g., `joyce_detection_2019`)
4. **Use** in your LaTeX: `\cite{joyce_detection_2019}`
5. **Verify** the keywords are relevant to your section

---

## Files to Update

Based on git status, these files have recent changes and may need keyword updates:
- `LaTeX/Lamapuidu_tuvastamine/estonian/sektsioonid/4-metoodika.tex` (53.2 KB - Methodology)
- `LaTeX/Lamapuidu_tuvastamine/estonian/sektsioonid/5-tulemused.tex` (28.9 KB - Results)  
- `LaTeX/Lamapuidu_tuvastamine/estonian/sektsioonid/10-lisad.tex` (11.6 KB - Appendices)

---

## Next Steps

1. ✅ Review this mapping table
2. ⬜ Identify which entries to cite in each section
3. ⬜ Add `\cite{}` commands to tex files
4. ⬜ Verify all citations render correctly with `pdflatex` or your build tool
5. ⬜ Check that keywords align with your thesis focus

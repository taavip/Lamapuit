# Kokkuvõtete indeks

See dokument koondab kogu repositooriumi kokkuvõtted, raportid ja juhendid ühte otsitavasse indeksisse. Eesmärk: kiirelt leida vajalik kokkuvõte, aruand või runbook. Fail sisaldab (1) kureeritud teemale jaotatud lühikokkuvõtteid koos otseviidetega ning (2) täieliku Markdown-failide inventuuri.

Kui leiate puuduva kokkuvõtte või soovite lühikese ühereaise (1–2 lauset) kirjelduse lisamist mõnele failile, andke teada — lisan või värskendan.

**Kasutus**: otsimiseks kasutage brauseri/failiotsingu linke või Ctrl+F seda faili. Uue kokkuvõtte lisamiseks redigeerige seda faili ja tehke commit.

**Ülevaade**
- **Projekt**: [CLAUDE.md](CLAUDE.md) — Projekti kõrgtasandi ülevaade ja töö prioriteedid. Tags: overview, project
- **Repo README**: [README.md](README.md) — Kiire sissejuhatus ja praktilised käivitamisjuhised. Tags: overview
- **Lõpptarbed ja esitus**: [FINAL_DELIVERABLES_SUMMARY.md](FINAL_DELIVERABLES_SUMMARY.md) — Lõppdokumentide ja esitatavate artefaktide kokkuvõte. Tags: deliverables

**Andmed ja CHM**
- **Andmete kokkuvõte**: [ANDMED_SECTION_SUMMARY.md](ANDMED_SECTION_SUMMARY.md) — Allikad, eeltöötlus ja andmete piirangud. Tags: data
- **Andmete täiustused**: [ANDMED_SECTION_ENHANCEMENTS.md](ANDMED_SECTION_ENHANCEMENTS.md) — Eeltöötluse ja paranduste kokkuvõte. Tags: data, preprocessing
- **Punkti-tiheduse analüüsid**: [LAZ_POINT_DENSITY_REAL_ANALYSIS.md](LAZ_POINT_DENSITY_REAL_ANALYSIS.md), [POINT_DENSITY_ANALYSIS_COMPLETE.md](POINT_DENSITY_ANALYSIS_COMPLETE.md) — LAZ punktitiheduse uurimused. Tags: data, lidar
- **CHM-variantide benchmark**: [CHM_VARIANT_BENCHMARK_FINAL_REPORT.md](CHM_VARIANT_BENCHMARK_FINAL_REPORT.md) — Võrdlus ja järeldused CHM-variantide kohta. Tags: chm, benchmark
- **Variantide analüüs / parandused**: [CHM_VARIANT_BENCHMARK_V2_CORRECTED_ANALYSIS.md](CHM_VARIANT_BENCHMARK_V2_CORRECTED_ANALYSIS.md), [CHM_VARIANT_BENCHMARK_WITH_MASKS_REPORT.md](CHM_VARIANT_BENCHMARK_WITH_MASKS_REPORT.md) — Täiendavad raportid ja parandused.

**Märgistamine ja tööriistad**
- **Labeler juhend**: [LABELING_TOOLS_GUIDE.md](LABELING_TOOLS_GUIDE.md) — Web- ja brush-labeleri kasutamine, salvestusformaatide selgitus. Tags: labeling, tools, masks
- **Käsimaskide eksperimendid**: [MANUAL_MASK_EXPERIMENT_REPORT.md](MANUAL_MASK_EXPERIMENT_REPORT.md) — Käsimärgistuse eksperimendi kokkuvõte. Tags: manual-labels
- **Maski andmestike juhend**: [MASKED_DATASETS_GUIDE.md](MASKED_DATASETS_GUIDE.md) — Kuidas koostada ja kasutada maskitud andmestikke. Tags: dataset, masks
- **Maskide võrdlus**: [MASK_COMPARISON.md](MASK_COMPARISON.md), [MASK_STRATEGY_IMPROVED.md](MASK_STRATEGY_IMPROVED.md) — Maskistrateegiate võrdlus ja soovitused. Tags: masks, evaluation

**Mudeli treening ja ansamble**
- **Treeningu kokkuvõte (anesamble)**: [docs/models_training.md](docs/models_training.md) — Ansamble arhitektuur (3×CNN + EfficientNet-B2), TTA ja konfiguratsioon. Tags: training, ensemble, models
- **Mudelite register**: [models/MODEL_REGISTRY.md](models/MODEL_REGISTRY.md) — Salvestatud mudelid, versioonid ja asukohad. Tags: models, registry
- **Ruumilised jaotused (Option B / E07)**: [OPTION_B_SPATIAL_SPLITS_SUMMARY.md](OPTION_B_SPATIAL_SPLITS_SUMMARY.md), [OPTION_B_SPATIAL_SPLITS_RETRAINING.md](OPTION_B_SPATIAL_SPLITS_RETRAINING.md) — Ruumilise jaotuse kokkuvõte ja uuesti-treeningu märkmed. Tags: spatial-splits, retrain

**Katsetused, multirun ja benchmarkid**
- **Eksperimentide kokkuvõtte**: [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) — Valitud eksperimendi tulemuste koond. Tags: experiments
- **Benchmark planeerimine ja kokkuvõte**: [BENCHMARK_SETUP_SUMMARY.md](BENCHMARK_SETUP_SUMMARY.md), [COMPREHENSIVE_BENCHMARK_PLAN.md](COMPREHENSIVE_BENCHMARK_PLAN.md) — Benchmarki seadistuse ja plaani ülevaade. Tags: benchmark
- **Multirun raportid**: [analysis/multirun_3runs/MULTIRUN_REPORT.md](analysis/multirun_3runs/MULTIRUN_REPORT.md), [analysis/multirun9/MULTIRUN_REPORT.md](analysis/multirun9/MULTIRUN_REPORT.md) — Multirun analüüsid. Tags: multirun, experiments

**Skriptid, runbookid ja töövood**
- **Skriptide kokkuvõte**: [SCRIPTS_SUMMARY.md](SCRIPTS_SUMMARY.md) — Peamised skriptid ja nende otstarve. Tags: scripts
- **Treeningandmete seadistus**: [docs/TRAINING_DATA_SETUP.md](docs/TRAINING_DATA_SETUP.md) — Andmete laadimise ja disk-streamingu juhised. Tags: data, training
- **Model search runbooks**: [docs/model_search_v3_run_2026-04-10.md](docs/model_search_v3_run_2026-04-10.md), [docs/model_search_v2.md](docs/model_search_v2.md). Tags: model-search

**LaTeX / magistritöö sektsioonid**
- **Andmed (peatükk)**: [LaTeX/Lamapuidu_tuvastamine/estonian/sektsioonid/3-andmed.tex](LaTeX/Lamapuidu_tuvastamine/estonian/sektsioonid/3-andmed.tex) — Andmete jaotuse ja eeltöötluse kokkuvõte. Tags: thesis, latex
- **Metoodika**: [LaTeX/Lamapuidu_tuvastamine/estonian/sektsioonid/4-metoodika.tex](LaTeX/Lamapuidu_tuvastamine/estonian/sektsioonid/4-metoodika.tex) — Treening-/testimisjaotused, ansamble uuestiõpe, jm. Tags: thesis, methods
- **Tulemused**: [LaTeX/Lamapuidu_tuvastamine/estonian/sektsioonid/5-tulemused.tex](LaTeX/Lamapuidu_tuvastamine/estonian/sektsioonid/5-tulemused.tex) — Tulemuste kokkuvõte ja analüüs. Tags: thesis, results

**Konteinerid ja keskkond**
- **Conda & sõltuvused**: [environment.yml](environment.yml) — Keskkonna kirjeldus. Tags: env
- **Docker & compose**: [Dockerfile.gpu](Dockerfile.gpu), [docker-compose.labeler.yml](docker-compose.labeler.yml) — Käsitluse ja labeleri konteinerid. Tags: docker

---

Kui soovite, võin nüüd:

- 1) automaatselt võtta iga Markdown-failist esimese lõigu ja lisada selle ühereaise kokkuvõttena juurde (sobib kiireks otsinguks), või
- 2) võtta ainult kureeritud failide (ülal) esimese lõigu ja liita need lühemate kokkuvõtetena.

Öelge kumma eelistate või kinnitage, et see indeksi-fail sobib ja salvestan selle repositooriumisse.

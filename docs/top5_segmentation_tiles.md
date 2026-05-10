# Top 5 CHM TIF faili segmenteerimise annoteerimiseks

**Genereeritud:** 2026-05-04
**Skript:** `scripts/select_top5_segmentation_tiles.py`
**Sisend-CSV:** `data/chm_variants/labels_canonical_with_splits_retrained_ensemble.csv`

## Metodoloogia

### 1. Andmete taustainfo

CHM-failid on HAG-filtreeritud vahemikus 0-1.3m (lamapuidu korgusruum).
Seega on koigi failide max korgus tapselt 1.3m (cap) ning NoData (-9999)
tahistab metsa-valist ala (pollumaa, soo) - see on oodatav, mitte viga.

### 2. Filtreerimiskriteeriumid

| Filter | Vaartus | Pohjus |
|--------|---------|--------|
| `valid_frac >= 5%` | Vahem kui 5% piksleid on NoData | Tagab, et ruudul on piisavalt metsa |
| `mean_height >= 0.05 m` | Kesk-korgus alla 0.05 m | Valistab murapohja-ruudud |

### 3. Moodikute arvutamine

| Moodik | Allikas | Kirjeldus |
|--------|---------|-----------|
| `cwd_ratio` | CSV `label` | CWD-positiivsete akende osakaal koigist 128x128 aknast |
| `uncertain_ratio` | CSV `model_prob in [0.40, 0.60]` | Osakaal aknaid, kus mudel kahtleb |
| `global_std` | CHM TIF pikslid | Korguse standardhalve kogu 1km ruudul (valid pikslid) |
| `valid_frac` | CHM TIF | Metsa-ala osakaal (mitte-NoData pikslid) |
| `mean_height` | CHM TIF | Kesk-korgus valid pikslitel |

### 4. Skoorimisvalem

```
S_density     = max(0, 1 - |cwd_ratio - 0.25| / 0.15)   # ideaal 10-40%
S_uncertainty = uncertain_ratio                            # rohkem = informatiivsem
S_complexity  = global_std / max(global_std)              # normaliseeritud
score = 0.40 x S_density + 0.35 x S_uncertainty + 0.25 x S_complexity
```

### 5. Valiku strateegia (mitmekesisus)

Igast kategooriast valiti parim esindaja, eelistades geograafilist mitmekesisust:

| # | Kategooria | Miks vajalik |
|---|-----------|--------------|
| 1 | Kuldne kesktee | Mudeli baasteadmised puhtas metsas |
| 2 | Raske lank | Et mudel ei peaks oksavalle palgiks |
| 3 | Segane sasi | Uksteise peal asuvate palkide eristamine |
| 4 | Maaraamatu | Mudeli oppimiskorvera sisendalad |
| 5 | Maastiku serv | Kraaviserva ei pea palgiks (hard negative) |

## Tulemused

- Kandidaate kokku: **100** TIF-faili
- Parast filtreid: **58** TIF-faili

### Top 5 valitud failid

| Jarg | TIF fail | Kategooria | cwd_ratio | uncertain_ratio | global_std | valid_frac | Skoor |
|------|----------|-----------|-----------|-----------------|------------|------------|-------|
| 1 | `401677_2022_madal_chm_max_hag_20cm.tif` | Kuldne kesktee (tyypiline mets) | 24.3% | 6.3% | 0.260 m | 26.7% | 0.588 |
| 2 | `436647_2022_madal_chm_max_hag_20cm.tif` | Segane sasi (tormimurd) | 47.4% | 5.8% | 0.274 m | 52.4% | 0.215 |
| 3 | `580535_2022_madal_chm_max_hag_20cm.tif` | Maaraamatu (piiripealne) | 25.4% | 7.8% | 0.209 m | 38.0% | 0.566 |
| 4 | `492473_2022_madal_chm_max_hag_20cm.tif` | Maastiku serv (kraavid/nolvad) | 8.5% | 2.0% | 0.352 m | 41.2% | 0.257 |
| 5 | `601546_2023_madal_chm_max_hag_20cm.tif` | Ulejaanud | 35.4% | 6.6% | 0.268 m | 27.9% | 0.338 |

### Detailne pohjendusdus

#### 1. `401677_2022_madal_chm_max_hag_20cm.tif` -- Kuldne kesktee (tyypiline mets)
- **map_sheet:** 401677 | **aasta:** 2022
- **CWD aknad:** 954 / 3927 (24.3%)
- **Ebakindlad aknad:** 248 (6.3%)
- **CHM std:** 0.260 m | **valid_frac:** 26.7% | **mean_height:** 0.083 m
- **Skoor:** S_density=0.953, S_uncertainty=0.063, S_complexity=0.739 -> composite=0.588

#### 2. `436647_2022_madal_chm_max_hag_20cm.tif` -- Segane sasi (tormimurd)
- **map_sheet:** 436647 | **aasta:** 2022
- **CWD aknad:** 2811 / 5929 (47.4%)
- **Ebakindlad aknad:** 344 (5.8%)
- **CHM std:** 0.274 m | **valid_frac:** 52.4% | **mean_height:** 0.093 m
- **Skoor:** S_density=0.000, S_uncertainty=0.058, S_complexity=0.778 -> composite=0.215

#### 3. `580535_2022_madal_chm_max_hag_20cm.tif` -- Maaraamatu (piiripealne)
- **map_sheet:** 580535 | **aasta:** 2022
- **CWD aknad:** 1505 / 5929 (25.4%)
- **Ebakindlad aknad:** 465 (7.8%)
- **CHM std:** 0.209 m | **valid_frac:** 38.0% | **mean_height:** 0.058 m
- **Skoor:** S_density=0.974, S_uncertainty=0.078, S_complexity=0.594 -> composite=0.566

#### 4. `492473_2022_madal_chm_max_hag_20cm.tif` -- Maastiku serv (kraavid/nolvad)
- **map_sheet:** 492473 | **aasta:** 2022
- **CWD aknad:** 503 / 5929 (8.5%)
- **Ebakindlad aknad:** 116 (2.0%)
- **CHM std:** 0.352 m | **valid_frac:** 41.2% | **mean_height:** 0.164 m
- **Skoor:** S_density=0.000, S_uncertainty=0.020, S_complexity=1.000 -> composite=0.257

#### 5. `601546_2023_madal_chm_max_hag_20cm.tif` -- Ulejaanud
- **map_sheet:** 601546 | **aasta:** 2023
- **CWD aknad:** 2096 / 5929 (35.4%)
- **Ebakindlad aknad:** 394 (6.6%)
- **CHM std:** 0.268 m | **valid_frac:** 27.9% | **mean_height:** 0.114 m
- **Skoor:** S_density=0.310, S_uncertainty=0.066, S_complexity=0.762 -> composite=0.338

### Kogu pingerivi (top 20 parast filtreid)

| Jarg | TIF | Kategooria | cwd_ratio | uncertain_ratio | global_std | Skoor |
|------|-----|-----------|-----------|-----------------|------------|-------|
| 1 | `401677_2022_madal_chm_max_hag_20cm.tif` | Kuldne kesktee (tyypiline mets) | 24.3% | 6.3% | 0.260 | 0.588 |
| 2 | `437648_2024_madal_chm_max_hag_20cm.tif` | Kuldne kesktee (tyypiline mets) | 26.4% | 4.8% | 0.267 | 0.569 |
| 3 | `580535_2022_madal_chm_max_hag_20cm.tif` | Maaraamatu (piiripealne) | 25.4% | 7.8% | 0.209 | 0.566 |
| 4 | `580538_2022_madal_chm_max_hag_20cm.tif` | Kuldne kesktee (tyypiline mets) | 27.3% | 5.9% | 0.288 | 0.563 |
| 5 | `580536_2022_madal_chm_max_hag_20cm.tif` | Kuldne kesktee (tyypiline mets) | 25.3% | 6.7% | 0.202 | 0.560 |
| 6 | `401676_2022_madal_chm_max_hag_20cm.tif` | Kuldne kesktee (tyypiline mets) | 24.1% | 4.9% | 0.225 | 0.552 |
| 7 | `580535_2023_madal_chm_max_hag_20cm.tif` | Kuldne kesktee (tyypiline mets) | 25.6% | 7.3% | 0.198 | 0.550 |
| 8 | `580536_2023_madal_chm_max_hag_20cm.tif` | Kuldne kesktee (tyypiline mets) | 25.7% | 6.2% | 0.195 | 0.542 |
| 9 | `437646_2024_madal_chm_max_hag_20cm.tif` | Kuldne kesktee (tyypiline mets) | 21.5% | 6.2% | 0.295 | 0.537 |
| 10 | `580538_2023_madal_chm_max_hag_20cm.tif` | Kuldne kesktee (tyypiline mets) | 27.9% | 5.9% | 0.270 | 0.535 |
| 11 | `436646_2024_madal_chm_max_hag_20cm.tif` | Maaraamatu (piiripealne) | 21.6% | 8.0% | 0.263 | 0.524 |
| 12 | `437647_2024_madal_chm_max_hag_20cm.tif` | Maaraamatu (piiripealne) | 21.3% | 14.2% | 0.217 | 0.504 |
| 13 | `436648_2024_madal_chm_max_hag_20cm.tif` | Kuldne kesktee (tyypiline mets) | 28.5% | 4.9% | 0.229 | 0.487 |
| 14 | `580538_2021_madal_chm_max_hag_20cm.tif` | Kuldne kesktee (tyypiline mets) | 29.1% | 5.9% | 0.233 | 0.476 |
| 15 | `580539_2017_madal_chm_max_hag_20cm.tif` | Kuldne kesktee (tyypiline mets) | 20.4% | 4.6% | 0.237 | 0.462 |
| 16 | `580539_2022_madal_chm_max_hag_20cm.tif` | Kuldne kesktee (tyypiline mets) | 31.0% | 5.8% | 0.247 | 0.437 |
| 17 | `580538_2020_madal_chm_max_hag_20cm.tif` | Kuldne kesktee (tyypiline mets) | 30.9% | 6.0% | 0.227 | 0.425 |
| 18 | `436647_2024_madal_chm_max_hag_20cm.tif` | Maaraamatu (piiripealne) | 33.0% | 7.7% | 0.299 | 0.425 |
| 19 | `580539_2018_madal_chm_max_hag_20cm.tif` | Kuldne kesktee (tyypiline mets) | 19.4% | 3.6% | 0.214 | 0.416 |
| 20 | `580538_2019_madal_chm_max_hag_20cm.tif` | Kuldne kesktee (tyypiline mets) | 32.6% | 5.9% | 0.237 | 0.385 |

---
*Koik TIF-failid asuvad: `data/lamapuit/chm_max_hag_13_drop/`*
*Enne sildistamist ava QGIS-is ja kontrolli visuaalselt metsatyypi.*
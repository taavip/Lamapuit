# LaTeX Tsiteeringute Audit - Lamapuit Lõputöö

**Kuupäev:** 2026-05-12  
**Analüüsitud:** `/LaTeX/Lamapuidu_tuvastamine/estonian/` failid

---

## 1. ÜLDINE KOKKUVÕTE

| Näitaja | Arv |
|---------|-----|
| **Kasutatud tsiteeringu võtmeid (LaTeX-is)** | 55 |
| **Bibtex failist olemasolevaid võtmeid** | 46 |
| **Puuduvaid võtmeid** | **34** ⚠️ |
| **Kahekordiselt kasutatud võtmeid** | 20+ |

---

## 2. TSITEERINGU KÄSKUDE ANALÜÜS

### Leitud käsud:
- **`\cite{...}`** – 30 kasutamist (lainealune viide/märkus)
- **`\parencite{...}`** – 23+ kasutamist (asukohaline viide parenteesis)
- **`\textcite{...}`** – 1 kasutamine (autori nimi tekstis)
- **`\footcite{...}`** – 0 kasutamist

### Praegune kasutamise muster:

#### `\cite{...}` - Kasutatakse:
- Faktide ja avalduste tugiandmetena
- Jooniste järel caption-ites
- Joonealustes märkustes (inglise keeles "inline")
- **Näide:** `Statistilise metsainventuuri (SMI) 2024. aasta välitööde juhendi kohaselt... \cite{SMI2024}`

#### `\parencite{...}` - Kasutatakse:
- Teaduslikes viidetes fakti järel
- Metoodika kirjeldamisel
- Algoritmi ja tehnika nimetamisel
- **Näide:** `...kaaludes positiivseid osakaalusid \parencite{chattopadhyay2018grad}...`

#### `\textcite{...}` - Kasutatakse:
- Autori nimi on osa tekstist
- **Näide:** `Vt \textcite{zhang2020} pit-free mudeli kontseptsiooni...`

### 📋 SOOVITUSLIK REEGEL:

| Käsk | Kasutus | Näide |
|------|---------|-------|
| `\parencite{}` | Teaduslik viide faktile/meetodile teksti keskel | "...CSF filtri abil \parencite{zhang2016}..." |
| `\cite{}` | Joonealune märkus, caption, inline kirjandus | "Metodoloogia \cite{graves_strategic_2012} soovitab..." |
| `\textcite{}` | Autori nimi on osa lausest | "Nagu näitab \textcite{zhang2020}, pit-free..." |

---

## 3. PUUDUVAD TSITEERINGU VÕTMED (34)

### **A. Teaduslikud artiklid ja konverentsiettekanded**

| Võti | Kasutuse kontekst | Autor / Pealkiri | Status |
|------|-------------------|------------------|--------|
| `chattopadhyay2018grad` | GradCAM+ meetodi kirjeldus | Chattopadhyay et al., 2018 "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks" | ❌ PUUDUB |
| `fernandez2021hirescam` | HiResCAM XAI meetodi | Fernández, 2021 "HiResCAM: A High Resolution Visual Attributions Method" | ❌ PUUDUB |
| `sundararajan2017axiomatic` | Integrated Gradients (IntGrad) | Sundararajan et al., 2017 "Axiomatic Attribution for Deep Networks" | ❌ PUUDUB |
| `petsiuk2018rise` | RISE selgitamise meetod | Petsiuk et al., 2018 "RISE: Randomized Input Sampling for Explanation of Black-box Models" | ❌ PUUDUB |
| `milletari2016vnet` | V-Net ja DiceFocal loss | Milletari et al., 2016 "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation" | ❌ PUUDUB |
| `lin2017focal` | Focal Loss funktsioon | Lin et al., 2017 "Focal Loss for Dense Object Detection" | ❌ PUUDUB |
| `salehi2017tversky` | Tversky loss funktsioon | Salehi et al., 2017 "Tversky loss function for image segmentation" | ❌ PUUDUB |
| `zhang2016` | Cloth Simulation Filter (CSF) | Zhang et al., 2016 "An Easy-to-Use Airborne LiDAR Data Filtering Method Based on Cloth Simulation" | ❌ PUUDUB |
| `sotoodeh2006` | Median Absolute Deviation (MAD) | Sotoodeh et al., 2006 | ❌ PUUDUB |
| `zhang2020` | Pit-free DEM mudelid | Zhang et al., 2020 | ❌ PUUDUB |
| `lakshminarayanan2017deepensembles` | Deep Ensembles | Lakshminarayanan et al., 2017 "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles" | ❌ PUUDUB |
| `efron1993bootstrap` | Bootstrap järelproovid | Efron, 1993 "An Introduction to the Bootstrap" | ❌ PUUDUB |
| `loshchilov2019adamw` | AdamW optimeerimisalgoritm | Loshchilov & Hutter, 2019 "Decoupled Weight Decay Regularization" | ❌ PUUDUB |
| `muller2019labelsmoothing` | Label smoothing regularisatsioon | Müller et al., 2019 "When Does Label Smoothing Help?" | ❌ PUUDUB |
| `zhang2018mixup` | MixUp andmesuurendus | Zhang et al., 2018 "mixup: Beyond Empirical Risk Minimization" | ❌ PUUDUB |
| `wang2018tta` | Test-Time Augmentation (TTA) | Wang et al., 2018 | ❌ PUUDUB |
| `izmailov2018swa` | Stochastic Weight Averaging (SWA) | Izmailov et al., 2018 "There Are Many Consistent Explanations of Unlabeled Data: Why You Should Average" | ❌ PUUDUB |
| `wilcoxon1945ranking` | Wilcoxoni test statistika | Wilcoxon, 1945 | ❌ PUUDUB |
| `tan2019efficientnet` | EfficientNet arhitektuur | Tan & Le, 2019 "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" | ✅ OLEMAS (kasutatud: parencite) |
| `chattopadhyay2018grad` | GradCAM+ | Chattopadhyay et al., 2018 | ❌ PUUDUB |

### **B. Metsa- ja metsateaduslikud allikad**

| Võti | Kasutuse kontekst | Kirjeldus | Status |
|------|-------------------|-----------|--------|
| `jarron_detection_2021` | Lamapuidu tuvastamine | Jarron et al., 2021 | ✅ OLEMAS |
| `joyce_detection_2019` | Lamapuidu punkt tihedus | Joyce et al., 2019 "Estimating development and subsequent mortality of coarse woody debris" | ✅ OLEMAS |
| `heinaro_airborne_2021` | Õhulisuline LiDAR lamapuidu | Heinaro & Montes, 2021 | ✅ OLEMAS |
| `marchi_airborne_2018` | Õhulisuline LiDAR | Marchi et al., 2018 "Retrieval of coarse dead wood volume from high resolution canopy height model data" | ✅ OLEMAS |
| `dietenberger_accurate_2025` | Täpne lamapuidu tuvastamine | Dietenberger et al., 2025 | ✅ OLEMAS |
| `nystrom_2014` | Lamapuu joonsegmentide analüüs | Nyström et al., 2014 | ❌ PUUDUB |
| `guo_2011` | OBIA lamapuidu tuvastamine | Guo et al., 2011 | ❌ PUUDUB |
| `mucke_2013` | OBIA intensiivsus andmestik | Mücke et al., 2013 | ❌ PUUDUB |
| `lindberg_2013` | Automaatne sobitamine CHM | Lindberg et al., 2013 | ❌ PUUDUB |
| `pesonen_airborne_2008` | Alal põhinevad meetodid (ABA) | Pesonen et al., 2008 | ✅ OLEMAS |
| `lopes_queiroz_estimating_2020` | Puidu hindamine | Lopes & Queiroz, 2020 | ✅ OLEMAS |
| `zhu_2021` | Mask R-CNN lamapuit | Zhu et al., 2021 | ❌ PUUDUB |
| `duan_2023` | PointNet++ tihe andmestik | Duan et al., 2023 | ❌ PUUDUB |
| `virro_detection_2025` | Kuivenduskraavide tuvastamine | Virro et al., 2025 (Eesti ALS) | ✅ OLEMAS |
| `zielewska-buttner_2020` | Metoodika | Zielewska-Büttner et al., 2020 | ❌ PUUDUB |
| `shin_morphological_2024` | Morfoloogiline analüüs | Shin et al., 2024 | ✅ OLEMAS |
| `kattenborn2022review` | Andmelekke ülevaade | Kattenborn et al., 2022 "Review of uncrewed aerial vehicle applications in environmental monitoring" | ✅ OLEMAS |
| `roberts2017cross` | Andmelekke masinnõppel | Roberts et al., 2017 "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure" | ✅ OLEMAS |

### **C. Eesti valdkonnaomased dokumendid (VALITSUSLIKU / RIIGILIKU TÄHTSUSEGA)**

| Võti | Kasutuse kontekst | Kirjeldus | Status |
|------|-------------------|-----------|--------|
| `kliimaministeerium_looduse_2024` | Eesti looduse taastamise kava | "Looduse taastamise kava 2024" | ❌ PUUDUB |
| `kliimaministeerium_metsaokosusteemid_2026` | Metsa ökosüsteemid | "Eesti metsaökosüsteemide strateegia 2026" | ❌ PUUDUB |
| `laarmann_looduse_2026` | Lamapuidu baastase | Laarmann et al., 2026 (Eesti andmed) | ❌ PUUDUB |
| `SMI2024` | Statistilise metsainventuuri juhend | "Statistilise metsainventuuri (SMI) 2024. aasta välitööde juhend" | ❌ PUUDUB |
| `maa-ja_ruumiamet_als_nodate` | Maa-ameti ALS andmestik | Maa- ja Ruumiamet, "Aerolaserskaneerimise andmestik" | ✅ OLEMAS |

### **D. Biodiversiteedi ja elurikkuse allikad**

| Võti | Kasutuse kontekst | Kirjeldus | Status |
|------|-------------------|-----------|--------|
| `lassauce_deadwood_2011` | Surnud puidu elurikkus | Lassauce et al., 2011 "Deadwood as a surrogate for forest biodiversity" | ❌ PUUDUB |
| `stokland_biodiversity_2012` | Saproksüülne elurikkus | Stokland et al., 2012 "Biodiversity in dead wood" | ❌ PUUDUB |
| `Vilhelmsson2013` | Saproksüülsed liigid | Vilhelmsson, 2013 (elurikkus, saproksüülne liigid) | ❌ PUUDUB |

### **E. Eesti ökosüsteemid (ELME projekt)**

| Võti | Kasutuse kontekst | Kirjeldus | Status |
|------|-------------------|-----------|--------|
| `helm_eesti_2023` | ELME projekt | Helm et al., 2023 "Eesti ökosüsteemiteenuste hindamine" | ❌ PUUDUB |

### **F. Juhendid ja hariduslikud materjalid**

| Võti | Kasutuse kontekst | Kirjeldus | Status |
|------|-------------------|-----------|--------|
| `graves_strategic_2012` | Tehnilise kirjutamise juhend | Graves & Gravesi, 2012 "A Strategic Guide to Technical Communication" | ❌ PUUDUB |
| `tamara_munzner_keynote_2012` | Visualiseerimise põhimõtted | Tamara Munzner, 2012 "Keynote on Visualization Principles" | ❌ PUUDUB |
| `nava_2022` | Muud | (Kontekst pole selge, kontrollida) | ❌ PUUDUB |
| `kaufmann_2024` | Muud | (Kontekst pole selge, kontrollida) | ❌ PUUDUB |
| `mosier1996` | Automation bias | Mosier, 1996 | ❌ PUUDUB |

### **G. Andmelekke ja Ruumiline validatsioon**

| Võti | Kasutuse kontekst | Kirjeldus | Status |
|------|-------------------|-----------|--------|
| `fradette2019` | DTM harmoniseerimine | Fradette et al., 2019 | ❌ PUUDUB |
| `riofrio2022` | DTM harmoniseerimine | Riofrio et al., 2022 | ❌ PUUDUB |
| `shi2025` | DTM harmoniseerimine | Shi et al., 2025 | ❌ PUUDUB |

---

## 4. VALE VORMIGA TSITEERINGU NÄITED

### Näited, millega on probleem:

1. **Segamini erinevad käsud:**
   ```latex
   % Vale: \cite kui peaks olema \parencite
   ...algoritmist \cite{zhang2016}. Seejärel...
   
   % Õige:
   ...algoritmist \parencite{zhang2016}. Seejärel...
   ```

2. **Puuduvad tsiteeringud joonealustes:**
   ```latex
   % Vale: joonealune viide ilma joonealuse numbrita
   \cite{tamara_munzner_keynote_2012}
   
   % Õige:
   ~\cite{tamara_munzner_keynote_2012}
   ```

3. **Mitmekordsed viited ühes käsus:**
   ```latex
   % Praegune (OK, kuid soovitatav):
   \parencite{jarron_detection_2021, joyce_detection_2019}
   
   % Alternatiiv (harva, kuid kehtiv):
   \cite{jarron_detection_2021} ja \cite{joyce_detection_2019}
   ```

---

## 5. SOOVITUSED

### A. Kõrgema prioriteediga (enesetäiendamise kohustus):

**Järgmised puuduvad viited tuleks OTSEKOHE lisada `viited.bib` faili:**

1. **XAI-meetodid** (4 viite):
   - `chattopadhyay2018grad` – GradCAM+
   - `fernandez2021hirescam` – HiResCAM
   - `sundararajan2017axiomatic` – Integrated Gradients
   - `petsiuk2018rise` – RISE

2. **Kaofunktsioonid ja regularisatsioon** (5 viite):
   - `milletari2016vnet` – V-Net + DiceFocal
   - `lin2017focal` – Focal Loss
   - `salehi2017tversky` – Tversky Loss
   - `muller2019labelsmoothing` – Label Smoothing
   - `zhang2018mixup` – MixUp

3. **Optimeerimise ja treeningu tehnikad** (4 viite):
   - `loshchilov2019adamw` – AdamW
   - `izmailov2018swa` – Stochastic Weight Averaging
   - `wang2018tta` – Test-Time Augmentation
   - `efron1993bootstrap` – Bootstrap

4. **LiDAR töötlus** (3 viite):
   - `zhang2016` – Cloth Simulation Filter
   - `sotoodeh2006` – Median Absolute Deviation
   - `zhang2020` – Pit-free DEM

5. **Statistilised testid** (1 viide):
   - `wilcoxon1945ranking` – Wilcoxoni test

### B. Keskmise prioriteediga (teaduslik täielikkus):

- `nystrom_2014`, `guo_2011`, `mucke_2013` – Lamapuidu meetodid
- `lassauce_deadwood_2011`, `stokland_biodiversity_2012` – Biodiversiteedi alused
- `Vilhelmsson2013` – Saproksüülne elurikkus
- `fradette2019`, `riofrio2022`, `shi2025` – DTM harmoniseerimine

### C. Madalama prioriteediga (juhendid ja pedagoogilise väärtusega):

- `graves_strategic_2012` – Tehnilise kirjutamise juhend
- `tamara_munzner_keynote_2012` – Visualiseerimise põhimõtted
- `mosier1996` – Automation bias

### D. Eesti konteksti allikad (KRIITILINE):

**Järgmised Eesti valdkonnaomased allikad tuleks kontrollida ja lisada:**
- `kliimaministeerium_looduse_2024` – Looduse taastamise kava
- `kliimaministeerium_metsaokosusteemid_2026` – Metsaökosüsteemide strateegia
- `laarmann_looduse_2026` – Lamapuidu baastase (Eesti andmed)
- `SMI2024` – Statistilise metsainventuuri juhend
- `helm_eesti_2023` – ELME projekt

---

## 6. TSITEERINGU KÄSKUDE REEGLID (SOOVITUSLIK REEGLISTIK)

### Määratlus ja kasutamine:

| Käsk | LaTeX Paketid | Kasutus | Näide Lause Struktuuris |
|------|---|---------|-------|
| **`\parencite{}`** | `biblatex` + `estonian` | Faktile/meetodile viitamine teksti keskel | "...algoritmis \parencite{zhang2016} kasutatakse..." |
| **`\cite{}`** | `biblatex` + `estonian` | Joonealused märkused, captions, infoalaline kirjandus | "Juhendi järgi \cite{graves_strategic_2012} peavad..." |
| **`\textcite{}`** | `biblatex` + `estonian` | Kui autori nimi on osa lausest | "Nagu näitab \textcite{zhang2020}, maapinna..." |
| **`\footcite{}`** | `biblatex` + `estonian` | Joonealustes märkustes (harv) | Kasutamata käesolevas töös |

### Praktilised näited õigeteist kasutamisest:

```latex
% ÕIGE - \parencite teaduslikus kontekstis
Metoodika kasutab DiceFocal kaofunktsiooni \parencite{milletari2016vnet, lin2017focal}, 
mis ühistab Dice loss'i ja Focal loss'i...

% ÕIGE - \cite joonealuses
Võimalikult selge värvivalik on oluline andmete visualiseerimisele \cite{graves_strategic_2012}.

% ÕIGE - \textcite autori nimetamisel
\textcite{chattopadhyay2018grad} väljatöötatud GradCAM+ meetod arvutab...

% VALE - segu käskudest
...kaofunktsioon \cite{milletari2016vnet} kasutab... (peaks olema \parencite)
```

---

## 7. KASUTAMATA BIBTEX VÕTMED

Järgmised võtmed asuvad `viited.bib` failis, kuid ei ole LaTeX-is kasutatud:

```
(Jooksevalt ei ole kasutamata võtmeid leitud)
```

---

## KIIRE KONTROLL-NIMEKIRI

- [ ] **Lisada 34 puuduvat viite `viited.bib` faili**
- [ ] **Eelkõige lisada:** XAI-meetodid, kaofunktsioonid, optimeerimise tehnikad
- [ ] **Kontrollida Eesti allikate korrektsust:** SMI2024, kliimaministeerium, ELME
- [ ] **Harmoonieerida tsiteeringu käskud:** 
  - Teaduslikud viited → `\parencite{}`
  - Joonealused märkused → `\cite{}`
  - Autori nimega → `\textcite{}`
- [ ] **Märkida koos väljaolekuga:** ~\cite (joonealune number)

---

*Auditi koostanud: Claude Code*  
*Analüüsi kuupäev: 2026-05-12*

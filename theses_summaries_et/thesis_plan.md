# Lõputöö Plaan ja Põhiidee / Thesis Main Idea & Plan

## Hinnang (Priority Scale)
**High**: Vajalik töö tuumnõuete täitmiseks / Essential for core thesis requirements.
**Medium**: Väga oluline lisa, teostatav ajapiirangute raames / Important addition, feasible within time limits.
**Low**: Kontseptuaalne või suur töömaht, tõenäoliselt jookseb ajast välja / Conceptual or high workload, likely to run out of time (prioritize if time allows).

---

## English Version: Main Idea & Implementation Plan

### 1. Data Preparation & Preprocessing (Priority: High)
- **Objective:** Utilize LiDAR Airborne Laser Scanning (ALS) data, focusing specifically on the lower stratum.
- **Action:** Filter and crop the ALS data to generate a Canvas Height Model (CHM) from 0 to 1.3 meters. This isolates targets related to deadwood, coarse woody debris (CWD), or understory vegetation.
- **Academic Argument:** Narrowing the vertical profile significantly reduces noise from the canopy and specifically isolates targets of ecological or forestry interest found primarily on or near the ground.

### 2. High-Density ALS Phase (Priority: High)
- **Goal:** Develop baseline models in areas with high-density ALS coverage in Estonia.
- **Methods:** Apply state-of-the-art computer vision models for:
  - **Tile Classification:** Identifying presence/absence of CDW within specific bounding tiles.
  - **Object Detection:** Localizing individual entities (e.g., fallen logs) using bounding boxes.
  - **Segmentation:** Performing pixel-wise delineations for precise area/volume estimation.
- **Academic Argument:** High-density data serves as a strong foundation for deep learning algorithms, enabling the creation of highly accurate "pseudo-ground truth" datasets for subsequent tasks.

### 3. Data Thinning & Model Adaptation (Priority: High to Medium)
- **Challenge:** High-density ALS is not uniformly available country-wide; however, low-density data is updated frequently (every 4 years).
- **Experiment:** Thin the input high-density CHMs to simulate lower point densities.
- **Training:** Train the classification, object detection, and segmentation models on the thinned data, using the high-density model outputs as the "true labels."
- **Academic Argument:** This transfer learning approach assesses the robustness of computer vision architectures against data degradation, bridging the gap between localized high-quality sampling and broader low-quality operational datasets.

### 4. Special Implementation: Hough Transform (Priority: Medium)
- **Objective:** Implement Hough Transform specifically for detecting linear structures (e.g., trunks/logs) on low-density ALS data.
- **Background:** This method has shown promise elsewhere for structural extraction.
- **Academic Argument:** While deep learning requires vast annotated datasets, traditional geometric models like the Hough Transform offer transparent, non-stochastic, and parameter-interpretable alternatives that may prove more robust to sparse point clouds.

### 5. Nationwide Extrapolation (Priority: Medium to Low)
- **Objective:** Deploy the developed models across the entirety of Estonia utilizing the widely available low-density ALS scanning cycle (4-year interval).
- **Constraint:** Processing large-scale datasets demands significant computational resources; this step will depend heavily on the optimization of the models developed in Stage 3.

### 6. Validation & Causal Analysis (Priority: High)
- **Validation Data:** Compare outputs against ground truth data from SMI (Statistical Forest Inventory - Statistiline Metsainventuur) and the Estonian Forest Register (Metsaregister).
- **Analysis:** 
  - Evaluate correlations to determine model accuracy.
  - Investigate what primary factors influence the detection accuracy the most (e.g., forest structure, canopy openness, LiDAR density variations).
- **Academic Argument:** Ground-truthing against standardized, institutional field data validates the scientific and practical applicability of the remote sensing methodology. Identifying the limiting factors (e.g., canopy occlusion) provides essential context for future research and operational guidelines.

---

## Eestikeelne Versioon: Lõputöö Põhiidee ja Plaan

### 1. Andmete ettevalmistus ja eeltöötlus (Prioriteet: Kõrge)
- **Eesmärk:** Kasutada LiDAR ALS (Airborne Laser Scanning) andmeid, keskendudes maapinna lähedasele kihile.
- **Tegevus:** Filtreerida andmeid ning luua võrastikumudel (CHM) kõrgusvahemikus 0–1,3 meetrit.
- **Akadeemiline põhjendus:** Vertikaalse profiili kitsendamine vähendab oluliselt kõrgemast taimestikust tekkivat müra ja aitab isoleerida objekte (nt lamapuit, alusmets), mis on ökoloogiliselt ja metsanduslikult olulised.

### 2. Kõrge tihedusega ALS faas (Prioriteet: Kõrge)
- **Eesmärk:** Töötada välja baasmudelid Eesti piirkondades, kus on saadaval kõrge tihedusega ALS andmed.
- **Meetodid:** Rakendada arvutinägemise (computer vision) mudeleid järgnevateks ülesanneteks:
  - **Paanide klassifitseerimine (Tile Classification):** Objektide lamapuidu olemasolu tuvastamine rasterpaanidel.
  - **Objektituvastus (Object Detection):** Üksikute objektide asukoha määramine.
  - **Segmenteerimine (Segmentation):** Pikslipõhine eraldamine täpseks pindala ja mahu hindamiseks.
- **Akadeemiline põhjendus:** Kõrge tihedusega andmed võimaldavad luua täpseid referentsandmestikke (hilisem "tõde" madala tihedusega mudelitele), mis on süvaõppe algoritmide edukaks treenimiseks hädavajalik.

### 3. Andmete hõrendamine ja mudelite kohandamine (Prioriteet: Kõrgemast Keskmiseni)
- **Probleem:** Kõrge tihedusega ALS andmeid ei ole saadaval üle kogu riigi, kuid madala tihedusega andmeid uuendatakse iga 4 aasta tagant.
- **Eksperiment:** Hõrendada algseid sisendandmeid, et simuleerida madalamat punktitihedust.
- **Treenimine:** Treenida klassifitseerimis-, tuvastus- ja segmenteerimismudeleid hõrendatud andmetel, kasutades "tõeliste märgistena" (true labels) kõrge tihedusega mudelite tulemusi.
- **Akadeemiline põhjendus:** Selline siirdeõppe (transfer learning) lahendus hindab mudelite paindlikkust andmekvaliteedi langemisel, sidudes lokaalsed kõrgekvaliteedilised uuringud riiklike madalakvaliteediliste seireandmetega.

### 4. Spetsiifiline lahendus: Hough Transformatsioon (Prioriteet: Keskmine)
- **Eesmärk:** Rakendada Hough' teisendust lineaarsete struktuuride (nt tüvede/lamapuidu) tuvastamiseks madala tihedusega ALS andmetelt.
- **Taust:** See meetod on näidanud potentsiaali sarnastes uuringutes mujal.
- **Akadeemiline põhjendus:** Erinevalt suurest andmemahust sõltuvatest süvaõppemudelitest pakuvad traditsioonilised geomeetrilised lähenemised tõlgendatavust ega vaja mahukat annoteerimist, olles seeläbi potentsiaalselt stabiilsemad väga hõredate punktipilvede puhul.

### 5. Rakendamine üle Eesti (Prioriteet: Keskmisest Madalani)
- **Eesmärk:** Rakendada treenitud mudeleid kogu Eesti ulatuses, kasutades riiklikku madala tihedusega (4-aastase tsükliga) ALS andmestikku.
- **Piirang:** Riikliku tasandi andmestike töötlemine nõuab tohutut arvutusvõimsust ja sõltub otseselt ajaressursist ning mudelite efektiivsusest. Tõenäoliselt piiratakse seda analüüsi lõputöö mahu sees juhusliku valimiga.

### 6. Valideerimine ja põhjuslik analüüs (Prioriteet: Kõrge)
- **Valideerimisandmed:** Võrrelda mudelite väljundeid reaalsete alusandmetega SMI-st (Statistiline Metsainventuur) ja Metsaregistrist.
- **Analüüs:**
  - Hinnata korrelatsioone reaalse metsaandmestikuga.
  - Tuvastada, mis mõjutab avastamistäpsust kõige enam (nt metsa struktuur, võrastiku avatus / liituvus, LiDAR-i punktitiheduse variatsioonid).
- **Akadeemiline põhjendus:** Lokaalsete ja standardiseeritud inventuuriandmetega valideerimine tõestab kaugseire metoodika teaduslikku väärtust ja praktilist rakendatavust. Leitud korrelatsioonid valgustingimuste, metsa struktuuri ja tuvastustäpsuse vahel annavad olulist uut teavet sensorfüüsika ja metsaökoloogia seostest.

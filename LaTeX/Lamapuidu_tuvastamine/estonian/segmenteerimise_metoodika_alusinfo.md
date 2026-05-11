# Segmenteerimise metoodika alusinfo

See dokument koondab lõpliku segmenteerimise metoodika kirjutamiseks vajaliku tehnilise alusinfo. Fookus on viimasel kasutatud töövool: GeoPackage'i polügoonide rasteriseerimine, kehtiva analüüsiala mask, mudelile antavate paanide koostamine ning püsiva testala ja kahe foldi jaotus.

## 1. Lõpliku metoodika eesmärk

Segmenteerimise eesmärk oli muuta lamapuidu tuvastamine paanipõhisest klassifitseerimisest pikslipõhiseks kaardistamiseks. Klassifitseerimisel piisab teadmisest, kas 128×128 paanis on lamapuitu. Segmenteerimisel peab mudel õppima lamapuidu kuju ja asukohta iga piksli tasemel, mistõttu on vajalik eraldi maskiandmestik.

Lõplikus metoodikas käsitleti lamapuitu semantilise segmenteerimisena: iga kehtiva analüüsiala piksel on kas lamapuit või taust. Piksleid väljaspool kontrollitud ala ei kasutatud treeningkao arvutamisel, sest neid ei saa usaldusväärselt tõlgendada taustana.

## 2. Sisendfailid ja nende roll

Peamised sisendfailid olid:

- `data/labels/cdw_labels_MP.gpkg` — käsitsi märgendatud lamapuidu polügoonid;
- `data/labels/valid_area.gpkg` — kontrollitud ala piir, mille sees märgendamata piksleid tohib käsitleda taustana;
- `seg_pipeline/input/baseline_chm.tif` — referentsraster, mille ruudustikku, ulatust ja koordinaatsüsteemi kasutati maski rasteriseerimisel;
- `seg_pipeline/input/composite_4band.tif` — lõplik CHM sisendvariant mudeli treenimiseks.

Lamapuidu GeoPackage sisaldas 1236 objekti. Polügoonide kogupindala oli 6492.0 m², keskmine pindala 5.25 m² ja mediaan 4.21 m². Kehtiva ala polügooni pindala oli 457769.0 m².

Joonis: `joonised/segmenteerimine_alusinfo/seg_gpkg_mask_workflow.png`

## 3. GPKG polügoonide rasteriseerimine

Mõlemad GeoPackage'i kihid teisendati EPSG:3301 koordinaatsüsteemi ja rasteriseeriti samale 5000×5000 pikslisele ruudustikule kui CHM. Rasteriseerimisel kasutati `rasterio.features.rasterize` loogikat ning `all_touched=True` seadet, et kitsad lamapuidu polügoonid ei kaoks rastervõrku teisendamisel ära. See on oluline, sest lamapuit on sageli kitsas objekt ja võib 0,2 m piksli juures paikneda pikslipiiridel.

Rasteriseerimise järel koostati 3-kanaliline juhendmask:

1. `target` — 1 tähendab lamapuitu, 0 tähendab tausta;
2. `valid_mask` — 1 tähendab, et pikslit kasutatakse loss'i arvutamisel, 0 tähendab ignoreeritud pikslit;
3. `ensemble_prob` — lõplikus V10 töövoos täideti nullidega, et säilitada ühilduvus varasema pipeline'iga.

Pikslite loogika oli järgmine:

- lamapuit: piksel on `valid_area.gpkg` sees ja `cdw_labels_MP.gpkg` polügooni sees;
- taust: piksel on `valid_area.gpkg` sees, kuid lamapuidu polügoonist väljas;
- ignoreeritud: piksel on kontrollitud alast väljas või CHM-is mittekehtiv.

Kogu maskis oli kehtivaid piksleid 11,444,080 ehk 45.78% rasterpinnast. Lamapuidu piksleid oli 226,584, mis moodustas 1.98% kehtivast analüüsialast. See näitab tugevat klasside tasakaalustamatust.

## 4. Püsiv testala ja foldide jaotus

Raster jagati viieks vertikaalseks 1000 veeru laiuseks triibuks. Kuna piksli suurus oli 0,2 m, vastas üks triip 200 m laiusele alale. Läänepoolne triip 0 jäeti püsivaks testalaks. Seda ei kasutatud mudeli valikul ega hüperparameetrite häälestamisel.

Lõplikus kahe foldiga jaotuses kasutati järgmisi rolle:

- püsiv testala: triip 0;
- fold 0 valideerimine: triip 1;
- fold 0 treening: triibud 2, 3 ja 4;
- fold 1 treening: triip 1;
- fold 1 valideerimine: triibud 2, 3 ja 4.

Joonis: `joonised/segmenteerimine_alusinfo/seg_final_spatial_split.png`

### Triipude pikslijaotus

| Triip | Veerud | Roll | Lamapuit px | Taust px | Ignoreeritud px | LP valid (%) | Valid kogu (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-999 | test | 61 844 | 4 938 156 | 0 | 1.24 | 100.00 |
| 1 | 1000-1999 | fold0 val / fold1 train | 101 104 | 3 901 241 | 997 655 | 2.53 | 80.05 |
| 2 | 2000-2999 | fold0 train / fold1 val | 21 950 | 1 483 285 | 3 494 765 | 1.46 | 30.10 |
| 3 | 3000-3999 | fold0 train / fold1 val | 26 043 | 495 003 | 4 478 954 | 5.00 | 10.42 |
| 4 | 4000-4999 | fold0 train / fold1 val | 15 643 | 399 811 | 4 584 546 | 3.77 | 8.31 |

### Testala ja foldide pikslijaotus

| Roll | Triibud | Lamapuit px | Taust px | Ignoreeritud px | LP valid (%) | Valid kogu (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Püsiv testala | 0 | 61 844 | 4 938 156 | 0 | 1.24 | 100.00 |
| Fold 0 treening | 2, 3, 4 | 63 636 | 2 378 099 | 12 558 265 | 2.61 | 16.28 |
| Fold 0 valideerimine | 1 | 101 104 | 3 901 241 | 997 655 | 2.53 | 80.05 |
| Fold 1 treening | 1 | 101 104 | 3 901 241 | 997 655 | 2.53 | 80.05 |
| Fold 1 valideerimine | 2, 3, 4 | 63 636 | 2 378 099 | 12 558 265 | 2.61 | 16.28 |

Joonis: `joonised/segmenteerimine_alusinfo/seg_final_pixel_distribution.png`

## 5. Paanide moodustamine mudelile

Mudeli sisendiks ei antud tervet 5000×5000 rasterpilti korraga, vaid sellest lõigati 256×256 piksliga paanid. Paanide samm oli 192 pikslit, mis tähendab 64 pikslit ülekatet. Maapinnal vastas üks paan 51,2×51,2 meetrile ning samm 38,4 meetrile.

Joonis: `joonised/segmenteerimine_alusinfo/seg_patch_extraction_scheme.png`

Paan jäeti andmestikust välja, kui selles oli vähem kui 328 kehtivat pikslit. See välistas peaaegu tühjad või valdavalt ignoreeritud piirkonnad. Iga treeningnäide koosnes kahest omavahel samas rasteraknas olevast osast:

- sisendpilt: CHM kanalid, lõplikus põhivariandis `composite_4band`;
- juhend: sama akna `target` ja `valid_mask`.

Paanide jaotus rollide kaupa oli:

| Roll | Paanid | Pos. paanid | Pos. paanid (%) | LP px paanides | Valid px paanides |
| --- | --- | --- | --- | --- | --- |
| Püsiv testala | 130 | 81 | 62.31 | 103 252 | 8 519 680 |
| Fold 0 treening | 95 | 77 | 81.05 | 110 536 | 4 505 505 |
| Fold 0 valideerimine | 118 | 108 | 91.53 | 173 474 | 6 901 546 |
| Fold 1 treening | 118 | 108 | 91.53 | 173 474 | 6 901 546 |
| Fold 1 valideerimine | 95 | 77 | 81.05 | 110 536 | 4 505 505 |

## 6. Mudelile antav lõplik sisend

Lõplikus metoodikas kasutati komposiitset CHM sisendit, sest see koondab mitu lamapuidu jaoks olulist infokihti. Komposiitsisendis on nii silutud kõrgusmuster, toorem kõrgussignaal, baasmudeli lokaalsed kõrgusväärtused kui ka kehtivate pikslite mask. See on lamapuidu puhul põhjendatud, sest osa objekte tuleb esile ainult tooretes lokaalse kõrguse muutustes, osa aga pigem silutud pikliku struktuurina.

Treeningul normaliseeriti CHM kanalid treeningandmestiku statistika alusel. Maskikanalit käsitleti binaarse infona. Mudelile anti korraga sisendpaan, sihtmask ja kehtivusmask; loss arvutati ainult nendel pikslitel, kus `valid_mask=1`.

## 7. Automaatne ablation ja lõpliku konfiguratsiooni valik

Lõpliku segmenteerimismetoodika valikul kasutati skripti `run_full_ablation_automated_top2.sh`. Selle eesmärk oli vältida olukorda, kus iga faasi parim üksiktulemus lukustab kogu järgneva otsinguruumi liiga vara. Selle asemel kanti igast faasist edasi kaks parimat konfiguratsiooni.

Joonis: `joonised/segmenteerimine_alusinfo/seg_top2_ablation_workflow.png`

Töövoog koosnes järgmistest sammudest:

1. `Preflight` — vajadusel seoti CHM sisendfailid `seg_pipeline/input` kausta ning ehitati iga CHM variandi jaoks uuesti `patch_index_*.csv` ja `band_stats_*.json`.
2. `Faas 2` — võrreldi CHM andmestiku variante: `baseline`, `raw`, `gauss`, `masked` ja `composite`.
3. `Faas 3` — kahe parima CHM variandi peal võrreldi mudeliarhitektuure. Lukustatud vaikeseadistuses kasutati kandidaate `3B`, `3C` ja `3E`, vastavalt U-Net++ EfficientNet-B0, U-Net++ EfficientNet-B2 ja DeepLabV3+ EfficientNet-B2.
4. `Faas 4` — kahe parima andmestiku/mudeli kombinatsiooni peal võrreldi kaofunktsioone ja nende parameetreid. Lukustatud kandidaadid olid `4A`, `4D`, `4F` ja `4H`, hõlmates DiceFocalit, tasakaalustatud Tverskyt, kõrgema täpsuse Tverskyt ning Tversky+clDice kombinatsiooni.
5. `Faas 5` — kahe parima andmestiku/mudeli/loss'i kombinatsiooni peal võrreldi augmentatsiooni ja regulariseerimise seadistusi. Lukustatud kandidaadid olid `5A`, `5D` ja `5E`, mis eristasid augmentatsioonita treeningut, täisaugmentatsiooni koos soft target'ite ja SWA-ga ning sama seadistust ilma SWA-ta.
6. `Faas 6` — pärast mudelivaliku lukustamist tehti lõplik testhinnang. Selles faasis kasutati `--evaluate-test` ja `--final-train-all` seadeid, st mudel treeniti kõigil mitte-test triipudel ning hinnati püsival testalal ehk triibul 0.

Valikumõõdik oli lukustatud `val_cldice`. See sobib lamapuidu ülesandesse paremini kui ainult pikslipõhine Dice, sest lamapuit on piklik ja katkendlik objektiklass ning oluline on säilitada objekti teljeline pidevus. Faaside 2-5 jooksul testala ei kasutatud. Konfiguratsioonid järjestati järgmise reegliga:

1. suurem keskmine `val_cldice`;
2. võrdse tulemuse korral suurem keskmine `val_f1`;
3. seejärel väiksem `val_cldice` standardhälve;
4. lõpuks suurem foldide/ridade katvus.

Selline järjestus eraldab mudelivaliku ja lõpliku hindamise. Metoodika seisukohast on oluline rõhutada, et püsivat testala ei kasutatud ei CHM variandi, arhitektuuri, loss'i ega augmentatsiooni valimiseks. Testala avati alles siis, kui lõplik konfiguratsioon oli valideerimistulemuste põhjal lukustatud.

## 8. Metoodika kirjutamise tuum lõputöös

Peatükis `Segmenteerimise metoodika` peaks rõhk olema järgmisel loogikal:

1. miks paanisildid ei ole pikslipõhise ülesande jaoks piisavad;
2. kuidas `cdw_labels_MP.gpkg` ja `valid_area.gpkg` muudeti rastermaskiks;
3. miks kontrollitud ala mask on vajalik, et märgendamata ala ei muutuks ekslikult taustaks;
4. kuidas püsiv testala ja fold0/fold1 ruumiliselt eraldati;
5. kuidas 256×256 paanid, target ja valid mask moodustasid mudeli treeningnäite;
6. miks komposiit-CHM valiti lõplikuks sisendiks;
7. kuidas automaatne top-2 ablation valis andmestiku, mudeli, loss'i ja augmentatsiooni;
8. kuidas lõplik väljund on tõenäosuskaart, millest saab läve abil binaarse lamapuidu maski.

Kõige olulisem akadeemiline mõte on see, et segmenteerimise metoodika usaldusväärsus sõltub vähem ühest mudeliarhitektuurist ja rohkem sellest, kas juhendmask eristab korrektselt kolme seisundit: lamapuit, usaldusväärne taust ja mittehinnatav ala. Ilma `valid_area.gpkg` kihita oleks suur osa märgendamata alast mudeli jaoks vale-negatiivne treeninginfo.

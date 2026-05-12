# Klassifitseerimise tulemuste loogiline töövoog

Koostatud 2026-05-12 peatükkide `4-metoodika.tex` ja `5-tulemused.tex` korrastamiseks.

## Põhijäreldus

Kasutaja pakutud järjestus on sisuliselt õige. Tulemuste peatükk peab näitama puhast uurimistöö loogikat: esmalt tekkis märgistusandmestik, seejärel katsetati sobivaid klassifitseerimismudeleid, moodustati pilootansambel, hinnati kogu paanistik tõenäosuslikult, laiendati märgistust aktiivõppe abil, võrreldi ansambli koosseisu ja agregeerimismeetodeid, treeniti lõplik ansambel ning hinnati seda sõltumatul ruumilis-ajalise jaotusega testandmestikul.

Tabel 6 ehk `tab:ensemble-methods-comparison` tuleb paigutada ansambelmudeli valiku juurde. Selle roll ei ole töö ajalugu kommenteerida, vaid anda kvantitatiivne põhjendus sellele, miks kolme CNN-Deep-Attn mudeli kõrvale sobib EfficientNet-B2 ning milline tõenäosuste koondamise viis andis võrdluses parima tulemuse.

Soovitatud sõnastusloogika:

> Pärast aktiivõppega laiendatud märgistusandmestiku moodustamist võrreldi neljanda ansambliliikme ja tõenäosuste agregeerimise alternatiive. Võrdlus näitas, et EfficientNet-B2 täiendas kolme CNN-Deep-Attn mudelit kõige paremini ning kaalutud hääletus saavutas kõrgeima keskmise F1-skoori.

Selline sõnastus hoiab töö struktuuri selge ja ei too tulemuste peatükki metakommentaari selle kohta, millises järjekorras arvutuslikud lisavõrdlused vormistati.

## Soovitatud järjestus tulemuste alampeatükis

### 1. Käsitsi paanimärgistus ja `auto_skip`

Tulemuste osa võiks alustada lühikese andmestiku lähtekohaga: esmalt märgistati paane käsitsi ning kasutati `auto_skip` loogikat selgelt tühjade paanide automaatseks negatiivseks märkimiseks. See on tulemuste peatükis vajalik ainult kontekstina, mitte metoodika kordusena.

Olulised arvud:

- Pilootfaasi treeninglogis loeti kokku 21 998 silti: 3 380 `cdw` ja 18 618 `no_cdw`.
- Sellest moodustus pilootansambli jaoks 15 850 treening-, 3 962 valideerimis- ja 2 186 testpaani.
- 15 850 treeningpaani koosseisu hinnanguline jaotus oli ligikaudu 5 461 käsitsi märgistatud ning 11 389 automaatselt vahele jäetud / valideeritud paani. Seda tuleb sõnastada hinnanguna, mitte absoluutse mõõtmisena.

Viited:

- [`output/tile_labels/train_ensemble.log`](../output/tile_labels/train_ensemble.log) - rida `Labels: 21998 total`, `Train=15850 Val=3962`, test ja treeningu käik.
- [`output/tile_labels/ensemble_meta.json`](../output/tile_labels/ensemble_meta.json) - ametlik pilootansambli metaandmestik.
- [`docs/ensemble_training_data_analysis_20250510.md`](ensemble_training_data_analysis_20250510.md) - pilootandmestiku rekonstruktsioon ja märgistusallikate hinnang.

Parandus `4-metoodika.tex` jaoks:

- Praegune lause "Ansambel treeniti 15 850 käsitsi märgistatud õpepildil" on liiga tugev. Täpsem oleks "15 850 käsitsi kinnitatud, käsitsi märgistatud ja `auto_skip` abil negatiivseks kinnitatud õpepaanil" või "15 850 kureeritud õppepaanil".

### 2. Esialgne mudeliotsing: millised klassifikaatorid töötavad

Järgmine loogiline samm on lai mudeliotsing: enne lõpliku ansambli kirjeldamist tuleb näidata, millised arhitektuurid üldse töötasid. Siin tuleb praegune `tab:model-search-v1` õigesse kohta.

Olulised tulemused:

- Mudeliotsing kasutas 19 812 treeningpaani ja 2 186 testpaani.
- Testil olid parimad:
  - top-5 virnaansambel: AUC 0,9982, F1 0,9672;
  - top-5 pehme hääletus: AUC 0,9982, F1 0,9646;
  - parim üksikmudel CNN-Deep-Attn-headwide: AUC 0,9977, F1 0,9644.
- Ristvalideerimise alternatiividest olid tugevad ConvNeXt-small, ConvNeXt-tiny ja EfficientNet-B2, kuid need ei ületanud selles etapis CNN-Deep-Attn perekonda üksikmudelina.

Viited:

- [`analysis/model_search_v1_ranked_test.csv`](../analysis/model_search_v1_ranked_test.csv) - top-5 ansambli ja üksikmudelite testitulemused.
- [`analysis/model_search_v1_ranked_cv.csv`](../analysis/model_search_v1_ranked_cv.csv) - ristvalideerimise tulemused ja arhitektuuride võrdlus.
- [`docs/models_training.md`](models_training.md) - mudeliotsingu kokkuvõte ja artefaktide viited.
- [`docs/model_search_and_v2_review_2026-04-07.md`](model_search_and_v2_review_2026-04-07.md) - mudeliotsingu ülevaade.

Tulemuste peatükis peab tabelile eelnema üks sissejuhatav lõik, mitte tabel kohe tühja alampeatüki alguses.

### 3. Pilootansambel ja selle kasutamine esmaseks paanihindamiseks

Pärast mudeliotsingut tuleb kirjeldada esialgset märgistamisansamblit. Selle roll ei olnud veel lõplik järeldus kogu uuringualale, vaid tööriist aktiivõppe ja paanide prioriseerimise jaoks.

Pilootansambli koosseis:

- kolm CNN-Deep-Attn mudelit juhuseemetega 42, 43 ja 44;
- üks EfficientNet-B2 mudel;
- TTA kasutusel;
- CNN-mudelid treeniti 50 epohhi;
- EfficientNet-B2 treeniti 30 epohhi;
- label smoothing 0,05;
- MixUp alfa 0,3.

Olulised tulemused:

- CNN-Deep-Attn seed 42: val AUC 0,9969, val F1 0,9580, lävi 0,87;
- CNN-Deep-Attn seed 43: val AUC 0,9972, val F1 0,9570, lävi 0,81;
- CNN-Deep-Attn seed 44: val AUC 0,9974, val F1 0,9596, lävi 0,83;
- EfficientNet-B2: val AUC 0,9963, val F1 0,9463, lävi 0,72;
- ansambel testil: AUC 0,9987, F1 0,9701, lävi 0,68, `n_test=2186`.

Viited:

- [`output/tile_labels/ensemble_meta.json`](../output/tile_labels/ensemble_meta.json) - pilootansambli mõõdikud ja hüperparameetrid.
- [`output/tile_labels/train_ensemble.log`](../output/tile_labels/train_ensemble.log) - treeningu tegelik käik, üksikmudelite val F1 ja ansambli test.
- [`docs/MODEL_DEVELOPMENT_TIMELINE_PRE_4ENSEMBLE_20260511.md`](MODEL_DEVELOPMENT_TIMELINE_PRE_4ENSEMBLE_20260511.md) - arenduse kronoloogia.

Soovitus:

- Praegune `tab:initial-ensemble` sobib siia.
- Tabeli pealkiri võiks täpsustada, et üksikmudelite read on valideerimisvalimilt, ansambli rida aga TTA-ga testvalimilt.

### 4. Kõigi paanide esmane hindamine ja aktiivõppe järjekord

Pilootansambliga hinnati suurem hulk paane ning igale paanile salvestati `model_prob`. Selle põhjal loodi käsitsi ülevaatuse järjekord.

Olulised arvud:

- Drop-above-1,3 m CHM töövoos loodi 119 CHM rasterit.
- Neist märgistati automaatselt 100 rasterit; 19 jäeti `nodata` reegli tõttu kõrvale.
- Toortöövoos tekkis 586 124 märgistatud paani.
- Põhilises kanonilises andmestikus kasutatakse 580 136 rida. Tekstis tuleb vältida 586 124 ja 580 136 läbisegi kasutamist ilma selgituseta; tulemuste ja ruumilise jaotuse põhitekstis on kasutusel 580 136.
- Automaatennustuste histogrammi jaoks kasutati 553 937 paani.

Aktiivõppe valim:

- Madala kindluse vahemik oli `[0,39; 0,61]`.
- Sellesse vahemikku jäi 29 628 paani.
- Ülejäänutest valiti 5% juhuvalim ehk 26 215 paani.
- Käsitsi ülevaatuse järjekord kokku: 55 843 paani ehk 10,08% automaatselt sildistatud paanidest.

Viited:

- [`docs/THESIS_DROP13_FILTERED_CHM_PIPELINE.md`](THESIS_DROP13_FILTERED_CHM_PIPELINE.md) - drop13 CHM töövoog, kõikide rasterite hindamine ja järjekorra loomise käsk.
- [`analysis/onboarding_new_laz/drop13_prediction_confidence_recommendation.json`](../analysis/onboarding_new_laz/drop13_prediction_confidence_recommendation.json) - `[0,39; 0,61]`, 5% spot-check ja 55 843 paani.
- [`analysis/onboarding_new_laz/drop13_prediction_confidence_recommendation.md`](../analysis/onboarding_new_laz/drop13_prediction_confidence_recommendation.md) - sama info loetavas vormis.
- [`output/onboarding_labels_v2_drop13/manual_review_queue_pre_split.csv`](../output/onboarding_labels_v2_drop13/manual_review_queue_pre_split.csv) - käsitsi ülevaatuse järjekord.

### 5. Märgiste kvaliteet: keerulised paanid ja juhuvalim

Pärast `model_prob` arvutamist vaadati käsitsi üle nii mudeli jaoks rasked paanid kui ka juhuvalim kõrge kindlusega paanidest. Tulemuste peatükis võiks see olla eraldi lühike lõik, sest see põhjendab, miks hilisem treeningvalim ei ole lihtsalt mudeli enda peegeldus.

Olulised kvaliteedinäitajad:

- `queue_spotcheck` on esmane automaatmärgiste kvaliteedi hinnang, sest see pärineb autojääkidest, mitte sihilikult rasketest paanidest.
- `queue_spotcheck`: 5 856 ülevaadatud ja hinnatud paani, kooskõla 0,9092, täpsus 0,8792, tundlikkus 0,8422, F1 0,8603.
- `queue_low_confidence` on tahtlikult raskete juhtude valim; seda ei tohi üldise mudelikvaliteedi KPI-na spot-checkiga kokku segada.
- `queue_low_confidence`: 6 279 ülevaadatud, 6 278 hinnatud, kooskõla 0,5616, F1 0,4746.

Viited:

- [`output/onboarding_labels_v2_drop13/spotcheck_metrics_summary.json`](../output/onboarding_labels_v2_drop13/spotcheck_metrics_summary.json) - spot-check ja low-confidence kvaliteedimõõdikud.

Soovitus:

- Tulemuste tekstis rõhutada, et madala kindlusega rühma kehvem F1 ei ole mudeli üldine F1, vaid tõend, et järjekord tõesti rikastas keerulisi juhtumeid.

### 6. Suurendatud ja filtreeritud valim

Pärast strateegilist käsitsi ülevaatust ja automaatsete kõrge kindlusega ridade filtreerimist tekkis suurem valim, mille põhjal sai teha lõpliku ruumilis-ajalise jaotuse ning ansambli täpsemad võrdlused.

Põhijaotuse arvud:

- Kokku: 580 136 märgistust.
- Sobilikud read: 142 465 ehk 24,56%.
- Test: 56 521 ehk 9,74%.
- Valideerimine: 13 850 ehk 2,39%.
- Treening: 67 290 ehk 11,60%.
- Välistatud/puhver: 442 475 ehk 76,27%.
- Treeningkomplekt: 50 635 `cdw` ja 16 655 `no_cdw`, CDW osakaal 75,25%.
- Testikomplekt: 39 504 `cdw` ja 17 017 `no_cdw`, CDW osakaal 69,89%.

Märgistusallika jaotus ruumilises splitis (oluline):

- `manual`: 12 177 rida kokku; treeningus 4 158, testis 2 347, valideerimises 868.
- `auto_skip`: 31 837 rida; kõik on sobilikud negatiivsed näited.
- `auto`: 536 122 rida; treeningusse jõudis 50 006 kõrge kindlusega rida, testikomplekti 38 420.

Viited:

- [`SPLIT_ASSIGNMENT_REPORT.md`](../SPLIT_ASSIGNMENT_REPORT.md) - põhiline 580 136 rea splitiraport.
- [`scripts/assign_label_splits.py`](../scripts/assign_label_splits.py) - sobilikkuse loogika `manual`, `auto_skip`, `model_prob > 0.95` või `< 0.05`.
- [`output/tile_labels_spatial_splits/training_metadata.json`](../output/tile_labels_spatial_splits/training_metadata.json) - lõpliku testikomplekti mõõdikud.

Märkus:

- `model_search_v4` kasutab rangemaid lävesid `t_high=0.9995` ja `t_low=0.0698`, kuid 67 290 / 13 850 / 56 521 põhijaotuse raport kasutab `>0.95` ja `<0.05`. Peatükis tuleb otsustada, kumb töövoog on põhiline. Kui lõplik tulemuste peatükk jääb 67 290 treeningpaani juurde, peab metoodika ja tulemuste tekst viitama samale lävekomplektile.

### 7. Ansambelmudeli valik ja agregeerimismeetodid

See on koht, kuhu praegune Tabel 6 loogiliselt kuulub. Tabeli teaduslik roll on põhjendada lõpliku ansambli koosseisu ning tõenäosuste koondamise viisi.

Olulised tulemused:

- Metoodika: 5x5 repeated stratified CV, 25 hindamist meetodi kohta, per-fold threshold tuning.
- Parim kandidaat: EfficientNet-B2 koos kaalutud hääletusega.
- EfficientNet-B2 + weighted vote:
  - F1 mean 0,9866;
  - F1 SD 0,0042;
  - 95% UV `[0,9849; 0,9883]`;
  - AUC 0,9810;
  - läve keskmine 0,5256;
  - p-väärtus lihtsa pehme hääletuse vastu `<0,0001`.
- EfficientNet-B2 + LR stacking oli praktiliselt sama: F1 0,9865, AUC 0,9810.
- ConvNeXt-small + LR stacking oli samuti tugev: F1 0,9857, kuid jäi EfficientNet-B2 weighted vote tulemusele alla.
- Võitja JSON-is: `candidate=effnet_b2`, `method=weighted_vote`, `f1=0.986598`.

Viited:

- [`output/ensemble_4th_model_comparison_v2/comparison_v2_results.json`](../output/ensemble_4th_model_comparison_v2/comparison_v2_results.json) - Tabel 6 täpne allikas.
- [`output/ensemble_4th_model_comparison_v2/run.log`](../output/ensemble_4th_model_comparison_v2/run.log) - jooksu logi.
- [`output/ensemble_4th_model_ablation/ablation_summary.txt`](../output/ensemble_4th_model_ablation/ablation_summary.txt) - neljanda mudeli ablation, kus EfficientNet-B2 võitis väikese marginaaliga.
- [`output/ensemble_4th_model_ablation/ablation_results.json`](../output/ensemble_4th_model_ablation/ablation_results.json) - ablationi masinloetav väljund.
- [`scripts/ensemble_4th_model_comparison_v2.py`](../scripts/ensemble_4th_model_comparison_v2.py) - võrdluse skript.

Soovitatud tekst Tabel 6 ette:

> Pärast suurendatud märgiste kogumi moodustamist võrreldi lõpliku ansambli kandidaate ja ennustuste koondamise viise ühtses 5x5 ristvalideerimise raamistikus. Võrdluse eesmärk oli valida mudel, mis täiendaks kolme CNN-Deep-Attn baasmudelit, ning hinnata, kas ennustusi on otstarbekam koondada pehme hääletuse, kaalutud hääletuse või virnaansambli abil. Tulemused näitasid, et EfficientNet-B2 lisamine kolme CNN-Deep-Attn mudeli kõrvale ning kaalutud hääletus andsid parima F1-skoori ja olid logistilise virnaansambliga praktiliselt samaväärsed, kuid lihtsamini tõlgendatavad.

Tabeli pealkiri võiks olla:

> Ansambli agregeerimismeetodite võrdlus 5x5 ristvalideerimisel.

### 8. Lõpliku ansambli treenimine ruumilis-ajalise splitiga

Pärast Tabel 6 otsustuslikku kinnitust tuleb kirjeldada lõpliku ansambli treenimist. See peab tulema pärast ruumilis-ajalise jaotuse esitamist, sest muidu jääb kasutamata testandmestiku väide põhjendamata.

Lõplik ansambel:

- CNN-Deep-Attn seed 42;
- CNN-Deep-Attn seed 43;
- CNN-Deep-Attn seed 44;
- EfficientNet-B2;
- salvestatud failid `output/tile_labels_spatial_splits/*_spatial.pt`.

Lõplik testitulemus:

- Testkomplekt: 56 521 paani.
- CDW testis: 39 504.
- No-CDW testis: 17 017.
- Ensemble AUC: 0,98847.
- Ensemble F1: 0,98194.
- Otsustuslävi: 0,4.
- Hindamise ajatempel: 2026-04-25T08:10:59Z.

Viited:

- [`output/tile_labels_spatial_splits/training_metadata.json`](../output/tile_labels_spatial_splits/training_metadata.json) - lõpliku testitulemuse peamine allikas.
- [`OPTION_B_SPATIAL_SPLITS_SUMMARY.md`](../OPTION_B_SPATIAL_SPLITS_SUMMARY.md) - lõpliku ümbertreeningu kokkuvõte.
- [`OPTION_B_SPATIAL_SPLITS_COMPARISON.md`](../OPTION_B_SPATIAL_SPLITS_COMPARISON.md) - algse ja ruumilise splitiga ümbertreenitud ansambli võrdlus.
- [`scripts/test_evaluate_spatial_splits.py`](../scripts/test_evaluate_spatial_splits.py) - testhindamise skript.
- [`scripts/retrain_ensemble_spatial_splits.py`](../scripts/retrain_ensemble_spatial_splits.py) - ruumilise splitiga treenimise skript.

Soovitus:

- Praeguses tulemuste tekstis dubleerivad read 102 ja 104 sama infot. Jätta üks tugev lõik ja lisada sinna allikana `training_metadata.json`.
- "ületades esialgset märgistamisansamblit 1,2 protsendipunkti võrra" on F1 suhtes õige suund, kuid AUC vähenes 0,9987 -> 0,9885, sest lõplik test on rangem ja ruumiliselt lahutatud. Seda tuleb vältida esitamast lihtsa "kõik paranes" narratiivina.

### 9. Kogu andmestiku lõplik ümberhindamine

Pärast lõpliku ansambli treenimist hinnati kõik paanid uuesti. See on klassifitseerimise tulemuste alampeatüki loogiline viimane mudeliväljundi samm enne vigade analüüsi ja edasist kasutust.

Olulised arvud:

- Ümberhinnatud fail: `data/chm_variants/labels_canonical_with_splits_retrained_ensemble.csv`.
- Kokku: 580 136 silti.
- Retrained NaN: 0.
- Originaalennustustega võrreldes oli 548 299 paanil mõlemad tõenäosused olemas.
- Originaalil oli 31 837 NaN, ümbertreenitud ansamblil 0 NaN.
- Original mean prob: 0,3824.
- Retrained mean prob: 0,3757.
- Keskmine absoluutne erinevus: 0,05515.
- Mediaanne absoluutne erinevus: 0,02695.

Viited:

- [`data/chm_variants/labels_canonical_with_splits_retrained_ensemble.csv`](../data/chm_variants/labels_canonical_with_splits_retrained_ensemble.csv) - lõplikud tõenäosused kõigile paanidele.
- [`OPTION_B_SPATIAL_SPLITS_COMPARISON.md`](../OPTION_B_SPATIAL_SPLITS_COMPARISON.md) - ümberhindamise võrdlusstatistika.
- [`scripts/postprocess_spatial_split_retraining.py`](../scripts/postprocess_spatial_split_retraining.py) - postprocess, mis kutsub ümberhindamise ja raporti loomise.
- [`scripts/recalculate_model_probs_tta_ensemble.py`](../scripts/recalculate_model_probs_tta_ensemble.py) - lõpliku TTA ansambli tõenäosuste arvutus.

Soovitus:

- Tulemuste peatükis tuleb seda kirjeldada pärast testitulemusi, mitte enne.
- Seda ei tohiks nimetada valideerimiseks. See on inferents / kogu andmestiku ümberhindamine.

### 10. Vigade analüüs

Vigade analüüs peab tulema pärast lõplikku testitulemust ja kogu andmestiku ümberhindamist. Siis on selge, millise mudeli vigu analüüsitakse.

Soovitus:

- Praegune FP/FN lõik sobib, kuid vajab konkreetseid näitefaile või vähemalt sidumist `OPTION_B_SPATIAL_SPLITS_COMPARISON.md` suurimate tõenäosusmuutuste loendiga.
- Kui visuaalseid jooniseid ei ole, võiks placeholderid asendada kas tegelike näidetega või ajutiselt eemaldada.

## Puuduvad või täpsustamist vajavad tulemused

Need punktid tulenevad eeskätt `4-metoodika.tex` klassifitseerimise peatükist. Kui metoodikas on tegevus kirjeldatud, peaks tulemuste peatükk kas näitama selle tegevuse tulemust või selgelt ütlema, et tegevus oli abistav, mitte eraldi hinnatud katse.

1. **Käsitsi märgistamise maht etappide kaupa.** Praegu on kasutatav koondarv 12 177 käsitsi märgistatud või üle vaadatud paani, kuid tulemuste peatükis peaks olema eristus: esmane käsitsi märgistus, aktiivõppe järel üle vaadatud madala kindlusega paanid ning 5% juhuvalim.
2. **Märgistatud kaardilehtede arv.** Vaja on kinnitada, mitu kaardilehte olid käsitsi täielikult või osaliselt üle vaadatud, mitu hinnati automaatselt ja mitu jäeti andmepuuduse tõttu kõrvale.
3. **`auto_skip` statistika.** Kas tulemuste peatükis esitatav 31 837 `auto_skip` paani on kogu lõplik automaatselt negatiivseks kinnitatud hulk? Kui jah, tuleb see siduda ruumilise jaotuse ning klassijaotusega.
4. **Ebakindlate ehk `unknown` märgiste käsitlus.** Märgistustöövoog lubab ebaselgeid juhtumeid, kuid tulemuste peatükk peab ütlema, kas need eemaldati, lahendati hilisema ülevaatusega või kodeeriti lõplikult ühte kahest klassist.
5. **Välitööde katse.** Metoodika viitab välitööde piiratud kasutatavusele. Tulemuste peatükki on vaja vähemalt lühike tulemus: kontrollitud alade arv, ligikaudne aeg ja peamine järeldus.
6. **Seletuskaartide kasutamine.** Kui IntGrad, HiResCAM, GradCAM++ ja RISE olid üksnes märgistamise abivahendid, piisab ühest lausest. Kui nende abil hinnati mudeli käitumist, on vaja lisada näidisjoonis või kokkuvõtlik tulemus.
7. **CHM-variantide klassifitseerimistulemus.** Klassifitseerimise CHM-võrdluse tulemused on olemas failis `output/chm_variant_selection/results.csv`, kuid LaTeX-i tulemuste peatükis neid veel tabelina ei esitata. Lisaks on automaatselt loodud raport `CHM_VARIANT_SELECTION_REPORT.md` aegunud ja tuleb enne tsiteerimist uuesti genereerida.
8. **Lõpliku testi segadusmaatriks.** Olemas on AUC, F1 ja lävend, kuid akadeemiliselt oleks tugevam lisada ka täpsus, tundlikkus, TP, FP, TN ja FN. Kui ennustusfailid on alles, saab need arvutada.
9. **Käsitsi märgiste põhine testitulemus.** Kuna osa lõplikust testikomplektist võib põhineda kõrge kindlusega automaatmärgistel, oleks vaja eraldi tulemust ainult käsitsi märgistatud või käsitsi üle vaadatud testpaanidel. Kui seda ei tehta, tuleb piirang tulemuste tõlgenduses nimetada.
10. **Lõpliku koguandmestiku klassijaotus.** Pärast viimast hindamist tuleks esitada, mitu paani klassifitseeriti lõpliku lävendi korral lamapuiduks ja mitu mitte-lamapuiduks.
11. **Lävendite järjepidevus.** Esialgne ansambel kasutas lävendit 0,68, ansambli võrdluses on optimaalne keskmine lävend ligikaudu 0,526 ning lõplikus testis 0,40. Tekst peab selgitama, milline lävend millises etapis kehtib.
12. **Lõpliku agregeerimise tegelik rakendus.** Tabel 6 näitab kaalutud hääletuse paremust, kuid tuleb kontrollida, kas lõplikus tootmisjooksus kasutati kaalutud hääletust või lihtsat/TTA keskmistamist. Väide peab vastama tegelikule koodile ja väljundfailile.
13. **Ruumilise puhvertsooni lõplik valik.** Metoodika ja tulemused peavad kasutama sama lõplikku ruumilist jaotust. Kui klassifitseerimise lõppmudelis kasutati 12,8 m vahet, ei tohi samasse lõplikku väitesse segada hilisemat 51,2 m alternatiivi.

## Algne küsimuste loend

1. Mitu paani märgistati käsitsi enne pilootansamblit ja mitu lisandus aktiivõppe ülevaatuses?
2. Mitu kaardilehte olid käsitsi üle vaadatud ning kas ülevaatus oli täielik või osaline?
3. Kuidas käsitleti lõplikus andmestikus `unknown` või ebaselgeid silte?
4. Kas välitööde kohta on olemas kontrollitud alade arv, kuupäevad või paanide arv?
5. Kas CHM-variantide klassifitseerimise võrdluse kohta on olemas lõplik tabel, mida võib tulemuste peatükis kasutada?
6. Kas lõpliku testi ennustused on alles, et arvutada segadusmaatriks ning eraldi käsitsi märgiste põhine tulemus?
7. Kas lõplik koguandmestiku hindamine andis valitud lävendi juures lamapuidu ja mitte-lamapuidu paanide lõpliku arvu?
8. Kas lõpliku ansambli inferents kasutas kaalutud hääletust või lihtsat mudelite keskmistamist?

## Vastused autori küsimustele

| Küsimus | Vastus olemasolevate failide põhjal | Kasutus tulemuste peatükis |
|---|---|---|
| Käsitsi märgistatud paanid enne pilootansamblit ja aktiivõppes | Pilootansambli treeninglogi kinnitab 21 998 silti, millest moodustati 15 850 treening-, 3 962 valideerimis- ja 2 186 testpaani. Pilootansambli treeninghulga käsitsi märgiste täpne arv ei ole logis eraldi kinnitatud; olemas on rekonstrueeritud hinnang ligikaudu 5 461 käsitsi märgistatud ja 11 389 `auto_skip` / valideeritud paani kohta. Lõplikus kanonilises andmestikus on 12 177 `manual` allikaga paani. Aktiivõppe järjekorda planeeriti 55 843 paani, kuid kvaliteedimõõdikutes on hinnatud 12 134 paani. | Peatükis tuleb eristada kinnitatud arve ja hinnangulist pilootandmestiku allikajaotust. |
| Käsitsi üle vaadatud kaardilehed | Lõplikus 580 136 paaniga failis esineb `manual` allikas 12 kaardilehel, 27 kaardileht-aasta ehk rasterfaili ulatuses. Kogu lõplik andmestik hõlmab 23 kaardilehte ja 100 rasterfaili. Standardiseerimise raportis oli lähteindeksis 119 TIFF-faili, millest 100 sobitati lõpliku andmestikuga. | Tulemuste tekstis kasutada kinnitatud arve: 12 käsitsi allikaga kaardilehte, 27 käsitsi allikaga rasterfaili, 100 lõplikku rasterfaili. |
| `unknown` või ebaselged märgised | Lõplikus kanonilises failis ja lõplikus tõenäosusfailis on `unknown` märgiseid 0. Lõplikud klassid on ainult `cdw` ja `no_cdw`. | Tulemuste peatükis võib kirjutada, et ebaselged märgised ei jõudnud lõplikku kaheklassilisse andmestikku eraldi klassina. |
| Välitööde arvud | Repositooriumist ei leitud klassifitseerimise tulemuste jaoks kasutatavat välitööde koondarvu. Olemas on metoodiline märkus välitööde piiratud kasutatavuse kohta, kuid puudub kontrollitud alade arv või paanide arv. | Ilma täiendava autoripoolse arvuta mitte lisada tulemuste tabelit. Sobib arutelu või metoodilise piiranguna. |
| CHM-variantide klassifitseerimistabel | Jah, klassifitseerimise CHM-variantide hindamine toimus skriptiga `scripts/chm_variant_selection.py`. Katse kasutas 17 403 kureeritud kirjet, millest 3 481 moodustasid ruumilise V4 testhulga. Praegune `CHM_VARIANT_SELECTION_REPORT.md` on aegunud, sest see võtab kokku ainult varasema osalise seisu. Ajakohane esmane allikas on `output/chm_variant_selection/results.csv`. Lõpetatud variantide põhjal saavutas parima üksikmudeli tulemuse Gaussi silumisega CHM koos EfficientNet-B2 mudeliga: keskmine test-F1 0,8561 ja keskmine test-AUC 0,9342. | Tulemuste peatükki tuleb lisada CHM-sisendvariantide klassifitseerimistabel. Praegune üldine väide “alla 1% F1” tuleb asendada tabeliga ning eristada katse tulemus lõpliku tootmistoru tegelikult kasutatud CHM-ist. |
| Lõpliku testi segadusmaatriks ja käsitsi märgiste alamhulk | Ametlik lõpliku testi fail `training_metadata.json` sisaldab AUC, F1, lävendit ja testhulga suurust, kuid mitte segadusmaatriksit. Lõplikust CSV-st saab arvutada diagnostilise segadusmaatriksi, kuid see ei anna sama F1-väärtust kui ametlik testmetaandmestik; seetõttu ei tohi seda kasutada ametliku testitabelina ilma hindamisskripti uuesti jooksutamata ja ennustusi salvestamata. | Ametlikku tulemuste tabelisse lisada AUC, F1, lävend ja testhulga suurus. Segadusmaatriks lisada ainult pärast ametliku testhindamise uuesti salvestamist. |
| Lõplik koguandmestiku klassijaotus | Lõpliku tõenäosusfaili põhjal on lävendil 0,40 prognoositud 172 761 lamapuidu paani ja 407 375 mitte-lamapuidu paani. Puuduvaid tõenäosusi on 0. | Sobib kogu andmestiku inferentsi kokkuvõttesse, kuid allpool märgitud skriptitee kontroll tuleb enne lõplikku LaTeX-i lahendada. |
| Lõpliku ansambli agregeerimine | `test_evaluate_spatial_splits.py` kasutab nelja mudeli tõenäosuste lihtsat keskmistamist. `recalculate_model_probs_tta_ensemble.py` kasutab 8x TTA-d ja nelja mudeli lihtsat pehmet hääletust. Kaalutud hääletust nendes skriptides ei rakendata. Lisaks laadib kogu andmestiku ümberhindamise skript praegu `output/tile_labels/*.pt` mudeleid, mitte `output/tile_labels_spatial_splits/*_spatial.pt` mudeleid. | Tulemuste tekstis ei tohi väita, et lõplik inferents kasutas kaalutud hääletust, kui skript jäi lihtsa keskmistamise juurde. Koguandmestiku lõpliku hindamise fail tuleb enne lõplikku väidet kontrollida või uuesti genereerida ruumilise splitiga mudelitest. |

## Tulemuste peatükki sobivad tabelid ja kirjeldused

Alljärgnevad tabelid on vormistatud tulemuste peatüki jaoks. Kirjeldused esitavad tulemused neutraalselt; tõlgendused ja metoodilised piirangud on koondatud eraldi aruteluplokki.

### Tabel A. Kanonilise märgistusandmestiku jaotus allika järgi

| Märgise allikas | Paanide arv | Osakaal | Lamapuit (`cdw`) | Mitte-lamapuit (`no_cdw`) |
|---|---:|---:|---:|---:|
| Käsitsi märgistatud või üle vaadatud (`manual`) | 12 177 | 2,10% | 4 461 | 7 716 |
| CHM-põhiselt automaatselt vahele jäetud (`auto_skip`) | 31 837 | 5,49% | 0 | 31 837 |
| Mudeliga automaatselt hinnatud (`auto`) | 536 122 | 92,41% | 160 896 | 375 226 |
| Kokku | 580 136 | 100,00% | 165 357 | 414 779 |

Tulemuskirjeldus: kanoniline klassifitseerimisandmestik sisaldas 580 136 paani. Märgise allika järgi moodustasid suurima osa automaatselt hinnatud paanid. Käsitsi märgistatud või üle vaadatud paane oli 12 177 ning `auto_skip` reegliga mitte-lamapuiduks märgitud paane 31 837.

Allikas: `output/onboarding_labels_v2_drop13_standardized/summary.json`, `data/chm_variants/labels_canonical_with_splits_retrained_ensemble.csv`.

### Tabel B. Lõpliku andmestiku ruumiline ulatus allika järgi

| Allikas | Kaardilehti | Kaardileht-aasta/rasterfaili kombinatsioone | Paane |
|---|---:|---:|---:|
| `manual` | 12 | 27 | 12 177 |
| `auto_skip` | 22 | 83 | 31 837 |
| `auto` | 23 | 99 | 536 122 |
| Kokku lõplikus andmestikus | 23 | 100 | 580 136 |

Tulemuskirjeldus: lõplik kanoniline andmestik hõlmas 23 kaardilehte ja 100 rasterfaili. Käsitsi allikaga paane esines 12 kaardilehel ning 27 kaardileht-aasta kombinatsioonis.

Allikas: `data/chm_variants/labels_canonical_with_splits_retrained_ensemble.csv`.

### Tabel C. Aktiivõppe valimi moodustamine

| Valimi osa | Valikureegel | Paanide arv | Osakaal automaatselt prognoositud paanidest |
|---|---|---:|---:|
| Madala kindlusega paanid | `model_prob` vahemikus 0,39 kuni 0,61 | 29 628 | 5,35% |
| Juhuvalim ülejäänud paanidest | 5% väljaspool madala kindlusega vahemikku | 26 215 | 4,73% |
| Käsitsi ülevaatuse järjekord kokku | Madal kindlus + juhuvalim | 55 843 | 10,08% |

Tulemuskirjeldus: esialgse ansambli tõenäosuste alusel moodustati käsitsi ülevaatuse järjekord 55 843 paanist. Järjekorda lisati 29 628 madala kindlusega paani ja 26 215 juhuvalimisse kuulunud paani.

Allikas: `analysis/onboarding_new_laz/drop13_prediction_confidence_recommendation.json`.

### Tabel D. Käsitsi ülevaatuse kvaliteedinäitajad

| Ülevaatuse rühm | Hinnatud paane | Kooskõla | Täpsus | Tundlikkus | F1 |
|---|---:|---:|---:|---:|---:|
| 5% juhuvalim | 5 856 | 0,909 | 0,879 | 0,842 | 0,860 |
| Madala kindlusega valim | 6 278 | 0,562 | 0,440 | 0,516 | 0,475 |
| Koondülevaade | 12 134 | 0,729 | 0,614 | 0,662 | 0,637 |

Tulemuskirjeldus: käsitsi ülevaatuse logides oli hinnatud 12 134 paani. Juhuvalimi F1 oli 0,860 ning madala kindlusega valimi F1 oli 0,475.

Allikas: `output/onboarding_labels_v2_drop13/spotcheck_metrics_summary.json`.

### Tabel E. Esialgse mudeliotsingu parimad tulemused

| Mudel või ansambel | AUC | F1 | Täpsus | Tundlikkus | Lävend |
|---|---:|---:|---:|---:|---:|
| Top-5 logistiline virnaansambel | 0,9982 | 0,9672 | 0,9789 | 0,9558 | 0,924 |
| Top-5 pehme hääletus | 0,9982 | 0,9646 | 0,9646 | 0,9646 | 0,596 |
| CNN-Deep-Attn-headwide | 0,9977 | 0,9644 | 0,9701 | 0,9587 | 0,730 |
| CNN-Deep-Attn | 0,9973 | 0,9630 | 0,9673 | 0,9587 | 0,600 |

Tulemuskirjeldus: esialgses mudeliotsingus saavutas kõrgeima F1-skoori top-5 logistiline virnaansambel. Üksikmudelitest oli parim CNN-Deep-Attn-headwide.

Allikas: `analysis/model_search_v1_ranked_test.csv`.

### Tabel F. Pilootansambli treeningparameetrid ja testitulemus

| Näitaja | Väärtus |
|---|---:|
| Treeningpaanid | 15 850 |
| Valideerimispaanid | 3 962 |
| Testpaanid | 2 186 |
| CNN-Deep-Attn mudeleid | 3 |
| EfficientNet-B2 mudeleid | 1 |
| CNN-Deep-Attn epohhid | 50 |
| EfficientNet-B2 epohhid | 30 |
| Label smoothing | 0,05 |
| MixUp alfa | 0,30 |
| TTA | jah |
| Test AUC | 0,9987 |
| Test F1 | 0,9701 |
| Lävend | 0,68 |

Tulemuskirjeldus: pilootansambel koosnes kolmest CNN-Deep-Attn mudelist ja ühest EfficientNet-B2 mudelist. Ansambel saavutas pilootandmestiku testhulgal AUC väärtuse 0,9987 ja F1 väärtuse 0,9701.

Allikas: `output/tile_labels/ensemble_meta.json`.

### Tabel G. Ansambelmudeli valiku ja agregeerimismeetodite võrdlus

| Neljas mudel | Agregeerimismeetod | Keskmine F1 | F1 standardhälve | 95% usaldusvahemik | Keskmine AUC | Keskmine lävend |
|---|---|---:|---:|---|---:|---:|
| EfficientNet-B2 | Kaalutud hääletus | 0,9866 | 0,0042 | [0,9849; 0,9883] | 0,9810 | 0,5256 |
| EfficientNet-B2 | Logistiline virnaansambel | 0,9865 | puudub koondväljas | puudub koondväljas | 0,9810 | puudub koondväljas |
| ConvNeXt-small | Logistiline virnaansambel | 0,9857 | puudub koondväljas | puudub koondväljas | puudub koondväljas | puudub koondväljas |

Tulemuskirjeldus: ansambli neljanda mudeliliikme ja agregeerimismeetodi võrdluses saavutas kõrgeima keskmise F1-väärtuse EfficientNet-B2 koos kaalutud hääletusega.

Allikas: `output/ensemble_4th_model_comparison_v2/comparison_v2_results.json`.

### Tabel H. Ruumilis-ajalise jaotuse suurus ja klassijaotus

| Jaotus | Paane | Osakaal kogu andmestikust | Lamapuit (`cdw`) | Mitte-lamapuit (`no_cdw`) |
|---|---:|---:|---:|---:|
| Treening | 67 290 | 11,60% | 50 635 | 16 655 |
| Valideerimine | 13 850 | 2,39% | 10 382 | 3 468 |
| Test | 56 521 | 9,74% | 39 504 | 17 017 |
| Puhver või kasutamata | 442 475 | 76,27% | 64 836 | 377 639 |
| Kokku | 580 136 | 100,00% | 165 357 | 414 779 |

Tulemuskirjeldus: lõpliku ruumilis-ajalise jaotuse alusel kasutati treeninguks 67 290 paani, valideerimiseks 13 850 paani ja sõltumatuks testimiseks 56 521 paani.

Allikas: `SPLIT_ASSIGNMENT_REPORT.md`, `data/chm_variants/labels_canonical_with_splits_retrained_ensemble.csv`.

### Tabel I. Lõpliku ansambli ametlik sõltumatu testitulemus

| Näitaja | Väärtus |
|---|---:|
| Testpaanid | 56 521 |
| Lamapuidu paanid testis | 39 504 |
| Mitte-lamapuidu paanid testis | 17 017 |
| AUC | 0,9885 |
| F1 | 0,9819 |
| Lävend | 0,40 |

Tulemuskirjeldus: ruumilis-ajalise jaotuse alusel treenitud lõplik ansambel saavutas sõltumatul testhulgal AUC väärtuse 0,9885 ja F1 väärtuse 0,9819.

Allikas: `output/tile_labels_spatial_splits/training_metadata.json`.

### Tabel J. Kogu andmestiku tõenäosusfaili koondstatistika

| Näitaja | Väärtus |
|---|---:|
| Paanide koguarv | 580 136 |
| Puuduvad tõenäosused | 0 |
| Keskmine tõenäosus | 0,3757 |
| Standardhälve | 0,3244 |
| Minimaalne tõenäosus | 0,0460 |
| Maksimaalne tõenäosus | 0,9975 |
| Lamapuiduks prognoositud paanid lävendil 0,40 | 172 761 |
| Mitte-lamapuiduks prognoositud paanid lävendil 0,40 | 407 375 |

Tulemuskirjeldus: kogu andmestiku tõenäosusfailis oli tõenäosus olemas kõigil 580 136 paanil. Lävendil 0,40 klassifitseerus 172 761 paani lamapuiduks ja 407 375 paani mitte-lamapuiduks.

Allikas: `data/chm_variants/labels_canonical_with_splits_retrained_ensemble.csv`, `OPTION_B_SPATIAL_SPLITS_COMPARISON.md`.

Tähtis vormistusmärkus: Tabel J sobib tulemuste peatükki alles pärast seda, kui on kinnitatud, et koguandmestiku tõenäosusfail on arvutatud lõplike ruumilise splitiga mudelitega. Praegune skriptitee vajab kontrolli, sest `recalculate_model_probs_tta_ensemble.py` laadib vaikimisi algsed `output/tile_labels/*.pt` mudelid.

### Tabel K. CHM-sisendvariantide klassifitseerimiskatse ulatus

| Näitaja | Väärtus |
|---|---:|
| Kasutatud kirjeid kokku | 17 403 |
| Treeningu ja ristvalideerimise osa | 13 922 |
| Lukustatud testhulga suurus | 3 481 |
| Ristvalideerimise osade arv | 5 |
| Testitud mudelid | ConvNeXt-small, EfficientNet-B2 |
| Testitud lõpetatud CHM-variandid | `original`, `raw`, `gauss` |
| Alustatud, kuid lõpetamata variant olemasolevas logis | `composite_3band` |
| Hindamismõõdik | test-F1 ja test-AUC |

Tulemuskirjeldus: CHM-sisendvariantide klassifitseerimiskatse kasutas ruumilise V4 jaotusega lukustatud testhulka. Olemasolevas `results.csv` failis on lõpetatud tulemused kolmele ühekanalilisele sisendile: originaalne CHM, harmoniseeritud töötlemata CHM ja Gaussi silumisega harmoniseeritud CHM.

Allikas: `scripts/chm_variant_selection.py`, `output/chm_variant_selection/results.csv`, `output/chm_variant_selection/run.log`.

### Tabel L. CHM-sisendvariantide tulemused EfficientNet-B2 mudeliga

| CHM-sisend | Jooksude arv | Keskmine test-F1 | F1 standardhälve | 95% usaldusvahemik | Keskmine test-AUC | AUC standardhälve |
|---|---:|---:|---:|---|---:|---:|
| Gaussi silumisega harmoniseeritud CHM (`gauss`) | 5 | 0,8561 | 0,0100 | [0,8473; 0,8649] | 0,9342 | 0,0054 |
| Originaalne ühekanaliline CHM (`original`) | 10 | 0,8463 | 0,0130 | [0,8382; 0,8544] | 0,9328 | 0,0075 |
| Harmoniseeritud töötlemata CHM (`raw`) | 10 | 0,8121 | 0,0121 | [0,8046; 0,8196] | 0,8970 | 0,0083 |

Tulemuskirjeldus: EfficientNet-B2 mudeliga saavutas kõrgeima keskmise test-F1 väärtuse Gaussi silumisega harmoniseeritud CHM. Originaalse ühekanalilise CHM-i keskmine test-F1 oli 0,8463 ning harmoniseeritud töötlemata CHM-i keskmine test-F1 oli 0,8121.

Allikas: `output/chm_variant_selection/results.csv`.

### Tabel M. CHM-sisendvariantide koondtulemus kõigi lõpetatud mudelijooksude lõikes

| CHM-sisend | Mudelijookse | Keskmine test-F1 | F1 standardhälve | 95% usaldusvahemik | Keskmine test-AUC |
|---|---:|---:|---:|---|---:|
| Gaussi silumisega harmoniseeritud CHM (`gauss`) | 10 | 0,7614 | 0,1000 | [0,6995; 0,8234] | 0,7280 |
| Originaalne ühekanaliline CHM (`original`) | 20 | 0,7232 | 0,1926 | [0,6388; 0,8076] | 0,7185 |
| Harmoniseeritud töötlemata CHM (`raw`) | 20 | 0,6394 | 0,2839 | [0,5150; 0,7639] | 0,6976 |

Tulemuskirjeldus: kõigi lõpetatud ConvNeXt-small ja EfficientNet-B2 jooksude koondis oli kõrgeim keskmine test-F1 samuti Gaussi silumisega CHM-il.

Allikas: `output/chm_variant_selection/results.csv`.

### Märkus CHM-variantide raporti kohta

Fail `output/chm_variant_selection/CHM_VARIANT_SELECTION_REPORT.md` ei ole praegusel kujul piisav lõplikuks tsiteerimiseks, sest see genereeriti enne hilisemate `gauss` tulemuste lisandumist ning kajastab ainult osalist võrdlust. Lõputöös tuleks kasutada kas otse `results.csv` alusel arvutatud tabeleid või genereerida raport uuesti käsuga:

```bash
python3 scripts/chm_variant_selection_analyze.py --results output/chm_variant_selection/results.csv --output output/chm_variant_selection
```

Kui soovitakse võrrelda ka `composite_3band` sisendit, tuleb `scripts/chm_variant_selection.py` jooks lõpuni viia, sest olemasolev `run.log` katkeb `composite_3band` variandi ConvNeXt-small mudeli kolmanda osa järel.

## CHM-variantide täiendamise plaan LaTeX-failides

### `4-metoodika.tex`

1. Säilitada alampeatükk `CHM variantide loomine klassifitseerimismudeli optimeerimiseks`, kuid täiendada seda eraldi lõiguga `Variantide hindamise katsekorraldus`.
2. Lisada katsekorralduse lõiku järgmised andmed: 17 403 kureeritud kirjet, 13 922 treeningu/ristvalideerimise kirjet, 3 481 lukustatud testpaani, 5 ruumilist osa, `StratifiedGroupKFold` kohapõhise rühmitamisega, mudelid ConvNeXt-small ja EfficientNet-B2, 30 epohhi ja varajane peatamine.
3. Täpsustada, et klassifitseerimise olemasolevas lõpetatud võrdluses on täielikud tulemused sisenditele `original`, `raw` ja `gauss`; `composite_3band` logi jäi pooleli ning seda ei tohiks esitada lõpliku klassifitseerimistulemusena enne jooksu lõpetamist.
4. Parandada vastuolu “viis erinevat varianti” ja tegelikult lõpetatud klassifitseerimiskatse vahel. Sobiv sõnastus: “Loodi mitu CHM-varianti; klassifitseerimise sisendikatses on lõplikult dokumenteeritud kolme ühekanalilise variandi tulemused.”
5. Lisada viide katse artefaktidele: `scripts/chm_variant_selection.py`, `output/chm_variant_selection/results.csv`, `output/chm_variant_selection/run.log`.

### `5-tulemused.tex`

1. Asendada praegune alampeatükk `CHM variantide võrdlus klassifitseerimises` tabelipõhise tulemusega.
2. Eemaldada või ümber sõnastada praegune väide, et mitmekanalilised variandid andsid “alla 1% F1 kasvu” ja suurendasid andmemahtu üle kolme korra. Olemasolev lõpetatud klassifitseerimistabel toetab kindlalt `gauss` ja `original` võrdlust EfficientNet-B2 mudeliga; mitmekanalilise variandi tulemus ei ole olemasolevas logis lõpetatud.
3. Lisada tulemuste tabel EfficientNet-B2 kohta: `gauss` F1 0,8561, `original` F1 0,8463, `raw` F1 0,8121. See on kõige selgem tabel, sest EfficientNet-B2 on ka lõpliku ansambli neljas liige.
4. Soovi korral lisada teine väiksem koondtabel kõigi lõpetatud mudelijooksude kohta: `gauss` F1 0,7614, `original` F1 0,7232, `raw` F1 0,6394.
5. Paigutada CHM-variantide tulemus enne lõpliku ansambli treenimise ja kogu andmestiku hindamise lõiku, sest see on sisendandmete valiku katse.
6. Lisada lühike neutraalne tulemuskirjeldus: “Kõrgeim keskmine test-F1 saavutati Gaussi silumisega harmoniseeritud CHM-iga. Originaalne ühekanaliline CHM jäi EfficientNet-B2 võrdluses ligikaudu ühe protsendipunkti võrra madalamale.”
7. Lisada arutelu jaoks eraldi märkus, mitte tulemuste põhiteksti: kui lõplik inferents jäi originaalse `drop13` CHM-i peale, tuleb arutelus selgitada, miks sisendvariandi katse parim tulemus ja lõplik tootmistoru ei ole samad.

## Arutellu suunatavad tähelepanekud

Need punktid ei peaks olema tulemuste peatükis põhijäreldustena, kuid need on olulised arutelu või metoodilise piirangu jaoks.

1. Lõpliku testmetaandmestiku ja lõpliku koguandmestiku CSV diagnostiline F1 ei lange kokku. Enne segadusmaatriksi lisamist tuleb ametlik testhindamine uuesti käivitada nii, et testennustused salvestatakse eraldi faili.
2. Koguandmestiku ümberhindamise skript kasutab vaikimisi algseid pilootmudeleid, mitte ruumilise splitiga salvestatud `*_spatial.pt` mudeleid. Kui see ei olnud taotluslik, tuleb lõplik inferents uuesti arvutada.
3. Käsitsi märgiste diagnostiline alamhulk annab madalama tulemuse kui kogu testhulk. Seda ei tohiks esitada põhitulemusena enne ametliku testennustuse salvestamist, kuid see on arutelus oluline kvaliteedikontrolli teema.
4. Pilootansambli AUC oli suurem kui lõpliku ruumilis-ajalise testi AUC, kuid testandmestikud ei ole omavahel samaväärsed. Seda võrdlust sobib käsitleda arutelus, mitte tulemuste põhitõlgendusena.
5. Välitööde kohta puudub kasutatav kvantitatiivne tulemuste tabel. Kui autoril on kontrollitud alade arv või välitööde kuupäevad, saab lisada lühikese tulemuse; muidu jääb see metoodiliseks piiranguks.
6. CHM-sisendvariantide katse põhjal oli parim lõpetatud üksikmudeli tulemus Gaussi silumisega CHM-il, kuid lõplik tootmistoru paistab kasutavat originaalset `drop13` CHM-i. Kui lõplik inferents ei ole Gaussi variandiga tehtud, tuleb arutelus selgitada praktilise toru ja ablation-katse tulemuse erinevust.

### Arutelu peatüki täienduste paigutus

| Mõte | Paigutus `6-arutelu.tex` peatükis | Staatus |
|---|---|---|
| CHM-sisendvariantide katse parim tulemus oli Gaussi silumisega harmoniseeritud CHM-il, kuid lõplik tootmistoru võib kasutada originaalset `drop13` CHM-i. | `Peamised järeldused ja õppetunnid`; lisaks `LiDAR-andmestiku piirangud`. | Lisatud. |
| Lõplik testkomplekt sisaldab lisaks käsitsi märgistele ka `auto_skip` ja kõrge kindlusega automaatmärgiseid. | Uus alalõik `Klassifitseerimise hindamise ja märgistusallikate piirangud`. | Lisatud. |
| Ametlik testhindamine ja kogu andmestiku tõenäosusfail on erinevad sammud; segadusmaatriks vajab salvestatud testennustusi. | Uus alalõik `Klassifitseerimise hindamise ja märgistusallikate piirangud`. | Lisatud. |
| Pilootansambli ja lõpliku ruumilis-ajalise testi mõõdikud ei ole otse võrreldavad. | Uus alalõik `Klassifitseerimise hindamise ja märgistusallikate piirangud`. | Lisatud. |
| CHM-silumine parandab struktuuride nähtavust, kuid võib vähendada mikrodetaile; mitmekanaliliste sisendite mõju vajab lõpuni viidud katset. | `LiDAR-andmestiku piirangud`. | Lisatud. |
| Kvaliteetset sõltumatut referentsandmestikku oli vähe ning osa tegelikust lamapuidust võib jääda kasutatud ALS/CHM-andmekombinatsioonis nähtamatuks. | Uus alalõik `Referentsandmestiku piirangud`. | Lisatud. |
| Droonipõhise fotogramm-meetria, droonilidari või käsilidariga tuleks koguda täpsemat treening- ja kontrollandmestikku. | `Tuleviku arendused`. | Lisatud. |
| Tulevikus tuleb ühtlustada treeningu, testhindamise ja koguandmestiku inferentsi mudelifailid ning sisendvariandid. | `Tuleviku arendused`. | Lisatud. |
| Tulevikus tuleks salvestada ametliku testhindamise paanipõhised ennustused ja esitada märgistusallikate kaupa tulemused. | `Tuleviku arendused`. | Lisatud. |

## Tabelite soovitatud järjestus

1. `tab:model-search-v1` - esialgne mudeliotsing ja top-5 ansambli esmane edu.
2. `tab:initial-ensemble` - pilootansambli neli mudelit ja TTA ansambel.
3. Uus väike tabel või tekstiplokk - aktiivõppe järjekord: `[0,39; 0,61]`, 29 628 low-confidence, 26 215 spot-check, kokku 55 843.
4. Uus tabel - CHM-sisendvariantide klassifitseerimiskatse: `gauss`, `original`, `raw`; põhitabel EfficientNet-B2 mudeliga.
5. `tab:ensemble-methods-comparison` - Tabel 6, ansambli koosseisu ja hääletusmeetodi võrdlus.
6. Uus lõpptulemuse tabel - lõplik ruumilis-ajaline ansambel: train 67 290, val 13 850, test 56 521, F1 0,9819, AUC 0,9885, threshold 0,4.
7. Soovi korral väike inferentsi kokkuvõtte tabel - 580 136 paani ümberhinnatud, NaN 0, mean prob 0,3757.

## Praegused ebakohad `5-tulemused.tex` failis

Fail: [`LaTeX/Lamapuidu_tuvastamine/estonian/sektsioonid/5-tulemused.tex`](../LaTeX/Lamapuidu_tuvastamine/estonian/sektsioonid/5-tulemused.tex)

- Read 13-37: tulemuste alampeatükk algab kohe tabeliga. Enne peaks olema kontekst ja esialgse märgistuse/mudeliotsingu kirjeldus.
- Read 39-67: Tabel 6 on enne mudeliotsingu seletust ja enne pilootansambli tulemusi. Loogiliselt tuleb see pärast aktiivõppe/suurendatud valimi kirjeldust.
- Rida 39: pealkiri algab väikese tähega ja sõnastus "esialgsete mudelite ja ansambli võrdlus" ei vasta tabeli sisule. Täpsem: "Ansambelmudeli valik ja agregeerimismeetodid".
- Rida 72: viide `\autoref{sec:klassifitseerimismudelite-otsing}` on metoodikas olemas, aga tulemuste tekst võiks enne tabelit mitte hüpata "kolmeetapilise otsinguprotsessi" juurde, kui neid etappe pole veel tulemuste järjekorras avatud.
- Rida 79: "( 15 850 treeningpaani)" sisaldab üleliigset tühikut.
- Read 99-104: lõpliku ansambli tulemus on dubleeritud. Jätta üks lõik.
- Rida 102: lause algab väikese tähega "kinnitatud".
- Read 117-122: CHM variantide lõik on liiga üldine ja asub pärast lõplikku ansamblit. Kui see on klassifitseerimise sisendvariantide ablation, peaks see olema enne lõpliku mudeli treenimist; kui see on segmenteerimise CHM testide üldistatud järeldus, peab see liikuma segmenteerimise tulemuste juurde.
- Read 124-140: vigade analüüs sobib lõppu, kuid vajab sidumist lõpliku ansambli väljunditega ja tegelike näidetega.

## Praegused ebakohad `4-metoodika.tex` failis

Fail: [`LaTeX/Lamapuidu_tuvastamine/estonian/sektsioonid/4-metoodika.tex`](../LaTeX/Lamapuidu_tuvastamine/estonian/sektsioonid/4-metoodika.tex)

- Read 17 ja 23: sama subsection pealkiri on dubleeritud. Tuleks ühendada üheks alampeatükiks. +
- Rida 21: "väiksemal käsitsi märgistatud andmestikul" on liiga lihtsustatud, sest pilootandmestikus oli käsitsi märgistuse ja `auto_skip` segu. + täienda lühidalt
- Read 85-94: aktiivõppe töövoog on sisuliselt õiges kohas, kuid tekst hüppab kohe "kõigi 580 136 paani" hindamisele enne, kui mudeliotsingu ja pilootansambli roll on selgelt eristatud. + täienda pilootala ja mudeliotsingut ning see järel aktiivõppet
- Rida 88: "15 850 käsitsi märgistatud õpepildil" tuleks asendada "15 850 kureeritud õppepaanil", täpsustades käsitsi ja `auto_skip` allikad. + täienda lühidalt
- Rida 91: "ortofoto kaldaerofotod" näib olevat viga. Kui mõeldi kaldaerofotosid, tuleb täpsustada; kui mitte, eemaldada. + täpsusta, et kasutati kaldaerofotosid ja ortofotosid vastavalt vajadusele
- Rida 91: `Google Street View` kasutus vajab kinnitust. Kui seda tegelikult ei kasutatud süstemaatiliselt, siis mitte lisada. k+ kasutati vajadusel jätta alles
- Rida 94: "kasvatas märgistatud paanide baasi 580 136 kirjeni, millest ... puhastati välja 67 290" jätab mulje, et 67 290 on kogu puhastatud andmestik. Tegelikult on 67 290 treeninghulk; lisaks on 13 850 val ja 56 521 test. +  täpsemalt lahti kirjeldada
- Read 111-115: ruumilise puhvri seletus tuleb kooskõlastada sellega, kas lõplik töö kasutab 12,8 m vahet või hilisemat V3 51,2 m soovitust. `SPLIT_ASSIGNMENT_REPORT.md` toetab 12,8 m vahet; `output/spatial_split_experiments/RESULTS.md` soovitab eraldi V3/E07 puhul 51,2 m puhvrit väiksema buffer-waste'i tõttu. Tekst peab valima ühe lõpliku versiooni või selgitama, miks klassifitseerimise lõppmudelis kasutati Option B ja segmenteerimisel eraldi CVv5. + erinevad andmestikud ja kasutati erinevaid lahendusi. kinflasti ei kasutatud  V3/E07 puhul 51,2 m puhvrit.
- Rida 115: sobilikkuse läved `>0,95` ja `<0,05` on kooskõlas `SPLIT_ASSIGNMENT_REPORT.md` ja `scripts/assign_label_splits.py`-ga, kuid mitte `model_search_v4` rangemate lävedega `0,9995` / `0,0698`. Mitte segada neid samasse lõpliku väitesse. + täpsustada mida tegelikult skriptis kasutati ja panna see ka metoodikasse
- Rida 203: lause "Kolme CNN-Deep-Attn baasmudeli kaalud saadi varasemast ruumilise ja ajalise jaotamise treeningandmetel..." tekitab kronoloogilise segaduse. Tuleks sõnastada nii, et Tabel 6 kirjeldab ansambli valiku võrdlust, mitte kogu töövoo ajatelge. + vii sisse täpsustus
- Read 251-255: 5x5 hindamismetoodika on olemas, kuid tuleb eristada ansambli valiku võrdlust lõpliku mudeli põhihindamisest sõltumatul testandmestikul. + täienda lühidalt
- Read 257-261: "rakendati ... kogu uuringualal sh. testandmestikule" on sõnastuslikult riskantne. Testandmestikku kasutatakse mõõdikute arvutamiseks, kogu uuringuala inferents on eraldi samm pärast mudeli lukustamist.
- Segmenteerimise metoodikas read 280-288 räägivad kahest foldist ja triipudest, kuid tulemuste peatükis on CVv5 kolme ruumilise foldiga. Need peavad olema omavahel kooskõlla viidud. + juba viidud kasutati CVv5
- Read 295-296: Faas 4 on enumerate-loendis dubleeritud. Üks neist tuleb eemaldada või liita. + tehtud

## Soovitatud tekstiline sild Tabel 6 jaoks

LaTeX-i sobiv lõik võiks olla selline:

```tex
\subsubsection{Ansambelmudeli valik ja agregeerimismeetodid}
\label{ansambli-agregeerimise-tulemused}

Pärast aktiivõppe käigus laiendatud märgiste kogumi moodustamist võrreldi
neljanda ansambliliikme ja ennustuste koondamise alternatiive. Võrdlus vastab
otseselt küsimusele, milline arhitektuur sobib kolme CNN-Deep-Attn mudeli
kõrvale neljandaks liikmeks ning kas ennustusi tuleks koondada lihtsa pehme
hääletuse, kaalutud hääletuse või virnaansambli abil. Võrdlus viidi läbi
5x5 ristvalideerimisega, kus iga meetodi kohta saadi 25 sõltumatut hinnangut.

Parima üldise tulemuse saavutas EfficientNet-B2 koos kaalutud hääletusega:
F1-skoor $0{,}9866 \pm 0{,}0042$, 95\% usaldusvahemik
[0,9849; 0,9883] ja AUC $0{,}9810$
(Tabel~\ref{tab:ensemble-methods-comparison}). Kaalutud hääletus oli
logistilise regressiooni virnaansambliga praktiliselt samaväärne, kuid
arvutuslikult lihtsam ja tõlgendatavam. Seetõttu kasutati lõpliku
klassifitseerimisahela kirjelduses neljamudelist ansamblit
3x CNN-Deep-Attn + EfficientNet-B2 ning kaalutud/kalibreeritud
tõenäosuste koondamist.
```


## Üks võimalik alampeatüki kondikava

```tex
\subsection{Klassifitseerimise tulemused}
\label{klassifitseerimise-tulemused}

\subsubsection{Esialgne märgistusvalim ja mudeliotsing}
% käsitsi + auto_skip kontekst, seejärel Tabel model-search-v1

\subsubsection{Pilootansambel ja kõigi paanide esmane hindamine}
% Tabel initial-ensemble, seejärel model_prob kogu andmestikule

\subsubsection{Aktiivõppe valim ja märgiste kvaliteet}
% [0,39;0,61], 5% juhuvalim, spotcheck/low-confidence tulemused

\subsubsection{Ansambelmudeli valik ja agregeerimismeetodid}
% Tabel 6 ehk tab:ensemble-methods-comparison

\subsubsection{Ruumilis-ajaline jaotus ja lõpliku ansambli testitulemus}
% 67 290 / 13 850 / 56 521, F1 0,9819, AUC 0,9885

\subsubsection{Kogu andmestiku lõplik ümberhindamine}
% 580 136 paani, retrained ensemble CSV, NaN=0, võrdlusstatistika

\subsubsection{Vigade analüüs}
% FP/FN ja konkreetsed näited
```

## Allikate kiirnimekiri

Peamised tulemuste allikad:

- [`analysis/model_search_v1_ranked_test.csv`](../analysis/model_search_v1_ranked_test.csv)
- [`analysis/model_search_v1_ranked_cv.csv`](../analysis/model_search_v1_ranked_cv.csv)
- [`output/tile_labels/ensemble_meta.json`](../output/tile_labels/ensemble_meta.json)
- [`output/tile_labels/train_ensemble.log`](../output/tile_labels/train_ensemble.log)
- [`docs/THESIS_DROP13_FILTERED_CHM_PIPELINE.md`](THESIS_DROP13_FILTERED_CHM_PIPELINE.md)
- [`output/onboarding_labels_v2_drop13_standardized/summary.json`](../output/onboarding_labels_v2_drop13_standardized/summary.json)
- [`analysis/onboarding_new_laz/drop13_prediction_confidence_recommendation.json`](../analysis/onboarding_new_laz/drop13_prediction_confidence_recommendation.json)
- [`output/onboarding_labels_v2_drop13/spotcheck_metrics_summary.json`](../output/onboarding_labels_v2_drop13/spotcheck_metrics_summary.json)
- [`SPLIT_ASSIGNMENT_REPORT.md`](../SPLIT_ASSIGNMENT_REPORT.md)
- [`output/ensemble_4th_model_comparison_v2/comparison_v2_results.json`](../output/ensemble_4th_model_comparison_v2/comparison_v2_results.json)
- [`output/ensemble_4th_model_ablation/ablation_summary.txt`](../output/ensemble_4th_model_ablation/ablation_summary.txt)
- [`output/tile_labels_spatial_splits/training_metadata.json`](../output/tile_labels_spatial_splits/training_metadata.json)
- [`OPTION_B_SPATIAL_SPLITS_SUMMARY.md`](../OPTION_B_SPATIAL_SPLITS_SUMMARY.md)
- [`OPTION_B_SPATIAL_SPLITS_COMPARISON.md`](../OPTION_B_SPATIAL_SPLITS_COMPARISON.md)
- [`data/chm_variants/labels_canonical_with_splits_retrained_ensemble.csv`](../data/chm_variants/labels_canonical_with_splits_retrained_ensemble.csv)
- [`output/chm_variant_selection/CHM_VARIANT_SELECTION_REPORT.md`](../output/chm_variant_selection/CHM_VARIANT_SELECTION_REPORT.md)
- [`output/chm_variant_selection/results.csv`](../output/chm_variant_selection/results.csv)
- [`output/chm_variant_selection/run.log`](../output/chm_variant_selection/run.log)

Peamised metoodika ja skripti allikad:

- [`scripts/train_ensemble.py`](../scripts/train_ensemble.py)
- [`scripts/label_all_rasters.py`](../scripts/label_all_rasters.py)
- [`scripts/recalculate_manual_review_queue.py`](../scripts/recalculate_manual_review_queue.py)
- [`scripts/assign_label_splits.py`](../scripts/assign_label_splits.py)
- [`scripts/ensemble_4th_model_comparison_v2.py`](../scripts/ensemble_4th_model_comparison_v2.py)
- [`scripts/retrain_ensemble_spatial_splits.py`](../scripts/retrain_ensemble_spatial_splits.py)
- [`scripts/test_evaluate_spatial_splits.py`](../scripts/test_evaluate_spatial_splits.py)
- [`scripts/recalculate_model_probs_tta_ensemble.py`](../scripts/recalculate_model_probs_tta_ensemble.py)
- [`scripts/postprocess_spatial_split_retraining.py`](../scripts/postprocess_spatial_split_retraining.py)
- [`scripts/chm_variant_selection.py`](../scripts/chm_variant_selection.py)
- [`scripts/chm_variant_selection_analyze.py`](../scripts/chm_variant_selection_analyze.py)

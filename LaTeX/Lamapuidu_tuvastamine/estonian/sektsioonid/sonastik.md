# Mõistete sõnastik

## Üldmõisted

| Eestikeelne termin | Ingliskeelne termin | Lühend | Selgitus |
| :--- | :--- | :--- | :--- |
| andmestik | dataset | - | ühenatud andmete kogum |
| töövoog / andmetöötlusahel | pipeline | - | järjestikused töötlusetapid |
| välistamisanalüüs | ablation study | - | komponentide järkjärguline eemaldamine mõju hindamiseks |
| kihistatud | stratified | - | klassidega tasakaalustatud jaotamine |
| ristvalideerimise osa | fold | - | andmestiku jagamine ristvalideerimiseks |
| andmeleke | data leakage | - | treening- ja testandmete vaheline andmete segunemine |

## Andmete ettevalmistus ja valideerimise meetodid

| Eestikeelne termin | Ingliskeelne termin | Lühend | Selgitus |
| :--- | :--- | :--- | :--- |
| ruumiline valideerimise | spatial validation | - | ruumiliselt eraldatud andmetega testimine |
| kaks parimat lahendust | two best solutions / top-2 carryover | - | parimad kandidaadid, mis liiguvad järgmisse etappi |
| juhuslikkuse lähteväärtus | random seed | - | juhuslike arvude generaatori algväärtus |
| bootstrap | bootstrap | - | järelproovide meetodil usaldusvahemike arvutamine |
| paaritud t-test | paired t-test | - | sõltuvate gruppide võrdlemise test |
| Wilcoxoni test | Wilcoxon signed-rank test | - | mitteparameetriline statistiline test |

## LiDAR andmed ja rasterid

| Eestikeelne termin | Ingliskeelne termin | Lühend | Selgitus |
| :--- | :--- | :--- | :--- |
| aerolaserskaneerimine | airborne laser scanning | ALS | õhust tehtava 3D kaardistamise meetod |
| valguse detektor ja kauguse mõõtmine | Light Detection and Ranging | LiDAR | laserimpulsside abil töötav kaardistamissüsteem |
| madal taimkatte kõrgusmudelid | Canopy Height Model | CHM | maapinnast puude kõrguse näitav raster |
| kõrgus maapinnast | Height Above Ground | HAG | maapinnale normaliseeritud kõrgus |
| maapinnamudel | Digital Terrain Model | DTM | maapinna kõrguste raster |
| riidesimulatsiooni filter | Cloth Simulation Filter | CSF | maapinna eristamise algoritm |
| mediaani absoluutne hälve | Median Absolute Deviation | MAD | anomaaliaid eemaldav statistiline meetod |
| bilineaarne interpoleerimise | bilinear resampling | - | rasterite resolutsiooni muutmise meetod |
| pildikildud / paanid | tiles | - | suuremale rasterfailile jagatud osad |

## Sügavõpe — üldmõisted

| Eestikeelne termin | Ingliskeelne termin | Lühend | Selgitus |
| :--- | :--- | :--- | :--- |
| sügavõpe | deep learning | - | mitmekihiliste närvivõrkudega masinõppimine |
| konvolutsiooniline närvivõrk | convolutional neural network | CNN | pildiobjektide tuvastamiseks spetsialiseerunud võrgutüüp |
| tüvimudel / selgroog | backbone | - | närvivõrgu põhiarhitektuur |
| siirdeõpe | transfer learning | - | eelmist mudelit kasutava uue ülesande lahendamine |
| baasmudel / algtase | baseline | - | võrdlusalus alternatiividele |

## Sügavõpe — treeningparameetrid

| Eestikeelne termin | Ingliskeelne termin | Lühend | Selgitus |
| :--- | :--- | :--- | :--- |
| partii | batch | - | ühes treeningsammus töödeldavate näidiste kogum |
| õppimismäär | learning rate | - | treeningsammu suuruse määraja |
| juhuslick väljalülitus | dropout | - | närvivõrgu regulariseerimise meetod |
| märgiste silumist | label smoothing | - | sihtvärtuste pehmendamine ülespetsiifilisuse vähendamiseks |
| MixUp andmesuurendus | MixUp augmentation | - | treeningpiltide segu andmesuurendus |
| juhuslik kaalude keskmistamine | stochastic weight averaging | SWA | närvivõrgu parameetrite keskmistamine |
| testiaegne andmerikastamine | test-time augmentation | TTA | piltide variatsioonide keskmistamine testimisel |

## Sügavõpe — närvivõrgu komponendid

| Eestikeelne termin | Ingliskeelne termin | Lühend | Selgitus |
| :--- | :--- | :--- | :--- |
| partii-normaliseerimine | batch normalization | BatchNorm | võrgustiku kihi sisendite normaliseerimine |
| alaldatud lineaarüksus | rectified linear unit | ReLU | aktivatsioonifunktsioon |
| tähelepanuplokk | attention block | AttnBlock | mudeli teatud osale fokuseerimise mehhanism |
| suru ja erguta mehhanism | squeeze-and-excitation | SE | kanali kaalumise mehhanism |
| jääkühendit | residual connection | - | otsene ühendus kahe kihi vahel |
| maksimumkoondamine | max pooling | - | piirkonna maksimaalse väärtuse valik |
| adaptiivsest keskmistavast koondamine | adaptive average pooling | - | muutuva suurusega sisendi liitviimine |
| tasandamine | flattening | - | multidimensionaalsete andmete muutmine vektoriks |
| klassifikaatorkiht | classifier layer | - | otsuse langetamise lõplik kiht |

## Sügavõpe — kaofunktsioonid

| Eestikeelne termin | Ingliskeelne termin | Lühend | Selgitus |
| :--- | :--- | :--- | :--- |
| ristentroopia kao | cross-entropy loss | - | klassifitseerimise veafunktsioon |
| fookuskao | focal loss | - | andmekaldekat käsitlev veafunktsioon |
| DiceFocal | DiceFocal loss | - | Dice'i ja Focal loss'i kombinatsioon |
| Tversky kao | Tversky loss | - | tasakaalustatud segmenteerimise veafunktsioon |

## Sügavõpe — mudeliarhitektuurid

| Eestikeelne termin | Ingliskeelne termin | Lühend | Selgitus |
| :--- | :--- | :--- | :--- |
| U-Net++ | U-Net++ | - | semantilise segmenteerimise närvivõrgu arhitektuur |
| DeepLabV3+ | DeepLabV3+ | - | atrous convolution'i kasutav segmenteerimise mudel |
| EfficientNet-B2 | EfficientNet-B2 | - | skaleeritud ja tõhus närvivõrgu arhitektuur |
| ResNet | ResNet | - | jääkühendeid kasutav närvivõrgu arhitektuur |
| ConvNeXt | ConvNeXt | - | modernsete konvolutsiooniliste mudelite perekond |
| Swin Transformer | Swin Transformer | - | tähelepanu-põhine närvivõrgu arhitektuur |
| DenseNet | DenseNet | - | tihedalt ühendatud närvivõrgu arhitektuur |
| MobileNet | MobileNet | - | ressursse säästev närvivõrgu arhitektuur |
| RegNet | RegNet | - | regulariseeritud närvivõrgu mudelite perekond |
| ImageNet | ImageNet | - | suure pildide andmestik mudelite eeltreenimiseks |

## Andmete aggregeerimine ja ennustamine

| Eestikeelne termin | Ingliskeelne termin | Lühend | Selgitus |
| :--- | :--- | :--- | :--- |
| pehme hääletus | soft voting | - | mudeli väljundite keskmistamine |
| kaalutud hääletus | weighted voting | - | mudeli väljundite kaalutud keskmistamine |
| virnaansambel | stacking | - | metamudeli kasutamine ennustuste kombineerimiseks |
| metamudel | meta-learner | - | muude mudelite väljundeid kombineeriv mudel |
| pehmed sihtväärtused | soft targets | - | tõenäosusvahemikus [0,1] olevad sihtvärtused |

## Klassifitseerimine ja segmenteerimine

| Eestikeelne termin | Ingliskeelne termin | Lühend | Selgitus |
| :--- | :--- | :--- | :--- |
| pikslitasandi klassifitseerimine | pixel-level classification | - | üksikute pikslite määramine klasside vahel |
| semantiline segmenteerimine | semantic segmentation | - | objektide ruumiliste piiride tuvastamine |
| eraldatud testvalim | held-out test set | - | treenimisest täielikult eraldatud testandmed |
| automatimine kalded | automation bias | - | ülimatu sõltuvus algoritmi ennustustest |

## Mudeli seletavus ja interpreteeritavus

| Eestikeelne termin | Ingliskeelne termin | Lühend | Selgitus |
| :--- | :--- | :--- | :--- |
| mudeli selgitatavus | explainable AI | XAI | mudeli otsustusprotsessi seletavus |
| Integrated Gradients | Integrated Gradients | IntGrad | gradiendipõhine seletavuse meetod |
| HiResCAM | HiResCAM | - | kõrge resolutsiooniga tähelepanu visualiseerimine |
| GradCAM+ | GradCAM+ | - | parandatud gradient-põhine klassiaktiveerimiskaart |
| RISE | Randomized Input Sampling for Explanation | - | juhuslikul sisendite maskeeringul põhinev seletavus |
| soojuskaart | heatmap | - | värvuskoodiga visualiseeritud intensiivsus |

## Hindamismõõdikud

| Eestikeelne termin | Ingliskeelne termin | Lühend | Selgitus |
| :--- | :--- | :--- | :--- |
| F1-skoor | F1-score | - | täpsuse ja saagise harmoonilise keskmise |
| täpsus | precision | - | õigesti leitud positiiviste osakaal |
| saagis / tundlikkus | recall / sensitivity | - | avastatud positiiviste osakaal |
| spetsiifilisus | specificity | - | õigesti märgistatud negatiivsete osakaal |
| AUC | Area Under the Curve | AUC | ROC-kõvera aluse pindala |
| ROC-kõver | ROC curve | - | tõetõenäosuse ja valenegatiivsuse suhte kõver |
| saagis-täpsuse kõver | Recall-Precision curve | PR curve | tuvastuse ja täpsuse vaheline kompromiss |
| Dice-koefitsient | Dice coefficient | - | segmenteerimise täpsuse mõõdik |
| clDice | centerline Dice | clDice | lineaarsete objektide topograafilise korrektsuse mõõdik |
| kattuva ja ühendatud ala suhe | Intersection over Union | IoU | objektide kattuvuse mõõdik |

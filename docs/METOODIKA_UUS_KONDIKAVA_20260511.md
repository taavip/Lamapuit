# Metoodika peatuki uus kondikava

**Fail:** `LaTeX/Lamapuidu_tuvastamine/estonian/sektsioonid/4-metoodika.tex`  
**Eesmärk:** ümber järjestada olemasolev metoodika nii, et ükski tööosa ei kaoks, kuid peatükk räägiks töö loogilises uurimisjärjekorras.

## Põhimõte

Metoodika peatükk peaks liikuma samas järjekorras nagu töö tegelikult arenes:

1. klassifitseerimise andmestiku loomine;
2. sildistamistööriist ja aktiivõpe;
3. ruumilis-ajaline train/val/test jaotus;
4. CHM-ide tootmine ja variantide valik;
5. klassifitseerimismudelite otsing, ansambel ja lõplik rakendamine;
6. eraldi segmenteerimise metoodika.

Oluline on mitte kustutada olemasolevaid osi. Praegune tekst tuleks peamiselt ümber tõsta ja jagada väiksemateks, selgema funktsiooniga alapeatükkideks.

## Soovitatud uus pealkirjastruktuur

```tex
\section{Metoodika}
\label{metoodika}

\subsection{Klassifitseerimise metoodika üldskeem}

\subsection{Klassifitseerimisandmete sildistamine}
\subsubsection{Sildistamistööriist}
\subsubsection{Sildistamisotsuse andmekihid}
\subsubsection{Mudeli selgitatavuse soojuskaardid}
\subsubsection{Märgistuse kvaliteedikontroll}

\subsection{Aktiivõppe töövoog klassifitseerimisel}
\subsubsection{Esialgne käsitsi märgistatud andmestik}
\subsubsection{Esialgne märgistamisansambel}
\subsubsection{Kogu andmestiku hindamine}
\subsubsection{Ebakindlate ja kontrollvalimi paanide prioriseerimine}
\subsubsection{Uute siltidega mudeli parandamine}

\subsection{Klassifitseerimisandmestiku ruumiline ja ajaline jaotamine}
\subsubsection{Andmelekke probleem}
\subsubsection{Sobilikkuse kriteeriumid}
\subsubsection{Ruumiline isoleerimisstrateegia}
\subsubsection{Ajalise järjepidevuse tagamine}
\subsubsection{Lõplik train/val/test jaotus}
\subsubsection{Jaotuse piirangud}

\subsection{CHM-ide tootmine ja sisendvariantide ettevalmistus}
\subsubsection{Lähteandmed ja kõrguste normaliseerimine}
\subsubsection{DTM-i arvutamise variandid}
\subsubsection{CHM-variantide loomine}
\subsubsection{CHM-variantide hindamise kriteerium}

\subsection{Klassifitseerimismudeli otsing ja lõplik ansambel}
\subsubsection{Mudeliarhitektuuride otsing}
\subsubsection{Ansambli liikme ja agregeerimismeetodi võrdlus}
\subsubsection{Hindamisprotokoll ja statistilised testid}
\subsubsection{Lõpliku ansambli koosseis}
\subsubsection{Lõpliku klassifitseerimismudeli rakendamine}

\subsection{Segmenteerimise metoodika}
\subsubsection{Segmenteerimisandmestiku valik}
\subsubsection{Segmenteerimisandmete sildistamine}
\subsubsection{Segmenteerimismudelite ja parameetrite valik}
\subsubsection{Ristvalideerimine ja sõltumatu hindamine}
\subsubsection{Lõplike tõenäosuskaartide arvutamine}
```

## Ümbertõstmise kaart olemasolevast failist

### 1. Klassifitseerimise metoodika üldskeem

**Võta olemasolevast tekstist:**

- peatüki sissejuhatus;
- joonis `fig:workflow`;
- praeguse esimese alampeatüki esimene lõik, mis kirjeldab iteratiivset töövoogu.

**Kirjutamise mõte:** anda lugejale kohe teada, et klassifitseerimine ja segmenteerimine olid kaks seotud, kuid eraldi töövoogu.

### 2. Klassifitseerimisandmete sildistamine

**Võta olemasolevast tekstist:**

- praegune `\subsection{Andmete sildistamine klasifitseerimis ülesande jaoks ja aktiivõppe töövoog}`;
- sildistamistööriista kirjeldus;
- 128 x 128 paanid, klaviatuuriga märkimine, CSV-väljund;
- andmekihtide loetelu;
- joonised `fig:labeling-ui` ja `fig:labeling-heatmap`;
- `\paragraph{Mudeli selgitatavuse meetodid: soojuskaartide ülevaade.}`;
- `\paragraph{Märgistuse kvaliteedikontroll.}`.

**Uus jaotus:**

- `Sildistamistööriist`: tehniline tööriist ja töövoog.
- `Sildistamisotsuse andmekihid`: CHM, ortofoto, ajaloolised CHM-id, WMS, välitööde piirang.
- `Mudeli selgitatavuse soojuskaardid`: IntGrad, HiResCAM, GradCAM+, RISE.
- `Märgistuse kvaliteedikontroll`: automation bias, ebakindel vahemik, 5% kontrollvalim.

**Märkus:** praegune tekst ütleb kohati `klasifitseerimis ülesande`; hiljem võiks parandada kujule `klassifitseerimisülesande`.

### 3. Aktiivõppe töövoog klassifitseerimisel

**Võta olemasolevast tekstist:**

- `\paragraph{Esialgne märgistamisansambel.}`;
- aktiivõppe taust praegusest esimesest alampeatükist;
- kvaliteedikontrolli osast ebakindlate paanide valik.

**Lisa või täpsusta hiljem:**

- esialgne käsitsi märgistatud andmestik: 15 850 treeningpilti, 3 962 valideerimispilti, 2 186 testpilti;
- kui palju oli käsitsi märgistatud ja kui palju auto-skip/inimese kinnitatud;
- kogu 119 kaardilehe hindamine esialgse ansambliga;
- prioriseerimine: ebakindlusvahemik 0,3-0,7 ja 5% juhuslik kontrollvalim;
- uute siltidega teise ansambli treenimine;
- ülejäänud kaardilehtede automaatne sildistamine teise ansambliga;
- selge sõnastus, et tegemist on pooljuhendatud töövooga.

**Soovitatud narratiiv:**

Alguses ei olnud võimalik kõiki 119 kaardilehte käsitsi sildistada. Seetõttu kasutati väiksemal andmestikul treenitud ansamblit tööriistana, mis hindas kogu andmestiku ja suunas inimese tähelepanu sinna, kus mudel oli ebakindel. See teeb aktiivõppe osa loogiliseks, mitte ei jäta muljet, et andmestik tekkis korraga valmis.

### 4. Klassifitseerimisandmestiku ruumiline ja ajaline jaotamine

**Võta olemasolevast tekstist:**

- praegune `\subsection{Andmestiku ruumiline ja ajaline jaotamine}`;
- kõik selle alampeatükid;
- tabel `tab:split-distribution`;
- joonis `fig:spatial-validation`.

**Uus jaotus:**

- `Andmelekke probleem`: ruumiline ja ajaline leke.
- `Sobilikkuse kriteeriumid`: manual, auto_skip, väga kõrge/madal mudelitõenäosus.
- `Ruumiline isoleerimisstrateegia`: stride, Tšebõšovi kaugus, testtsoon ja puhvertsoon.
- `Ajalise järjepidevuse tagamine`: sama asukoha eri aastate hoidmine samas rollis.
- `Lõplik train/val/test jaotus`: tabel.
- `Jaotuse piirangud`: 12,8 m puhvri piirang ja 50 m võrdlus.

**Tähelepanu vajav koht:**

Praeguses tekstis on juba kirjas nii 51,2 m testtsoon kui 12,8 m puhver. Seda ei pea kustutama, aga sõnastus peab olema väga täpne:

- 51,2 m kirjeldab testtsooni ligikaudset mõõtu;
- 12,8 m kirjeldab minimaalset vahet testi- ja treeningpikslite vahel;
- seda tuleb mitte nimetada ekslikult 51,2 m puhvriks.

### 5. CHM-ide tootmine ja sisendvariantide ettevalmistus

**Võta olemasolevast tekstist:**

- praegune `\subsection{Kõrgusmudeli variantide loomine mudeli treenimiseks}`;
- kõik viis CHM-varianti;
- `\paragraph{Maapinna kõrgusmudelite (DTM) arvutamise erinevused ja nende tehniline teostus.}`.

**Soovitatud asukoht:** enne mudeliotsingut, sest mudeliotsing sõltub sellest, millised sisendid mudelile anti.

**Uus jaotus:**

- `Lähteandmed ja kõrguste normaliseerimine`: LAZ, HAG/CHM, 20 cm resolutsioon.
- `DTM-i arvutamise variandid`: baasmudel vs harmoniseeritud CSF/MAD/interpoleeritud DTM.
- `CHM-variantide loomine`: baseline, harmonized raw, harmonized Gaussian, composite 4-band, masked raw 2-band.
- `CHM-variantide hindamise kriteerium`: millise mõõdikuga valiti parim variant.

**Tähelepanu vajav koht:**

Analüüsidokumentides on vastuolu: kas testiti viis erinevat CHM-meetodit või viis sama meetodi parameetrivarianti. Kondikavas jäta ruum mõlemale, aga lõplikus tekstis tuleb valida üks tõene versioon ja panna tabel:

```tex
\begin{table}[h!]
\centering
\caption{Testitud CHM-variantide parameetrid ja valikutulemused.}
\label{tab:chm-variants}
\begin{tabular}{llll}
\hline
Variant & Põhimeetod & Parameetrid & Valikumõõdik \\
\hline
... & ... & ... & ... \\
\hline
\end{tabular}
\end{table}
```

### 6. Klassifitseerimismudeli otsing ja lõplik ansambel

**Võta olemasolevast tekstist:**

- praegune `\subsection{Mudeliarhitektuuride otsing ja valik}`;
- praegune `\subsection{Ansambelmudeli koostamine ja valideerimine}`;
- praegune `\subsection{Lõpliku mudeli rakendamine}`;
- tabel `tab:model-architectures`;
- joonis `fig:ensemble-scheme`.

**Uus jaotus:**

- `Mudeliarhitektuuride otsing`: 35 katset, 16 arhitektuuri, CNN-Deep-Attn ja eelõpitud mudelid.
- `Ansambli liikme ja agregeerimismeetodi võrdlus`: EfficientNet-B2, ConvNeXt-tiny, ConvNeXt-small; soft voting, weighted voting, stacking.
- `Hindamisprotokoll ja statistilised testid`: 5 x 5 ristvalideerimine, bootstrap, Wilcoxon.
- `Lõpliku ansambli koosseis`: 3 x CNN-Deep-Attn + EfficientNet-B2.
- `Lõpliku klassifitseerimismudeli rakendamine`: kogu uuringuala tõenäosused.

**Märkus:** esialgne märgistamisansambel peaks jääma aktiivõppe peatükki. Siin peatükis räägitakse lõplikust mudelivalikust ja lõplikust ansamblist. Nii väldime kordust.

### 7. Segmenteerimise metoodika

**Praeguses `4-metoodika.tex` failis see osa sisuliselt puudub.**

**Võta mujalt või kirjuta juurde:**

- andmepeatükis mainitud kaardileht `406455_2021_tava`;
- eraldi sildistamine segmenteerimiseks;
- 1 236 CWD märgise definitsioon;
- 2-fold ristvalideerimise põhjendus;
- testitud mudelid, CHM-variandid, kaofunktsioonid ja augmentatsioonid;
- parima mudeli valimise mõõdik, nt Dice, precision, recall;
- lõplike tõenäosuskaartide arvutamine.

**Soovitatud alampeatükid:**

- `Segmenteerimisandmestiku valik`: miks üks eraldi kaardileht ja miks mitte samad 119 klassifitseerimise kaardilehte.
- `Segmenteerimisandmete sildistamine`: kas märgiti objektid, pikslimaskid või paanid.
- `Segmenteerimismudelite ja parameetrite valik`: mudelid, sisendid, kaofunktsioonid, augmentatsioonid.
- `Ristvalideerimine ja sõltumatu hindamine`: 2-fold CV, ruumiline jaotus, testosa.
- `Lõplike tõenäosuskaartide arvutamine`: mudeli väljundid ja nende kasutus.

## Vana ja uus järjestus lühidalt

| Praegune asukoht | Uus asukoht | Tegevus |
|---|---|---|
| Sissejuhatus ja töövoo joonis | Klassifitseerimise metoodika üldskeem | Säilitada, lühendada ja fokuseerida |
| Sildistamine + aktiivõpe ühes peatükis | Jagada kaheks: sildistamine ja aktiivõpe | Ümber tõsta, mitte kustutada |
| XAI soojuskaardid | Sildistamise alampeatükk | Säilitada |
| Esialgne märgistamisansambel | Aktiivõppe alampeatükk | Säilitada ja siduda 119 kaardilehe hindamisega |
| Ruumiline ja ajaline jaotamine | Pärast aktiivõpet | Säilitada, täpsustada 12,8 m vs 51,2 m |
| Mudeliarhitektuuride otsing | Pärast CHM-peatükki | Tõsta hilisemaks |
| Ansambelmudeli koostamine | Klassifitseerimismudeli otsingu alla | Säilitada |
| Lõpliku mudeli rakendamine | Klassifitseerimismudeli otsingu lõppu | Säilitada |
| CHM-variantide loomine | Enne mudeliotsingut | Tõsta ettepoole |
| Segmenteerimine | Uus alampeatükk lõppu | Juurde kirjutada |

## Kohad, mida enne lõplikku kirjutamist tuleb täpsustada

1. Kas CHM katsed olid viis erinevat sisendmeetodit või viis ühe meetodi parameetrivarianti?
2. Millised olid CHM variantide täpsed parameetrid ja tulemused?
3. Kui palju lõplikust 67 290 treeningpaanist oli käsitsi kinnitatud ja kui palju teise ansambli abil automaatselt sildistatud?
4. Kuidas täpselt põhjendada 12,8 m minimaalset puhvrit võrreldes 50 m autokorrelatsiooni soovitusega?
5. Mida tähendab segmenteerimise `1 236 CWD`: objektid, pikslimaskid, paanid või midagi muud?
6. Mis oli segmenteerimise 2-fold CV ruumiline jagamisloogika?
7. Milline oli segmenteerimise lõplik mudel ja valikumõõdik?

## Soovitatud järgmine tööetapp

Järgmises etapis võiks teha `4-metoodika.tex` sees ainult struktuurse ümbertõstmise:

1. luua uued pealkirjad;
2. tõsta olemasolevad lõigud uute pealkirjade alla;
3. lisada TODO-kommentaarid kohtadesse, kus täpsed arvud puuduvad;
4. mitte veel tugevalt ümber kirjutada teaduslikku sõnastust.

Selline vaheetapp annab peatükile õige luustiku ja teeb hilisema sisulise parandamise palju lihtsamaks.

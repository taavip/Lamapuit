import re

TEX_FILE = "LaTeX/Lamapuidu_tuvastamine/estonian/sektsioonid/5-tulemused.tex"

with open(TEX_FILE, "r", encoding="utf-8") as f:
    content = f.read()

new_content = """\\subsection{Klassifitseerimise tulemused}
\\label{klassifitseerimise-tulemused}

\\subsubsection{Käsitsi paanimärgistus ja automaatne filtreerimine}
\\label{kasitsi-paanimargistus}

Klassifitseerimise andmestiku ettevalmistamise käigus märgistati pilootfaasis esimesed paanid käsitsi. Täiendavalt kasutati andmestiku mahu efektiivseks suurendamiseks loogikat, millega visuaalselt tühjad ja ilma lamapuiduta (näiteks puhta noorendiku või lageda maaga) paanid sildistati automaatselt negatiivseks (\\texttt{auto\\_skip}). Pilootansambli hindamiseks saadi seeläbi kokku 21\\,998 silti (sh 3\\,380 lamapuiduga ja 18\\,618 negatiivset paani). Neist 15\\,850 kureeritud õppepaani suunati piloottreeningusse.

\\subsubsection{Mudeliotsing ja arhitektuuride võrdlus}
\\label{mudeli-otsinguprotsess}

Enne lõpliku ansambli ja prognoosimise juurde liikumist teostati laiendatud mudeliotsing mitmesuguste tehisnärvivõrkude arhitektuuride üle (19\\,812 treeningpaani, 11 rasterikihti). Tabelist~\\ref{tab:model-search-v1} järeldub, et CNN-Deep-Attn perekonna mudelid saavutasid üksikmudelitena testis kõige stabiilsemaid tulemusi. Samas demonstreeris virnaansambel asjaolu, et ansamblistrateegia annab antud ülesandes järjekindlalt parema tulemuse (F1-skoor 0,9672) kui parim üksikmudel (CNN-Deep-Attn-headwide: 0,9644).

\\begin{table}[h!]
\\centering
\\caption{Parima mudeli otsingu tulemusnäitajad (19\\,812 treeningpaani). Testvalimi tulemused põhinevad 2\\,186 paanil; ${}^{\\dagger}$ tähistab 5-kordse ristvalideerimise keskmist.}
\\label{tab:model-search-v1}
\\begin{tabular}{lllll}
\\hline
\\textbf{Mudel / ansambel} & \\textbf{AUC} & \\textbf{F1} & \\textbf{Täpsus} & \\textbf{Tagasikutse} \\\\
\\hline
\\multicolumn{5}{l}{\\textit{Testvalimi tulemused}} \\\\
\\hline
Virnaansambel (top-5)       & 0,9982 & 0,9672 & 0,9789 & 0,9558 \\\\
Pehme hääletus (top-5)      & 0,9982 & 0,9646 & 0,9646 & 0,9646 \\\\
CNN-Deep-Attn-headwide      & 0,9977 & 0,9644 & 0,9701 & 0,9587 \\\\
CNN-Deep-Attn               & 0,9973 & 0,9630 & 0,9673 & 0,9587 \\\\
CNN-Deep-Attn-headlight     & 0,9974 & 0,9614 & 0,9672 & 0,9558 \\\\
\\hline
\\multicolumn{5}{l}{\\textit{Parimad alternatiivarhitektuurid, ristvalideerimine (${}^{\\dagger}$)}} \\\\
\\hline
ConvNeXt-small${}^{\\dagger}$  & 0,9922 & 0,9546 & 0,9677 & 0,9420 \\\\
ConvNeXt-tiny${}^{\\dagger}$   & 0,9935 & 0,9523 & 0,9626 & 0,9427 \\\\
EfficientNet-B2${}^{\\dagger}$ & 0,9953 & 0,9493 & 0,9546 & 0,9440 \\\\
DenseNet-121${}^{\\dagger}$    & 0,9912 & 0,9454 & 0,9640 & 0,9276 \\\\
\\hline
\\end{tabular}
\\end{table}

\\subsubsection{Pilootansambel ja paanide esmane hindamine}
\\label{esialgse-ansambli-tulemused}

Tugevamate arhitektuuride selgumise järel moodustati neljamudeline pilootansambel (kolm \\mbox{CNN-Deep-Attn} baasmudelit ja üks EfficientNet-B2), mis andis võimaluse skaleerida ennustusi aktiivõppe tarbeks. Esialgne märgistamisansambel saavutas testvalimil F1-skoori 0,9701 ja ROC AUC väärtuse 0,9987 (Tabel~\\ref{tab:initial-ensemble}). TTA (ingl \\textit{Test-Time Augmentation}) põhine keskmistamine silus märgatavalt mudelite piiripealseid erinevusi.

\\begin{table}[h!]
\\centering
\\caption{Esialgse pilootansambli üksikmudelite tulemusnäitajad valideerimisvalimil.}
\\label{tab:initial-ensemble}
\\begin{tabular}{llll}
\\hline
\\textbf{Mudel} & \\textbf{Val AUC} & \\textbf{Val F1} & \\textbf{Otsustuslävi} \\\\
\\hline
CNN-Deep-Attn (seeme 42) & 0,9969 & 0,9580 & 0,87 \\\\
CNN-Deep-Attn (seeme 43) & 0,9972 & 0,9570 & 0,81 \\\\
CNN-Deep-Attn (seeme 44) & 0,9974 & 0,9596 & 0,83 \\\\
EfficientNet-B2           & 0,9963 & 0,9463 & 0,72 \\\\
\\hline
\\textbf{Ansambel (TTA)}   & \\textbf{0,9987} & \\textbf{0,9701} & \\textbf{0,68} \\\\
\\hline
\\end{tabular}
\\end{table}

\\subsubsection{Kõigi paanide hindamine ja aktiivõppe valiku moodustamine}
\\label{koigi-paanide-hindamine}

Loodud pilootansambliga hinnati märgistamise laiendamiseks kogu kättesaadav paanide kogum ($580\\,136$ märgistust). Eesmärk oli selekteerida manuaalseks ülevaatuseks piiripealsed ja raskesti klassifitseeritavad struktuurid, et maksimeerida järgneva lõpliku treeningusamba võimekust.
Madala kindluse vahemikuks määrati $p \\in [0{,}39; 0{,}61]$, kuhu jäi $29\\,628$ paani. Neile lisati 5\\% juhuvalim muudest tulemustest ($26\\,215$ paani), moodustades kokku $55\\,843$ paanist ($10{,}08\\%$ korpusest) koosneva eelisjärjekorra eksperdi hinnanguks.

\\subsubsection{Märgiste kvaliteet ja automaatmärgistuse tase}
\\label{margiste-kvaliteet}

Kaasnevalt analüüsiti mudeli automaatseid valepositiivseid ennustusi. Juhuvalimi peal saavutati automaatmärgistuse võimekuseks F1 $0{,}8603$ (kooskõla inimesega $\\approx 91\\%$). Sihipäraselt valitud keeruliste ja madala mudelikindlusega (ingl \\textit{low-confidence}) vahemiku puhul oli märgistaja ning pilootansambli kooskõla F1 vaid $0{,}4746$. Selline vastandus tõendab aktiivõppe sihistatud lähenemise edukust -- mudelile tekitati rikastatud materjali täpselt piirkondadest, kus esialgne mudel enim eksis.

\\subsubsection{Suurendatud märgistusvalim ja ruumilis-ajaline jaotus}
\\label{mojusa-valimi-jaotus}

Käsitsi ülevaatuse järel ning rangete reeglite kohaldamisel jagati kõik paanid treening-, valideerimis- ja testikomplektideks. Kasutusse võeti rangelt ruumiliselt eraldatud loogika, vähendamaks lekkepotentsiaali kattuvatest paanidest. Kokku eraldati valimisse $142\\,465$ märgistust, millest treeningusse jõudis $67\\,290$ ja testikomplekti turvaliselt isoleerituna $56\\,521$. Selline jagunemine ja filtreerimine hoidis kõrvale andmemürast ja autokorrelatsioonist. Joonisel~\\ref{fig:andmestiku_jaotus} on esitatud piltlik KDE tihedusjaotus, mis demonstreerib visuaalselt treeningu, testi, puhveralade ning spetsiifilise, keskmistatud aktiivõppevalimi jagunemist.

\\begin{figure}[h!]
\\centering
\\includegraphics[width=1\\linewidth]{joonised/andmestiku_jaotus.png}
\\caption{Klassifitseerimise andmestiku tihedusjaotus mudeli ennustatud tõenäosuse lõikes. Graafik näitab mudeli kindlust (x-telg) erinevates andmejaotustes: kogu korpus, treening-, test- ja väljajäetud puhverala struktuur ning spetsiaalselt käsitsi märgistatud andmepunktide jaotus (mis katab ühtlasemalt ka keskse määramatuse vahemikku).}
\\label{fig:andmestiku_jaotus}
\\end{figure}

\\subsubsection{Ansambli agregeerimismeetodite ja kandidaatide võrdlus}
\\label{ansambli-agregeerimise-tulemused}

Pärast aktiivõppega laiendatud märgistusandmestiku moodustamist võrreldi neljanda ansambliliikme ja tõenäosuste agregeerimise alternatiive 5$\\times$5 ristvalideerimise raamistikus. Eesmärk oli valida kindel täiendus kolme CNN-Deep-Attn baasmudelile. Võrdlus näitas, et EfficientNet-B2 lisamine täiendas neid kõige paremini. Meetoditest saavutas kaalutud hääletus (ingl \\textit{weighted vote}) kõrgeima keskmise F1-skoori ($0{,}9866$), olles logistilise regressiooni virnaansambliga samaväärne, ent tõlgendatavam (Tabel~\\ref{tab:ensemble-methods-comparison}).

\\begin{table}[h!]
\\centering
\\caption{Ansambli agregeerimismeetodite võrdlus 5$\\times$5 ristvalideerimisel (25 hindamist). Kõik edasised täiustatud meetodid edestavad lihtsat pehmet hääletust statistiliselt olulisel määral.}
\\label{tab:ensemble-methods-comparison}
\\begin{tabular}{lllcc}
\\hline
\\textbf{Kandidaat} & \\textbf{Meetod} & \\textbf{F1 95\\% UV} & \\textbf{AUC} & \\textbf{$p$ vs lihtne} \\\\
\\hline
EfficientNet-B2 & Pehme hääletus           & [0,9804; 0,9843] & 0,9761 & --- \\\\
EfficientNet-B2 & Pehme h. (häälest.~lävi) & [0,9819; 0,9855] & 0,9761 & 0,0011 ** \\\\
EfficientNet-B2 & \\textbf{Kaalutud hääletus} & \\textbf{[0,9849; 0,9883]} & \\textbf{0,9810} & $\\mathbf{<0{,}0001}$ *** \\\\
EfficientNet-B2 & Virnaansambel (LR)       & [0,9849; 0,9881] & 0,9810 & $<0{,}0001$ *** \\\\
EfficientNet-B2 & Virnaansambel (MLP)      & [0,9837; 0,9873] & 0,9802 & $<0{,}0001$ *** \\\\
\\hline
ConvNeXt-tiny   & Pehme hääletus           & [0,9802; 0,9839] & 0,9754 & --- \\\\
ConvNeXt-tiny   & Virnaansambel (LR)       & [0,9819; 0,9853] & 0,9778 & 0,0002 *** \\\\
ConvNeXt-tiny   & Virnaansambel (MLP)      & [0,9824; 0,9857] & 0,9772 & 0,0014 ** \\\\
\\hline
ConvNeXt-small  & Pehme hääletus           & [0,9788; 0,9825] & 0,9757 & --- \\\\
ConvNeXt-small  & Virnaansambel (LR)       & [0,9842; 0,9872] & 0,9789 & $<0{,}0001$ *** \\\\
ConvNeXt-small  & Kaalutud hääletus        & [0,9839; 0,9870] & 0,9789 & $<0{,}0001$ *** \\\\
\\hline
\\end{tabular}
\\end{table}

\\subsubsection{Lõpliku ansambli jõudlus}
\\label{lopliku-ansambli-tulemused}

Lõplik (neljamudeline) ansambel treeniti 67\\,290 lekkekontrolliga märgistusel. Testikomplektil ($56\\,521$ märgistust) ulatus F1-skoor $0{,}9819$-ni ning ROC AUC väärtus $0{,}9885$-ni. Ansamblipõhine lähenemine aitas eeskätt kaasa raskesti eristatavatel maastikel lokaalse müra tasandamisele.

\\begin{figure}[h!]
\\centering
\\fbox{\\parbox{0.92\\linewidth}{\\centering
\\textbf{Joonis 5.1.} ROC-kõver ja saagise-täpsuse (ingl \\textit{Recall-Precision}, PR) kõver lõpliku ansambli kohta testandmestikul.\\
Placeholder}}
\\caption{Mudeli eristusvõimet demonstreerivad ROC ja PR kõverad.}
\\label{fig:roc-pr-curve}
\\end{figure}

\\subsubsection{CHM variantide võrdlus}
\\label{chm-variantide-klassifitseerimise-tulemused}

Treeningfaasi esialgsetes eksperimentides vaadeldi mitmeid CHM-i arvutusmudeleid (näiteks erineva kõrgusfiltriga rastersisendid ja mitmekanalilised komposiidid). Lõplikus arenduses ja andmete märgistamise töövoos rakendati 20 cm resolutsiooniga 0--1,31 m kõrguseid filtreerivat künnisest kõrgemate eemaldamise meetodit (ingl \\textit{drop-above-threshold}). Ühekanaliline 20 cm raster pakkus kõige efektiivsema tasakaalu andmelihtsuse, arvutusresursside ja tuvastusvõimekuse vahel.

\\subsubsection{Klassifitseerimismudelis esinevate vigade analüüs}
\\label{klassifitseerimise-vigade-analuus}

Vaatamata kõrgele F1-skoorile esines mudeli ennustustes mõningaid iseloomulikke veamustreid, mille analüüsimine pakub olulist sisendit nii metsamajanduse praktikale kui ka edasistele uurimissuundadele.

\\paragraph{Valepositiivsed (FP)} Mudel kaldus aeg-ajalt lamapuiduks klassifitseerima vanu kuivenduskraavide kaldavalle, piklikke kivimoodustisi (näiteks vanad aiavaremed) ning teatud juhtudel masinate rööpaid. Neid struktuure iseloomustab lokaalses kõrgusmudelis sirgjooneline anomaalia, mis 20 cm CHM-pikslite puhul võib sarnaneda tüvele.

\\paragraph{Valenegatiivsed (FN)} Valenegatiivseid ilmnes enamasti vanemates metsasälikutes, kus tüvi oli kattunud paksu samblaga ja vajunud tugevalt huumusesse. Sellised puud ei kerki 1,3 m CHM kihist selgelt esile ning 1--4 pts/m$^2$ LiDAR andmestik ei taga alati piisavat detailsust maapinnalähedase kuju säilitamiseks.

\\begin{figure}[h!]
\\centering
\\fbox{\\parbox{0.92\\linewidth}{\\centering
\\textbf{Joonis 5.2.} Iseloomulike vigade näited: (a) Kändude ja kivide valepositiivne tuvastus, (b) tihedasse alustaimestikku uppunud lamapuu valenegatiivne juhtum.\\
Placeholder}}
\\caption{Mudeli vigade visuaalsed näited.}
\\label{fig:error-analysis}
\\end{figure}
"""

match = re.search(r"\\subsection\{Klassifitseerimise tulemused\}.*?\\end\{figure\}\n", content, re.DOTALL)
if match:
    updated_content = content[:match.start()] + new_content + content[match.end():]
    with open(TEX_FILE, "w", encoding="utf-8") as f:
        f.write(updated_content)
    print("Success: Placed the substituted text effectively.")
else:
    print("Match failed. Check the regular expression against the source file.")

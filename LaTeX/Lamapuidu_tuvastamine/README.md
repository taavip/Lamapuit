# Arvutiteaduse instituudi bakalaureuse- ja magistriõppe lõputööde LaTeX mall

Siin asub arvutiteaduse instituudi bakalaureuse- ja magistriõppe lõputööde LaTeX mall. See mall on mõeldud abistava juhendina, kuidas Teie võite oma lõputööd vormistada. Konkreetsete reeglite jaoks tuleks vaadata [Tartu Ülikooli arvutiteaduse instituudis kaitsvate lõputööde nõuete ja hindamise](https://cs.ut.ee/en/content/thesis-deadlines-and-guidelines) dokumenti.

Käesolev mall on uuendatud kevadel 2025; kõige uuema versiooni saab alati https://gitlab.cs.ut.ee/unitartucs-thesis-templates/bsc-msc-latex-template aadressilt.

## Ülesseadmine
### Overleaf

Laadige käesolev repositooriumi kood alla ZIP-failina (*Code* → *zip*). Seejärel, Overleaf keskkonnas, looge uus project ning valige *Upload Project*. Laadige eelnevalt allalaetud ZIP-fail üles. Seejärel valige juurkaustas olev `thesis.tex` fail ja kompilleerige projekt sealt.

Ärge unustage, et Tartu Ülikooli üliõpilasena on Teil võimalik tasuta kasutada Overleafi preemium-paketti: https://www.overleaf.com/edu/unitartu

### Visual Studio Code

Selleks, et kasutada seda malli **Visual Studio Code** töökeskkonnas, kõigepealt tõmmake ja paigaldage see töökeskkond: https://code.visualstudio.com/

Visual Studio (VS) Code on paljude võimsate laiendustega kergeloomuline töökeskkond. Selleks, et VS Code keskkonnas LaTeX-iga tööd teha, tuleks paigaldada järgnevad töökeskkonna laiendused:
* **LaTeX Workshop** – Põhiline laiendus, mis võimaldab LaTeX-iga tööd teha.
* **Hide Gitignored** – See laiendus peidab ära töökeskkonna failivaaturi paneelist suurearvulised LaTeX-i tööfailid.

Te peate ka LaTeX-i eraldi oma arvutisse paigaldama.

#### Windowsil

Tõmmake alla [Tex Live](https://www.tug.org/texlive/windows.html#:~:text=install%2Dtl%2Dwindows.exe) tarkvara. Paigaldamisel te võite valida *Advanced* seadistused ja muuta *full scheme* (8 GB) valiku *basic scheme* (~400 MB) valiku peale. Olenemata valikust peate te käsitsi paigaldama ka `latexmk` ja `latexmk.windows` paketid.

Kui te valisite *basic scheme* valiku, siis peate te paigaldama ka järgnevad paketid:
```
xcolor, parskip, etoolbox, microtype, kastrup, newtx, xpatch, xkeyval, xstring, fontaxes, tex-gyre, titlesec, caption, wrapfig, collectbox, adjustbox, footmisc, fancyvrb, fvextra, upquote, lineno, csquotes, cachefile, float, fp, latex2pydata, minted.windows, newfloat, pgf, pgfopts, minted, logreq, biblatex, biber.windows, biber, babel-estonian, hyphen-Estonian, euenc, tipa, xunicode, fontspec, lua-ul, tabularray, ninecolors, xurl
```

Kui VS Code töökeskond ütleb, et ta ei leia üles konkreetset paketti, siis kasutage Tex Live tarkvara, et see paigaldada.

Siin on ka üks põhjalik õpetus: https://blog.jakelee.co.uk/getting-latex-working-in-vscode-on-windows/

#### Linuxil

Ubuntu Linuxis on vaja paigaldada vajalikud paketid järgmise käsuga:
```
sudo apt install --no-install-recommends --no-install-suggests texlive-plain-generic texlive-latex-extra texlive-lang-european latexmk texlive-luatex ttf-mscorefonts-installer texlive-bibtex-extra biber python3-pygments
```

## Kasutamine

1. Avage juurkaustas olev `thesis.tex` ja kommenteerige sisse see rida, mis keeles Te oma lõputööd teete.
2. Minge, vastavalt oma keelevalikule, `estonian` või `english` kausta. Sealt leiate faili `põhi.tex` või `main.tex`. Selles failis täidke ära oma lõputööga seonduv info. Seal defineeritakse ka lõputöö dokumendis sisalduvi sektsioone.
3. Kaustas `sektsioonid` või `sections` on Teie lõputöö sektsioonide failid.
4. Kaustas `joonised` või `figures` on Teie lõputöö joonised.
5. Failis `seadistus.tex` või `config.tex` saate seadistada oma lõputöö seadeid. Näiteks valida, millist viitamisstiili kasutada soovite.
6. Failis `viited.bib` või `references.bib` on Teie lõputöö bibliograafiakirjed (soovitatav Zoterost eksportida).

## Mured ja kontaktinfo

Probleemide või küsimuste korral võtke ühendust ati.study@ut.ee

---
University of Tartu Institute of Computer Science - BSc and MSc thesis template from spring 2025.  The latest version can always be found at https://gitlab.cs.ut.ee/unitartucs-thesis-templates/bsc-msc-latex-template.

# The LaTeX Thesis Template for the Bachelor's and Master's Theses at the Institute of Computer Science

Here are the LaTeX thesis templates for to help with writing your bachelor's or master's thesis at the Institute of Computer Science. The template is for guidance on how Your thesis could be formatted. For specific rules, refer to the [Guidelines for preparing and grading of graduation theses at the Institute of Computer Science of the University of Tartu](https://cs.ut.ee/en/content/thesis-deadlines-and-guidelines) document





## Setup

### Overleaf

Download the contents of this repository as a ZIP file (*Code* → *zip*). Then, in Overleaf, create a new project and choose *Upload Project*. Upload the downloaded ZIP file. Then choose the `thesis.tex` file in the root folder and compile from there.

Keep in mind that as a student of the University of Tartu, you have free access to Overleaf Premium : https://www.overleaf.com/edu/unitartu

### Visual Studio Code

To use this template with **Visual Studio Code**, first download and install the IDE: https://code.visualstudio.com/

Visual Studio (VS) Code is a lightweight IDE with very powerful extensions. To work with LaTeX in VS Code, install the following extensions from within the IDE:
* **LaTeX Workshop** – The main extension that allows working with LaTeX.
* **Hide Gitignored** – This will hide all the many working files of LaTeX from your IDE-s Expolorer panel.

You also need to separately install LaTeX itself on your computer.

#### On Windows
Download the [Tex Live](https://www.tug.org/texlive/windows.html#:~:text=install%2Dtl%2Dwindows.exe) software. When installing, you can select *Advaned* and change *full scheme* (8 GB) to *basic scheme* (~400 MB). Regardless of your choice, you have to install the packages `latexmk` and `latexmk.windows` manually.

If you chose the *basic scheme*, you need to also install the following packages:
```
xcolor, parskip, etoolbox, microtype, kastrup, newtx, xpatch, xkeyval, xstring, fontaxes, tex-gyre, titlesec, caption, wrapfig, collectbox, adjustbox, footmisc, fancyvrb, fvextra, upquote, lineno, csquotes, cachefile, float, fp, latex2pydata, minted.windows, newfloat, pgf, pgfopts, minted, logreq, biblatex, biber.windows, biber, babel-estonian, hyphen-Estonian, euenc, tipa, xunicode, fontspec, lua-ul, tabularray, ninecolors, xurl
```
If at any time VS Code tells you that it cannot find a specific package, use the Tex Live software to install it.

There is a comprehensive tutorial also here: https://blog.jakelee.co.uk/getting-latex-working-in-vscode-on-windows/

#### On Linux

On Ubuntu Linux, you need to install the necessary packages with the following command:
```
sudo apt install --no-install-recommends --no-install-suggests texlive-plain-generic texlive-latex-extra texlive-lang-european latexmk texlive-luatex ttf-mscorefonts-installer texlive-bibtex-extra biber python3-pygments
```

## Usage

1. Open the `thesis.tex` from the root folder and comment in Your chosen thesis language. In subsequent instructions, let's assume You chose English.
2. Open the folder `english` and find the file `main.tex`. In that file, fill in the information about Your thesis. That file also defines what sections will be included in your document.
3. The folder `sections` stores the files for Your sections.
4. The folder `figures` stores Your figure files.
5. The `config.tex` file allows You to specify some settings for Your thesis. For example, which reference style You want to use.
6. The `references.bib` file stores Your bibliography entries (ones you can export from Zotero)

## Issues and Contact

In case of problems or questions, contact ati.study@ut.ee

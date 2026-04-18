\ProvidesClass{unitartucs/thesis}[Tartu Ülikooli arvutiteaduse instituudi bakalaureuse ja magistri õppeastmete lõputööde LaTeX mall]

% Üldine vormistus
\LoadClass[12pt]{article}
\usepackage[a4paper,margin=25.4mm]{geometry}

\usepackage{xcolor}
\definecolor{colorURL}{rgb}{0.15, 0.33, 0.09}

% Tüpograafia
\usepackage{parskip} % Taandrea asemel lõiguvahe
\widowpenalty=10000
\clubpenalty=10000
\hyphenpenalty=10000
\hbadness=10000
\sloppy
\usepackage{microtype}

% Keeleline tugi
\AddToHook{package/inputenc/before}{
    \usepackage{fvextra}
    \usepackage{csquotes}
}
\usepackage{babel}

\newcommand{\supervisorname}{Juhendaja}
\newcommand{\supervisorsname}{Juhendajad}
\newcommand{\infoname}{Infoleht}
\newcommand{\keywordsname}{Võtmesõnad}
\newcommand{\infoUT}{Tartu Ülikool}
\newcommand{\infoICS}{Arvutiteaduse instituut}

\addto\captionsestonian{
    \renewcommand{\supervisorname}{Juhendaja}
    \renewcommand{\supervisorsname}{Juhendajad}
    \renewcommand{\infoname}{Infoleht}
    \renewcommand{\abstractname}{Lühikokkuvõte}
    \renewcommand{\keywordsname}{Võtmesõnad}
    \renewcommand{\refname}{Viited}
    \renewcommand{\appendixname}{Lisad}
    \renewcommand{\infoUT}{
        {\fontsize{15}{15}\selectfont T\fontsize{12}{12}\selectfont ARTU}
        {\fontsize{15}{15}\selectfont Ü\fontsize{12}{12}\selectfont LIKOOL}
    }
    \renewcommand{\infoICS}{Arvutiteaduse instituut}
}

\addto\captionsenglish{
    \renewcommand{\supervisorname}{Supervisor}
    \renewcommand{\supervisorsname}{Supervisors}
    \renewcommand{\keywordsname}{Keywords}
    \renewcommand{\infoUT}{
        {\fontsize{15}{15}\selectfont U\fontsize{12}{12}\selectfont NIVERSITY}
        {\fontsize{12}{12}\selectfont OF}
        {\fontsize{15}{15}\selectfont T\fontsize{12}{12}\selectfont ARTU}
    }
    \renewcommand{\infoICS}{Institute of Computer Science}
}

% Fondid
\usepackage{fontspec}
\setmainfont{Times New Roman}
\setsansfont{Arial}
\setmonofont{Courier New}

% Peatükkide pealkirjad
\usepackage{titlesec}
% Heading 1
\titleformat{\section}{\fontsize{16}{16}\selectfont\rmfamily\bfseries\upshape}{\thesection.}{6pt}{}
\titlespacing{\section}{0pt}{0pt}{5pt}
% Heading 2
\titleformat{\subsection}{\fontsize{15}{15}\selectfont\rmfamily\bfseries\upshape}{\thesubsection}{6pt}{}
\titlespacing{\subsection}{0pt}{12pt}{4pt}
% Heading 3
\titleformat{\subsubsection}{\fontsize{14}{14}\selectfont\rmfamily\bfseries\upshape}{\thesubsubsection}{6pt}{}
\titlespacing{\subsubsection}{0pt}{8pt}{3pt}
% Heading 4 (mitte-standardne)
\titleclass\subsubsubsection{straight}[\subsubsection]
\titleformat{\subsubsubsection}{\fontsize{14}{14}\selectfont\rmfamily\bfseries\itshape}{\thesubsubsubsection}{6pt}{}
\titlespacing{\subsubsubsection}{0pt}{4pt}{2pt}
\newcounter{subsubsubsection}
\renewcommand{\thesubsubsubsection}{\thesubsubsection.\arabic{subsubsubsection}}
\renewcommand{\theparagraph}{\thesubsubsubsection.\arabic{paragraph}}
% Heading 5
\titleformat{\paragraph}{\fontsize{14}{14}\selectfont\rmfamily\mdseries\itshape}{\theparagraph}{6pt}{}
\titlespacing{\paragraph}{0pt}{0pt}{0pt}
% Heading 6
\definecolor{subparagraphgray}{rgb}{0.35, 0.35, 0.35}
\titleformat{\subparagraph}{\fontsize{14}{14}\selectfont\rmfamily\mdseries\itshape\color{subparagraphgray}}{\thesubparagraph}{6pt}{}
\titlespacing{\subparagraph}{0pt}{0pt}{0pt}

\setcounter{secnumdepth}{3} % Nummerdatud kuni \subsubsection
\newcommand{\sectionbreak}{\clearpage\phantomsection} % Esimese taseme peatükid algavad uuelt lehelt

% Sisukord
\usepackage{titletoc}
\titlecontents{section}[0pt]{}{\thecontentslabel.\enspace\normalsize}{\normalsize}{\titlerule*[.3pc]{.}\contentspage}
\titlecontents{subsection}[16pt]{}{\thecontentslabel\enspace\normalsize}{\normalsize}{\titlerule*[.3pc]{.}\contentspage}
\titlecontents{subsubsection}[32pt]{}{\thecontentslabel\enspace\normalsize}{\normalsize}{\titlerule*[.3pc]{.}\contentspage}
\titlecontents{subsubsubsection}[48pt]{}{\thecontentslabel\enspace\normalsize}{\normalsize}{\titlerule*[.3pc]{.}\contentspage}
\titlecontents{paragraph}[64pt]{}{\thecontentslabel\enspace\normalsize}{\normalsize}{\titlerule*[.3pc]{.}\contentspage}
\titlecontents{subparagraph}[80pt]{}{\thecontentslabel\enspace\normalsize}{\normalsize}{\titlerule*[.3pc]{.}\contentspage}

\setcounter{tocdepth}{3} % Sisukorras kuni \subsubsection

% Joonised
\RequirePackage[labelsep=period]{caption} % Punkt pealdiste sildi lõpus
\captionsetup{justification=centering} % Mitmerealise sildi viimane rida keskel mitte vasakul.

\usepackage{wrapfig}
\usepackage[export]{adjustbox}

% Mähitud joonise vasak ruum
\BeforeBeginEnvironment{wrapfigure}{\setlength{\columnsep}{30pt}}

% Mähitud joonise ülemine ruum
\patchcmd\WF@putfigmaybe{\lower\intextsep}{}{}{\fail}%
\AddToHook{env/wrapfigure/begin}{\setlength{\intextsep}{0pt}}

% Joonealused märkused
\addtolength{\skip\footins}{40pt plus 5pt}
\renewcommand{\footnoterule}{%
  \kern 0pt
  \hrule width 145pt height 0.01pt
  \kern 12pt
}
\usepackage[marginal]{footmisc}
\setlength{\footnotemargin}{4pt}

% Joonised
\usepackage{graphicx}

% Tabelid
\usepackage{colortbl}
\usepackage{tabularray}

% Koodinäited
\usepackage{minted}
\setminted{
    frame=single,
    framesep=10pt,
    fontsize=\footnotesize,
}

% Matemaatilised valemid
\usepackage{amsmath}

% Meta-andmete defineerimine
\newcommand\supervisor[1]{\renewcommand\@supervisor{#1}}
\newcommand\@supervisor{\@latex@error{No \noexpand\supervisor given}\@ehc}

\newcommand\curriculum[1]{\renewcommand\@curriculum{#1}}
\newcommand\@curriculum{\@latex@error{No \noexpand\curriculum given}\@ehc}

\newcommand\thesis[1]{\renewcommand\@thesis{#1}}
\newcommand\@thesis{\@latex@error{No \noexpand\thesis given}\@ehc}

\newcommand\keywords[1]{\renewcommand\@keywords{#1}}
\newcommand\@keywords{\@latex@error{No \noexpand\keywords given}\@ehc}

% Tiitelleht
\newcounter{@supervisori}
\newcounter{@supervisors}

\renewcommand*{\maketitle}{
    \setbox0=\vbox{\setcounter{@supervisors}{1}\newcommand*{\degree}[1]{}\renewcommand*{\and}{\stepcounter{@supervisors}}\@supervisor}%
    \thispagestyle{empty}

    \vspace*{-9.1 mm}

    \begin{center}

        % Päis
        {
            \Large{\infoUT} \\ % Some environments do not support small-caps for Times New Roman
            \fontsize{14}{16}\selectfont
            \infoICS\\
            \@curriculum\\
        }

        \vspace{63mm}

        % Pealkiri
        {
            {\fontsize{16}{16}\selectfont \textbf{\@author}} \\
            \vspace{2mm}
            {\fontsize{21}{24}\selectfont \textbf{\textsf{\@title}}} \\
            \vspace{3mm}
            {\fontsize{14}{16}\selectfont \textbf{\@thesis}}
        }

    \end{center}

    \vspace{17mm}

    % Juhendajad
    {
        \begin{flushright}
            {
                \large

                \renewcommand*{\and}{\\ }
                \newcommand*{\degree}[1]{, ##1}

                \ifnum\value{@supervisors}>1\supervisorsname\else\supervisorname\fi: \\ \@supervisor

            }
        \end{flushright}
    }

    \vfill

    % Jalus
    \centerline{\large Tartu {\@date}}
}

% Infoleht
\newenvironment{@info}[1]{
    \noindent\textbf{\large #1}
    \vspace*{1ex} % TODO: why 3ex only in one language in old template?

    \renewenvironment{abstract}{
        \noindent\textbf{\abstractname:}

        \noindent\ignorespaces % TODO: why need \ignorespaces?
    }{\vspace*{1ex}}

    \renewcommand*{\keywords}[1]{
        \noindent\textbf{\keywordsname:} \ifstrempty{##1}{\@keywords}{##1}

        \vspace*{1ex}
    }

    \newcommand*{\cercs}[1]{
        \noindent\textbf{CERCS:} ##1

        \vspace*{1ex}
    }
}{\filbreak\vspace{4ex}}

\newenvironment{info}{\begin{@info}{\@title}}{\end{@info}}
\newenvironment{otherInfo}[2]{\begin{otherlanguage}{#1}\begin{@info}{#2}}{\end{@info}\end{otherlanguage}}


% Lingid ja PDF-i meta-andmed
\AtEndPreamble{

    \usepackage[breaklinks]{hyperref}
    \hypersetup{
        colorlinks,
        allcolors=black,
        urlcolor=colorURL, % \url värv
        breaklinks
    }
    \urlstyle{same} % Lingid sama fondiga nagu tekst

    \usepackage[minimal]{lua-ul}
    \usepackage{calc}
    \newlength{\underLineDepth}
    \setlength{\underLineDepth}{1pt+0.25ex}
    \newunderlinetype\@underLine{%
        \leaders\vrule height -0.25ex depth \underLineDepth % Close underline like in Word
    }
    \usepackage{xurl} % Allow arbitrary line breaks in urls
    \NewDocumentCommand\underLine{+m}{{\@underLine#1}}
    \DeclareUrlCommand{\Hurl}{% https://tex.stackexchange.com/a/169734
        \def\UrlLeft##1\UrlRight{\@underLine##1} % Use \@underLine directly to allow line breaks in urls
    }

    % hyperref-i adapteerimine lisatud subsubsubsection-i jaoks
    \newcommand*{\toclevel@subsubsubsection}{4}
    \renewcommand*{\toclevel@paragraph}{5}
    \renewcommand*{\toclevel@subparagraph}{6}
    \hypersetup{ % PDF-i meta-andmed
        pdftitle={\@title},
        pdfsubject={\@title},
        pdfauthor={\@author},
        pdfkeywords={\@keywords},
    }

    \usepackage[numbered,open]{bookmark} % PDF-i sisukord

    \DeclareLanguageMapping{estonian}{unitartucs/estonian}
    \DeclareLanguageMapping{english}{unitartucs/english}

    \usepackage{csquotes}
}

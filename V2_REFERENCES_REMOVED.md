# V2 Model Search References - Removed from Thesis

## Date Removed
2026-05-04

## Removed from: 5-tulemused.tex

### Paragraph Removed (was line 21)
```
Mudeli otsingus V2 (35,832 tiili, 111 rasterikiht) ilmnes ConvNeXt perekonna 
üleolek üksikmudelitasandil: TTA-ga ConvNeXt-tiny saavutas ristvalideerimise 
F1-skoori 0,9233, ületades selgelt kõiki CNN-Deep-Attn variante. Tabel~\ref{tab:model-search-v2} 
võrdleb V2 esimese etapi tipparhitektuure. Kuna V2 teine etapp jäi pooleli 
(8/36 eksperimenti) ning ConvNeXt-põhist ansamblit ei testitud, jäi lõplikuks 
valikuks märtsis 2026 kinnitatud 4-mudeline ansambel, mis lõplikus ümberõppes 
(67,290 treeningtiili, ruumiline jagamine E07) saavutas F1-skoori 0,9819 ja ROC 
AUC väärtuse 0,9885 -- ületades esialgset märgistamisansamblit 1,2 protsendipunkti võrra.
```

### Table Removed (was lines 23-38)
```
\begin{table}[h!]
\centering
\caption{Mudeli otsingu V2 esimese etapi tipptulemused ristvalideerimise F1-skoori järgi (35,832 tiili, 111 rasterikiht).}
\label{tab:model-search-v2}
\begin{tabular}{lllll}
\hline
\textbf{Arhitektuur} & \textbf{Kaofunktsioon} & \textbf{Reg.} & \textbf{CV AUC} & \textbf{CV F1} \\
\hline
ConvNeXt-tiny  & CE     & MixUp     & 0,9868 & 0,9180 \\
ConvNeXt-small & CE     & MixUp     & 0,9887 & 0,9178 \\
EfficientNet-B2 & CE    & MixUp     & 0,9898 & 0,9162 \\
CNN-Deep-Attn (dropout) & CE & MixUp & 0,9901 & 0,9136 \\
CNN-Deep-Attn  & CE     & MixUp     & 0,9901 & 0,9126 \\
\hline
\end{tabular}
\end{table}
```

## References to Clean Up

### 1. Still in Methodology Chapter (4-metoodika.tex, line 93)
- **Label:** `\label{tab:ensemble-comparison-v2}`
- **Status:** NEEDS RENAMING
- **Reason:** This label refers to ensemble voting method comparison (V2), NOT model search V2, so it can stay
- **Action:** Consider renaming to `tab:ensemble-methods-comparison` for clarity

### 2. No longer referenced in Results Chapter
- `\ref{tab:model-search-v2}` — REMOVED ✓

## Deprecated Table Label
- `tab:model-search-v2` — no longer used anywhere

## Notes
- V1 model search section (line 14 of results) is retained as it shows methodology validity
- Ensemble voting comparison results (line 19) are retained as they justify final model choice
- Final 4-model ensemble results are now presented directly without reference to incomplete V2 experiments

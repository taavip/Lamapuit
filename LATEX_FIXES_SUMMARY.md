# LaTeX Compilation Fixes Summary

**Date**: 2026-04-28  
**Files Modified**: 3

---

## ✅ Fixes Applied

### 1. Ampersand Escaping (Text Mode)

**Problem**: Ampersand `&` used in regular text instead of alignment context

| File | Line | Original | Fixed | Issue |
|---|---|---|---|---|
| `4-metoodika.tex` | 329 | `Kuncheva & Whitaker` | `Kuncheva \& Whitaker` | Bare ampersand in author citation |
| `5-tulemused.tex` | 252 | `Waldmann & Sliwa` | `Waldmann \& Sliwa` | Bare ampersand in author citation |

**Root Cause**: LaTeX treats `&` as a special alignment character. In text mode, it must be escaped as `\&`.

---

### 2. Table Column Alignment

**Problem**: Extra alignment tabs — table header definition didn't match row cell count

| File | Line | Original | Fixed | Issue |
|---|---|---|---|---|
| `5-tulemused.tex` | 113 | `\begin{tabular}{lcc}` | `\begin{tabular}{lccc}` | Header defines 3 cols, rows have 4 |
| `5-tulemused.tex` | 142 | `\begin{tabular}{lcc}` | `\begin{tabular}{lccc}` | Header defines 3 cols, rows have 4 |

**Root Cause**: Both tables had 4 data columns but were declared with only 3 column specifiers (`lcc`).

**Fix**: Changed to `lccc` (4 columns: left, centered, centered, centered).

---

### 3. License File Macro References

**Problem**: Undefined control sequences and paragraph-ending errors in `11-litsents.tex`

| Line | Original | Fixed | Issue |
|---|---|---|---|
| 21 | `\author` | `\AuthorName` | Bare command instead of value retrieval |
| 25 | `\date` | `\ThesisDate` | Bare command instead of value retrieval |

**Root Cause**: The document preamble (`seadistus.tex`) defines helper macros to retrieve metadata:
- `\AuthorName` → retrieves `\@author` (set via `\author{...}` in main file)
- `\ThesisDate` → retrieves `\@date` (set via `\date{...}` in main file)

The license file was trying to use bare `\author` and `\date` commands, which don't output values in this context.

---

### 4. Superscript Character Encoding

**Problem**: Unicode superscript characters (⁻⁴, ⁻⁵) not available in Times New Roman font

| File | Line | Original | Fixed | Issue |
|---|---|---|---|---|
| `4-metoodika.tex` | 165 | `5×10⁻⁴` and `5×10⁻⁵` | `$5 \times 10^{-4}$` and `$5 \times 10^{-5}$` | Unicode superscripts not in font |
| `4-metoodika.tex` | 350 | `5 × 10⁻⁴` | `$5 \times 10^{-4}$` | Unicode superscripts not in font |
| `4-metoodika.tex` | 351 | `5 × 10⁻⁵` | `$5 \times 10^{-5}$` | Unicode superscripts not in font |
| `5-tulemused.tex` | 468 | `5×10⁻⁴` and `1×10⁻³` | `$5 \times 10^{-4}$` and `$1 \times 10^{-3}$` | Unicode superscripts not in font |

**Root Cause**: Direct Unicode superscript characters (U+207B, U+2074, U+2075, etc.) are not part of standard Times New Roman font coverage. LaTeX math mode provides proper rendering.

**Fix**: Converted to LaTeX math mode:
- `×` → `\times` (math mode times symbol)
- `⁻⁴` → `^{-4}` (math mode exponent)
- `⁻⁵` → `^{-5}` (math mode exponent)
- `⁻³` → `^{-3}` (math mode exponent)

---

## 📋 Remaining Issues (NOT FIXED PER USER REQUEST)

### Bibliography Citation-Key Mismatches

**Status**: 16 citations don't match entries in `viited.bib`

**Detailed mapping**: See [`BIBLIOGRAPHY_MAPPING.md`](BIBLIOGRAPHY_MAPPING.md)

**Examples**:
- Citation uses `joyce2019` → actual entry is `joyce_detection_2019`
- Citation uses `marchi2018` → actual entry is `marchi_airborne_2018`
- Citation uses `karm_orienteerumiskaardi_2016` → actual entry is `karm_orienteerumiskaardi_2015` (year mismatch!)

**Root Cause**: Bibliography entries in `viited.bib` use Zotero auto-generated keys with full-title prefixes, but LaTeX citations use simple short keys.

**User Decision**: Not changing viited.bib entries per request. Mapping table provided for reference.

---

## ⚠️ Warnings (Already Present in Earlier Logs)

### 1. Missing Bibliography Entries
- `graves_strategic_2012` — not in viited.bib, must be added
- `tamara_munzner_keynote_2012` — not in viited.bib, must be added

### 2. Font-Related Warnings
- `Command \showhyphens has changed` — LuaLaTeX v1.21 vs older version incompatibility (non-critical)
- `!h` float specifier changed to `!ht` — LaTeX preventing floating objects from being purely at-here placement (acceptable)

---

## 📊 Compilation Impact

| Error Type | Count | Severity | Status |
|---|---|---|---|
| Misplaced alignment tabs | 2 | 🔴 FATAL | ✅ **FIXED** |
| Extra alignment tabs | 6 | 🔴 FATAL | ✅ **FIXED** |
| Missing characters (superscripts) | 8 | 🟠 WARNING | ✅ **FIXED** |
| Undefined citations | 16 | 🔴 FATAL | ⏳ Pending mapping |
| Undefined control sequences | 2 | 🟠 WARNING | ✅ **FIXED** |
| Paragraph ended before \author | 1 | 🟠 WARNING | ✅ **FIXED** |

---

## 🔍 How to Verify Fixes

Run LaTeX compilation to confirm:

```bash
cd LaTeX/Lamapuidu_tuvastamine/estonian/
lualatex -interaction=nonstopmode thesis.tex
# or use your preferred LaTeX build tool
```

Expected outcomes:
- ✅ No "Misplaced alignment tab" errors
- ✅ No "Extra alignment tab" errors  
- ✅ No "Missing character" warnings for superscripts
- ✅ No "Undefined control sequence" for `\AuthorName` and `\ThesisDate`
- ⏳ Still may see undefined citation warnings (expected until citation keys are updated)

---

## 📝 Notes

- All fixes are **non-destructive** and **reversible** via git history
- No changes made to bibliography parameters per user request
- Superscript fix improves both font rendering and LaTeX best practices
- License file macro fix resolves dependency on proper helper macro definitions

# Bibliography Citation-to-Entry Mapping Report

**Generated**: 2026-04-28  
**Purpose**: Document mismatch between citations in LaTeX files and bibliography entry keys in `viited.bib`

---

## Critical Issues Summary

| Issue Type | Count | Severity |
|---|---|---|
| Citation key mismatches (entry exists with different name) | 14 | 🔴 HIGH — causes "undefined citation" errors |
| Completely missing entries | 2 | 🔴 HIGH — must be added to viited.bib |
| Year mismatches | 1 | 🟠 MEDIUM — citation key exists but year is wrong |

---

## Detailed Mapping Table

| Cited in Tex | Entry Status | Correct Key in viited.bib | Issue Type | Notes |
|---|---|---|---|---|
| `dietenberger2025` | ⚠️ Exists elsewhere | `dietenberger_accurate_2025` | Key mismatch | Zotero auto-key includes full title prefix |
| `graves_strategic_2012` | ❌ **MISSING** | N/A | Not in bib | Strategic visualization reference — needs to be added |
| `heinaro2021` | ⚠️ Exists elsewhere | `heinaro_airborne_2021` | Key mismatch | Entry exists with Zotero auto-key format |
| `hell2022` | ⚠️ Exists elsewhere | `hell_classification_2022` | Key mismatch | Entry exists with Zotero auto-key format |
| `joyce2019` | ⚠️ Exists elsewhere | `joyce_detection_2019` | Key mismatch | Entry exists with Zotero auto-key format |
| `kaminska2018` | ⚠️ Exists elsewhere | `kaminska_species-related_2018` | Key mismatch | Entry exists with hyphenated Zotero format |
| `karm_orienteerumiskaardi_2016` | ⚠️ Year wrong | `karm_orienteerumiskaardi_2015` | Year mismatch | **CRITICAL**: Entry key says 2015, but cite expects 2016 |
| `krisanski2021` | ⚠️ Exists elsewhere | `krisanski_sensor_2021` | Key mismatch | Entry exists with Zotero auto-key format |
| `lopesqueiroz2020` | ⚠️ Exists elsewhere | `lopes_queiroz_estimating_2020` | Key mismatch | Entry exists with Zotero auto-key format |
| `maaamet_als` | ⚠️ Exists elsewhere | `maa-ja_ruumiamet_als_nodate` | Key mismatch | Agency name differs in key |
| `marchi2018` | ⚠️ Exists elsewhere | `marchi_airborne_2018` | Key mismatch | Entry exists with Zotero auto-key format |
| `pesonen2008` | ⚠️ Exists elsewhere | `pesonen_airborne_2008` | Key mismatch | Entry exists with Zotero auto-key format |
| `ruumiamet_als_nodate` | ⚠️ Exists elsewhere | `maa-ja_ruumiamet_als_nodate` | Key mismatch | Agency name differs in key |
| `shin2024` | ⚠️ Exists elsewhere | `shin_morphological_2024` | Key mismatch | Entry exists with Zotero auto-key format |
| `tamara_munzner_keynote_2012` | ❌ **MISSING** | N/A | Not in bib | Keynote talk reference — needs to be added |
| `virro2025` | ⚠️ Exists elsewhere | `virro_detection_2025` | Key mismatch | Entry exists with Zotero auto-key format |
| `virro_2025` | ⚠️ Exists elsewhere | `virro_detection_2025` | Key mismatch | Duplicate citation (underscore variant) |

---

## Root Cause Analysis

**Citation Key Naming Mismatch**: 
- **Cited format**: `author[year]` or `author_year` (simple, short form)
- **viited.bib format**: `author_full-title-prefix[year]` (Zotero auto-generated keys with underscores)

**Example**:
- Citation: `\parencite{joyce2019}`
- BibTeX key in file: `@article{joyce_detection_2019,`
- Result: LaTeX cannot find `joyce2019` because the actual key is `joyce_detection_2019`

---

## Solution Approaches

### Option A: Update Citation Keys in viited.bib (NOT RECOMMENDED)
- Manually rename all 14+ entry keys to match citation format
- **Pros**: Clean consistency
- **Cons**: Loses Zotero auto-key pattern; hard to re-sync with Zotero; risky for large bibliography

### Option B: Update All Citations in .tex Files (RECOMMENDED)
- Find-and-replace in all `.tex` files to use correct Zotero keys
- Preserves bibliographic data integrity
- **Implementation**: Search each `.tex` file for the wrong key and replace with correct key
- **Files affected**: 
  - `sektsioonid/1-sissejuhatus.tex`
  - `sektsioonid/2-seotud-tood.tex`
  - `sektsioonid/3-andmed.tex`

### Option C: Add Citation Aliases to viited.bib
- Use BibTeX `@preamble` or custom aliases (not standard in all engines)
- **Not recommended** for simplicity

---

## Entries Completely Missing from viited.bib

These two entries are referenced in `.tex` files but **do not exist** in `viited.bib`:

1. **`graves_strategic_2012`** — Strategic visualization paper
   - Status: Referenced but not in bibliography
   - Action: Must add entry manually or locate in Zotero

2. **`tamara_munzner_keynote_2012`** — Tamara Munzner keynote talk
   - Status: Referenced but not in bibliography
   - Action: Must add entry manually or locate in Zotero

---

## LaTeX Parameter Issues (NOT TO CHANGE)

As per user request, **no parameters in viited.bib entries have been changed**. The issue is purely citation-key mismatch, not bibliography data format problems.

All entries maintain their original Zotero-normalized format:
- File paths: `files/###/filename.pdf` (relative references, not Windows paths)
- Keywords: Alphabetically sorted
- Publisher fields: Properly normalized

---

## Year Consistency Check

| Entry | Year in Key | Year in `date` field | Match? |
|---|---|---|---|
| `karm_orienteerumiskaardi_2015` | 2015 | TBD (check entry) | ⚠️ Key says 2015, but citation expects 2016 |

**Action**: Verify if this entry's date field is 2016. If yes, the **key should be renamed** `karm_orienteerumiskaardi_2016`. If no, the **citation should be changed** to `karm_orienteerumiskaardi_2015`.

---

## Next Steps

1. ✅ **Fixed**: License file now uses `\AuthorName` and `\ThesisDate` macros
2. ✅ **Fixed**: Ampersand escaping in text (4-metoodika.tex, 5-tulemused.tex)
3. ✅ **Fixed**: Table column alignment (5-tulemused.tex)
4. ⏳ **Pending**: Update citation keys in `.tex` files to match viited.bib entries
5. ⏳ **Pending**: Add two missing entries (`graves_strategic_2012`, `tamara_munzner_keynote_2012`)

---

## File References

- **Bibliography file**: `LaTeX/Lamapuidu_tuvastamine/estonian/viited.bib`
- **Citation configuration**: `LaTeX/Lamapuidu_tuvastamine/estonian/seadistus.tex` (line 29)
- **Affected tex files**:
  - `sektsioonid/1-sissejuhatus.tex`
  - `sektsioonid/2-seotud-tood.tex`
  - `sektsioonid/3-andmed.tex`

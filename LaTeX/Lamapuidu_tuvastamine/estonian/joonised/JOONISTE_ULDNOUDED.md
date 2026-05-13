# Üldnõuded joonistele

Need nõuded hoiavad kõik lõputöö joonised visuaalselt ühtsed ja trükivalmis.

1. Formaadi suhe peab olema 16:9 (`figsize=(16, 9)`).
2. Resolutsioon peab olema vähemalt 300 DPI (`dpi=300`).
3. Kasuta ühtset stiili: `seaborn-v0_8-paper`.
4. Voolujoonised ja graafikud peavad kasutama samu põhivärve:
   - sinine `#1f77b4`
   - punane `#d62728`
   - roheline `#2ca02c`
   - hall `#7f7f7f`
   - must `#000000`
5. Telgede nimetused peavad olema eesti keeles, paksus kirjas (`fontweight='bold'`) ja loetavad.
6. Telgede vahemikud peavad olema selgelt defineeritud (nt tõenäosuse ja PR-graafikute korral `0..1`).
7. Ruudustik peab olema hele ja mitte-domineeriv (`linestyle=':'`, `alpha≈0.6`).
8. Legend peab olema arusaadav, eesti terminoloogiaga, ning asuma nii, et see ei varja andmeid.
9. Joonise pealkiri olgu lühike ja sisuline; teadustöö põhitekst selgitab detailid.
10. Salvesta joonised PNG formaadis LaTeX kausta `LaTeX/Lamapuidu_tuvastamine/estonian/joonised/`.
11. Failinimed olgu kirjeldavad ja ühes stiilis (väiketähed, alakriipsud, ilma tühikuteta).
12. Kui joonisel on läved/lõikepunktid (threshold), märgi need järjekindlalt sama tüüpi katkendjoone või punktmarkeri abil.

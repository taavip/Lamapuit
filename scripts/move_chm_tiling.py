import re

ANDMED_FILE = "LaTeX/Lamapuidu_tuvastamine/estonian/sektsioonid/3-andmed.tex"
METOODIKA_FILE = "LaTeX/Lamapuidu_tuvastamine/estonian/sektsioonid/4-metoodika.tex"

with open(ANDMED_FILE, "r", encoding="utf-8") as f:
    andmed = f.read()

# Tuvastan ploki andmete failis
andmed_pattern = re.compile(
    r"(Lamapuidu märgistamise jaoks tekitati madal taimkattemudel.*?)(Mudelite sildistamiseks.*?|)?"
    r"(Sildistamise ja hilisema mudeli sisendi jaoks jagati suuremõõtmelised kaardilehed.*?\\end\{itemize\})",
    re.DOTALL
)

match = andmed_pattern.search(andmed)
if match:
    chm_tiling_text = match.group(0)
    
    # Asendan andmetes
    new_andmed = andmed.replace(
        chm_tiling_text,
        r"Mudelite sildistamiseks ja masinõppe sisendiks vajalik madala taimkatte kõrgusmudeli (CHM) loomise protsess ja selle tükeldamine paanideks on lahti kirjeldatud metoodika peatükis (vt alapeatükki \ref{subsec:workflow-base-chm})."
    )
    with open(ANDMED_FILE, "w", encoding="utf-8") as f:
        f.write(new_andmed)
    print("Updated 3-andmed.tex")

    # Uuendan metoodikas
    with open(METOODIKA_FILE, "r", encoding="utf-8") as f:
        metoodika = f.read()
    
    metoodika_pattern = re.compile(
        r"\\paragraph\{Baas-CHM-i genereerimine\} Esmase madala taimkatte kõrgusmudeli \(CHM\) arvutamine 20~cm.*?mudeli sisendvariantide väljatöötamisele\."
    )
    
    # CHM tekst peaks nüüd metoodikas olema
    replacement_text = "\\paragraph{Baas-CHM-i genereerimine}\n" + chm_tiling_text
    
    new_metoodika = metoodika_pattern.sub(replacement_text, metoodika)
    with open(METOODIKA_FILE, "w", encoding="utf-8") as f:
        f.write(new_metoodika)
    print("Updated 4-metoodika.tex")
else:
    print("Could not find the block in 3-andmed.tex")


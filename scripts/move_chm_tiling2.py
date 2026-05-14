import re

METOODIKA_FILE = "LaTeX/Lamapuidu_tuvastamine/estonian/sektsioonid/4-metoodika.tex"
with open(METOODIKA_FILE, "r", encoding="utf-8") as f:
    metoodika = f.read()

# I already modified 3-andmed.tex successfully (let's check).
# Actually I need the `chm_tiling_text` again.
ANDMED_FILE_BACKUP = "LaTeX/Lamapuidu_tuvastamine/estonian/sektsioonid/4-metoodika.tex" # wait, I don't have it anymore!


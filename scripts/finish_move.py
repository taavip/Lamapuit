import re
import sys

METOODIKA_FILE = "LaTeX/Lamapuidu_tuvastamine/estonian/sektsioonid/4-metoodika.tex"
with open(METOODIKA_FILE, "r", encoding="utf-8") as f:
    metoodika = f.read()

with open("removed_text.txt", "r", encoding="utf-8") as f:
    removed_text = f.read()

# Filter out standard git diff headers just in case
lines = removed_text.split('\n')
clean_lines = [l for l in lines if not l.startswith('---') and not l.startswith('+++')]
chm_tiling_text = '\n'.join(clean_lines).strip()

# Now find the line to replace in metoodika and replace it with string replace, not regex sub
metoodika_pattern = re.compile(
    r"\\paragraph\{Baas-CHM-i genereerimine\} Esmase madala taimkatte kõrgusmudeli \(CHM\) arvutamine 20~cm.*?mudeli sisendvariantide väljatöötamisele\."
)

match = metoodika_pattern.search(metoodika)
if match:
    original_str = match.group(0)
    replacement = "\\paragraph{Baas-CHM-i genereerimine}\n\n" + chm_tiling_text
    new_metoodika = metoodika.replace(original_str, replacement)
    with open(METOODIKA_FILE, "w", encoding="utf-8") as f:
        f.write(new_metoodika)
    print("Successfully updated 4-metoodika.tex with standard string replacement.")
else:
    print("Could not find the target text in 4-metoodika.tex!")

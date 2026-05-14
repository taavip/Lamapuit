import re

TEX_FILE = "LaTeX/Lamapuidu_tuvastamine/estonian/sektsioonid/5-tulemused.tex"

with open(TEX_FILE, "r", encoding="utf-8") as f:
    content = f.read()

# Eemaldan esimese duplikaadi (kuni Ansambli lähenemiseni), ja asendan
pattern = re.compile(
    r"\\subsubsection\{CHM variantide võrdlus\}.*?\\end\{figure\}\n+(Ansambli lähenemine aitas eeskätt vähendada valepositiivseid ennustusi raskesti \n?eristatavatel maastikel[^.]+\. Nelja mudeli \(3\n?\\times\$ CNN, 1\\times\$ EfficientNet\) keskmistatud ennustus silus tõhusalt poten\n?tsiaalsed anomaaliad\.)\n+\\subsubsection\{CHM variantide võrdlus klassifitseerimises\}",
    re.DOTALL
)

def replacer(match):
    ansambli_text = match.group(1).replace('\n', ' ')
    # Tahame selle ansambli teksti "Lõpliku ansambli jõudlus" lõppu panna.
    # Aga see eeldaks tervikteksti asendust mujal. Teeme lihtsamalt.
    return f"{ansambli_text}\n\n\\subsubsection{{CHM variantide võrdlus klassifitseerimises}}"

modified, count = pattern.subn(replacer, content)

# Et "Ansambli lähenemine" asi lahendada, teen veel ühe sub() 
if count > 0:
    with open(TEX_FILE, "w", encoding="utf-8") as f:
        f.write(modified)
    print("Fixed duplicates in 5-tulemused.tex")
else:
    print("No duplicates found with the regex pattern. Need manual verification.")


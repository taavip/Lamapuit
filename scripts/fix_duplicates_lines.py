import os

TEX_FILE = "LaTeX/Lamapuidu_tuvastamine/estonian/sektsioonid/5-tulemused.tex"

with open(TEX_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    if line.startswith(r"\subsubsection{CHM variantide võrdlus}") and r"klassifitseerimises" not in line:
        skip = True
    
    if skip and line.startswith(r"\subsubsection{CHM variantide võrdlus klassifitseerimises}"):
        skip = False

    if not skip:
        new_lines.append(line)

with open(TEX_FILE, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
print("Deleted the duplicated block from 5-tulemused.tex.")

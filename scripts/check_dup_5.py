import re
with open("LaTeX/Lamapuidu_tuvastamine/estonian/sektsioonid/5-tulemused.tex", "r") as f:
    content = f.read()

paras = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 30 and "\\begin{figure}" not in p]
seen = {}
for p in paras:
    if p in seen:
        print(f"DUPLICATE FOUND:\n{p}\n" + "-"*50)
    seen[p] = True

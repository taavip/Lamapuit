import glob
import os

folder = "LaTeX/Lamapuidu_tuvastamine/estonian/sektsioonid/"
tex_files = glob.glob(os.path.join(folder, "*.tex"))

for file_path in tex_files:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    new_content = content.replace("ruumilis-ajaline", "ajalis-ruumiline")
    new_content = new_content.replace("ruumilis-ajalist", "ajalis-ruumilist")
    new_content = new_content.replace("ruumilis-ajalise", "ajalis-ruumilise")
    new_content = new_content.replace("ruumilis-ajaliselt", "ajalis-ruumiliselt")

    if new_content != content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Updated replacements in {file_path}")

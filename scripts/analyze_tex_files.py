import os
import re

SECTIONS_DIR = "LaTeX/Lamapuidu_tuvastamine/estonian/sektsioonid"

def get_tex_files():
    files = [f for f in os.listdir(SECTIONS_DIR) if f.endswith(".tex")]
    files.sort(key=lambda x: int(re.search(r'^\d+', x).group()) if re.search(r'^\d+', x) else 999)
    return files

def analyze_files():
    for f in get_tex_files():
        path = os.path.join(SECTIONS_DIR, f)
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
            
        print(f"\n--- {f} ---")
        
        # Check for empty sections
        empty_sections = re.findall(r"\\(?:sub)*section\{[^}]+\}\s*(?=\\(?:sub)*section|$)", content)
        if empty_sections:
            print("Potentially empty sections:")
            for s in empty_sections:
                print(f"  - {s.strip()}")
                
        # Simple duplicate paragraph detection
        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 50]
        seen = set()
        duplicates = set()
        for p in paragraphs:
            # removing latex commands for cleaner comparison
            clean_p = re.sub(r'\\[a-zA-Z]+(?:\[[^\]]+\])?\{[^}]+\}', '', p)
            if clean_p in seen:
                duplicates.add(p[:100] + "...")
            else:
                seen.add(clean_p)
                
        if duplicates:
            print("Potential duplicate paragraphs found:")
            for d in duplicates:
                print(f"  - {d}")

analyze_files()

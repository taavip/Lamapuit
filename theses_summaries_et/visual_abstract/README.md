# Visuaalse kokkuvõtte generaator (Visual Abstract Generator)

See tööriist loeb LaTeX lähtefailist metadata ja loob modifitseeritava `visual_abstract.svg` faili, millest on lihtne eksportida PDF või PNG.

## Nõuded
- Python 3.10+
- Vajalikud paketid (vt `requirements.txt`)
- Kui soovid automaatselt eksportida PNG/PDF faile, on vajalik Cairo raamatukogu süsteemis:
  - Ubuntu/Debian: `sudo apt install libcairo2`
  - MacOS: `brew install cairo`

## Kasutamine

1. Loo virtuaalrakendus ja installi sõltuvused (soovituslikult `uv` abil):
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

2. Käivita skript:
   ```bash
   python generate_visual_abstract.py
   ```

3. Tulemusena genereeritakse `visual_abstract.svg` fail (kuhu lisatakse otse pildid base64 formaadis).

4. **Muutmine:** Ava loodud `visual_abstract.svg` programmiga nagu **Inkscape**, Adobe Illustrator või lausa VS Code, et liigutada tekste, muuta fonte või teha muid disainimuutusi.

5. **Lõplik konverteerimine:** Skript üritab luua kohe `output/visual_abstract.png` ja `output/visual_abstract.pdf` failid. Kui see ebaõnnestub puuduva Cairo raamatukogu tõttu, saad kujunduse otse Inkscape'ist sobivasse formaati eksportida.

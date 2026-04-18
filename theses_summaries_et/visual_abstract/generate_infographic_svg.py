import base64
import os

def img_to_b64(filepath):
    if not os.path.exists(filepath): return ""
    with open(filepath, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")

class SGVBuilder:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.elements = []
        self.defs = []

    def rect(self, x, y, w, h, fill="#ffffff", rx=0, stroke="none", sw=0):
        self.elements.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" rx="{rx}" stroke="{stroke}" stroke-width="{sw}" />')

    def text(self, x, y, text, font_size, fill="#000", anchor="start", weight="normal"):
        # simple multiline support by splitting on |
        lines = text.split('|')
        if len(lines) == 1:
            self.elements.append(f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{font_size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{text}</text>')
        else:
            t = f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{font_size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">'
            for i, line in enumerate(lines):
                dy = 0 if i == 0 else font_size * 1.2
                t += f'<tspan x="{x}" dy="{dy}">{line}</tspan>'
            t += '</text>'
            self.elements.append(t)

    def image(self, x, y, w, h, filepath):
        b64 = img_to_b64(filepath)
        if b64:
            self.elements.append(f'<image href="{b64}" x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid meet" />')
        else:
            # Placeholder if image missing
            self.rect(x, y, w, h, fill="#f0f0f0", stroke="#ccc", sw=2)
            self.text(x + w/2, y + h/2, "Image", 20, fill="#aaa", anchor="middle")

    def line(self, x1, y1, x2, y2, stroke="#ccc", sw=2, dash=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.elements.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}" {d}/>')

    def build(self, output_path):
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" width="{self.width}" height="{self.height}" style="background-color: white;">'
        svg += "".join(self.elements)
        svg += '</svg>'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg)

def build_infographic():
    svg = SGVBuilder(800, 1200)

    # Colours mapped from the original image
    c_bg_light = "#f0f8ff"
    c_bg_panel = "#e6f2fa"
    c_header_bg = "#4782b8"
    c_text_dark = "#1a3a54"
    
    # 0. HEADER GRAPHIC & TITLE
    svg.image(0, 0, 800, 180, "assets/crops/header_trees.png")
    # Title override text
    svg.rect(180, 20, 440, 110, fill="rgba(255,255,255,0.85)", rx=10)
    svg.text(400, 55, "ÜLE-EESTILINE", 28, fill=c_text_dark, anchor="middle", weight="bold")
    svg.text(400, 85, "LAMAPUIDU HINDAMINE", 32, fill=c_text_dark, anchor="middle", weight="bold")
    svg.text(400, 115, "KAUGSEIRE ABIL", 28, fill=c_text_dark, anchor="middle", weight="bold")

    # 1. TÖÖ EESMÄRK
    svg.rect(0, 150, 800, 180, fill="white")
    svg.rect(30, 155, 250, 30, rx=5, fill=c_header_bg)
    svg.text(45, 177, "TÖÖ EESMÄRK", 16, fill="white", weight="bold")
    svg.rect(30, 185, 740, 120, rx=10, fill=c_bg_panel)
    svg.image(40, 195, 150, 100, "assets/crops/icon_target.png")
    svg.text(210, 240, "Automaatne lamapuidu tuvastamine ja kvantifitseerimine üle Eesti,|kasutades LiDAR-andmeid", 20, fill=c_text_dark)

    # 2. ANDMESTIK JA METOODIKA
    svg.rect(0, 310, 800, 250, fill="white")
    svg.rect(30, 315, 300, 30, rx=5, fill=c_header_bg)
    svg.text(45, 337, "ANDMESTIK JA METOODIKA", 16, fill="white", weight="bold")
    svg.rect(30, 345, 740, 200, rx=10, fill=c_bg_panel)
    
    # Cols in Metoodika
    svg.rect(40, 355, 230, 30, fill=c_header_bg)
    svg.text(155, 376, "LIDAR PUNKTIPILV", 16, fill="white", anchor="middle", weight="bold")
    svg.image(40, 395, 230, 120, "assets/crops/icon_lidar.png")
    svg.rect(40, 515, 230, 25, fill="white")
    svg.text(155, 532, "Hõre ja tihe andmestik", 14, fill=c_text_dark, anchor="middle")
    
    svg.line(280, 360, 280, 520, stroke="#b0c4de", dash="5,5")

    svg.rect(285, 355, 230, 30, fill=c_header_bg)
    svg.text(400, 376, "TAIMKATTE KÕRGUSMUDEL", 15, fill="white", anchor="middle", weight="bold")
    svg.image(285, 395, 230, 120, "assets/crops/icon_chm.png")
    svg.rect(285, 515, 230, 25, fill="white")
    svg.text(400, 532, "0-1,3 m kõrgusel maapinnast", 14, fill=c_text_dark, anchor="middle")

    svg.line(525, 360, 525, 520, stroke="#b0c4de", dash="5,5")

    svg.rect(530, 355, 230, 30, fill=c_header_bg)
    svg.text(645, 376, "MASINNÄGEMISE MUDELID", 14, fill="white", anchor="middle", weight="bold")
    svg.image(530, 395, 230, 120, "assets/crops/icon_ml.png")
    svg.rect(530, 515, 230, 25, fill="white")
    svg.text(645, 532, "Objekti tuvastus ja grupeerimine", 14, fill=c_text_dark, anchor="middle")

    # 3. TULEMUSED
    svg.rect(0, 560, 800, 230, fill="white")
    svg.rect(30, 565, 150, 30, rx=5, fill=c_header_bg)
    svg.text(45, 587, "TULEMUSED", 16, fill="white", weight="bold")
    svg.rect(30, 595, 740, 180, rx=10, fill=c_bg_panel)

    svg.image(50, 605, 200, 90, "assets/crops/icon_res_lidar.png")
    svg.text(150, 715, "LiDAR Kontrollalad", 14, anchor="middle", fill=c_text_dark, weight="bold")

    svg.image(300, 605, 200, 90, "assets/crops/icon_res_smi.png")
    svg.text(400, 715, "SMI Proovitükid", 14, anchor="middle", fill=c_text_dark, weight="bold")

    svg.image(550, 605, 200, 90, "assets/crops/icon_res_metsa.png")
    svg.text(650, 715, "Metsaregistri Andmed", 14, anchor="middle", fill=c_text_dark, weight="bold")

    svg.text(320, 735, "✔ Lamapuidu jaotuskaardid|✔ Mahuhinnangud|✔ Ruumilised mustrid", 16, fill=c_text_dark)

    # 4. RAKENDUSED
    svg.rect(0, 790, 800, 250, fill="white")
    svg.rect(30, 795, 180, 30, rx=5, fill=c_header_bg)
    svg.text(45, 817, "RAKENDUSED", 16, fill="white", weight="bold")
    svg.rect(30, 825, 740, 205, rx=10, fill=c_bg_panel)

    svg.image(50, 835, 200, 90, "assets/crops/icon_app_carbon.png")
    svg.rect(50, 930, 200, 25, fill="white")
    svg.text(150, 948, "Süsinikuvaru Hinnang", 14, anchor="middle", fill=c_text_dark)

    svg.image(300, 835, 200, 90, "assets/crops/icon_app_nature.png")
    svg.rect(300, 930, 200, 25, fill="white")
    svg.text(400, 948, "Looduse Taastamine", 14, anchor="middle", fill=c_text_dark)

    svg.image(550, 835, 200, 90, "assets/crops/icon_app_lulucf.png")
    svg.rect(550, 930, 200, 25, fill="white")
    svg.text(650, 948, "LULUCF Aruandlus", 14, anchor="middle", fill=c_text_dark)

    svg.text(320, 975, "✔ Metsade seisundi hindamine|✔ Kliimapoliitika|✔ Metsanduslikud planeeringud", 16, fill=c_text_dark)

    svg.build("output/editable_infographic.svg")
    print("Murtud/Loodud info-graafika SVG: output/editable_infographic.svg")

if __name__ == "__main__":
    build_infographic()

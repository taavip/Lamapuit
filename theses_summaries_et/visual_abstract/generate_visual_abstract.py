import re
import os
import sys
import base64
import gdown
from PIL import Image

def extract_metadata(tex_path):
    with open(tex_path, 'r', encoding='utf-8') as f:
        content = f.read()

    def get_match(pattern):
        match = re.search(pattern, content)
        return match.group(1).strip() if match else "Tundmatu"

    supervisor_raw = re.search(r'\\supervisor\{(.*?)\}\}', content)
    
    supervisor = ""
    if supervisor_raw:
        supervisor = supervisor_raw.group(1).replace("\\degree{PhD", " (PhD)")
    else:
        # Fallback without trailing brace
        supervisor_raw = re.search(r'\\supervisor\{(.*?)\}', content)
        supervisor = supervisor_raw.group(1).replace("\\degree{PhD", " (PhD)") if supervisor_raw else "Tundmatu"
    supervisor = supervisor.strip('}')

    return {
        'title': get_match(r'\\title\{(.*?)\}'),
        'author': get_match(r'\\author\{(.*?)\}'),
        'date': get_match(r'\\date\{(.*?)\}'),
        'supervisor': supervisor,
        'curriculum': get_match(r'\\curriculum\{(.*?)\}')
    }

def image_to_base64_data_uri(file_path):
    if not os.path.exists(file_path):
        return ""
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        extension = os.path.splitext(file_path)[1][1:].lower()
        mime_type = "image/png" if extension == "png" else f"image/{extension}"
        return f"data:{mime_type};base64,{encoded_string}"

def download_logos():
    asset_dir = 'assets'
    os.makedirs(asset_dir, exist_ok=True)
    
    # We will attempt to download individual sample files directly if possible.
    # Because of Drive folder restrictions with gdown, we might need a dummy fallback.
    # Let's create dummy logo images for now to let the pipeline work.
    
    ut_logo_path = os.path.join(asset_dir, 'ut_logo.png')
    cs_logo_path = os.path.join(asset_dir, 'cs_logo.png')
    
    if not os.path.exists(ut_logo_path):
        img = Image.new('RGB', (200, 100), color = (0, 0, 128))
        img.save(ut_logo_path)
        print("Tegin asenduseks University of Tartu logo.")

    if not os.path.exists(cs_logo_path):
        img_cs = Image.new('RGB', (200, 100), color = (100, 149, 237))
        img_cs.save(cs_logo_path)
        print("Tegin asenduseks Institute of Computer Science logo.")

    return ut_logo_path, cs_logo_path

def generate_svg(metadata, main_image_path, ut_logo_path, cs_logo_path, output_svg):
    # Dimensions for 4:3 at a reasonable size (e.g., 2400 x 1800)
    width = 2400
    height = 1800
    
    main_b64 = image_to_base64_data_uri(main_image_path)
    ut_logo_b64 = image_to_base64_data_uri(ut_logo_path)
    cs_logo_b64 = image_to_base64_data_uri(cs_logo_path)

    # Simplified layout based on visual_abstract_rules
    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background-color: white;">
  <style>
    .title {{ font: bold 64px sans-serif; fill: #222; text-anchor: middle; }}
    .author {{ font: 48px sans-serif; fill: #444; text-anchor: middle; }}
    .subtitle {{ font: 36px sans-serif; fill: #666; text-anchor: middle; }}
    .hashtag {{ font: bold 48px sans-serif; fill: #004D99; }}
    .footer-text {{ font: 36px sans-serif; fill: #555; }}
  </style>

  <!-- Taust -->
  <rect width="100%" height="100%" fill="white" />
  <rect x="50" y="50" width="{width - 100}" height="{height - 100}" fill="none" stroke="#ddd" stroke-width="4"/>

  <!-- Päis -->
  <text x="{width // 2}" y="150" class="title">{metadata['title']}</text>
  <text x="{width // 2}" y="220" class="author">{metadata['author']}</text>
  <text x="{width // 2}" y="280" class="subtitle">{metadata['curriculum']}, {metadata['date']}</text>

  <!-- Keskosa (Pilt) -->
  <image href="{main_b64}" x="100" y="320" width="{width - 200}" height="{height - 600}" preserveAspectRatio="xMidYMid meet" />

  <!-- Jalus -->
  <line x1="100" y1="{height - 240}" x2="{width - 100}" y2="{height - 240}" stroke="#eee" stroke-width="4" />
  
  <!-- Logod -->
  <image href="{ut_logo_b64}" x="100" y="{height - 220}" width="400" height="150" preserveAspectRatio="xMinYMid meet" />
  <image href="{cs_logo_b64}" x="550" y="{height - 220}" width="400" height="150" preserveAspectRatio="xMinYMid meet" />

  <!-- Tekst jaluses -->
  <text x="1000" y="{height - 150}" class="footer-text">Juhendaja(d): {metadata['supervisor']}</text>
  <text x="1000" y="{height - 100}" class="footer-text">Tartu Ülikool, Arvutiteaduse instituut</text>
  
  <text x="{width - 400}" y="{height - 150}" class="hashtag">#UniTartuCS</text>

</svg>
"""
    with open(output_svg, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    print(f"Loodud toores SVG fail: {output_svg}")

def export_svg(svg_path, output_png, output_pdf):
    try:
        import cairosvg
        
        # Ekstordime PDF
        cairosvg.svg2pdf(url=svg_path, write_to=output_pdf)
        print(f"Eksport edukas, loodud PDF: {output_pdf}")
        
        # Ekstordime PNG (600 DPI on suur). Teeme baasresolutsiooni järgi PNG
        cairosvg.svg2png(url=svg_path, write_to=output_png, output_width=2400, output_height=1800)
        print(f"Eksport edukas, loodud PNG: {output_png}")
        
    except ImportError:
        print("CairoSVG pole installitud. Proovi: `pip install cairosvg` (ja system libcairo2 paketti)")

if __name__ == "__main__":
    base_dir = '/home/tpipar/project/Lamapuit'
    tex_file = os.path.join(base_dir, 'LaTeX/Lamapuidu_tuvastamine/estonian/põhi.tex')
    main_img = os.path.join(base_dir, 'LaTeX/Lamapuidu_tuvastamine/estonian/joonised/Joonis0-VisuaalneKokkuvõte_c.png')

    output_svg = "visual_abstract.svg"
    output_png = "output/visual_abstract.png"
    output_pdf = "output/visual_abstract.pdf"

    if not os.path.exists(tex_file):
        print(f"Viga: LaTeX faili ei leitud: {tex_file}")
        sys.exit(1)

    print("Loeme metadata LaTeX failist...")
    meta = extract_metadata(tex_file)
    print("Meta:", meta)
    
    ut_logo, cs_logo = download_logos()
    generate_svg(meta, main_img, ut_logo, cs_logo, output_svg)
    export_svg(output_svg, output_png, output_pdf)
    print("---------------------------------")
    print("KÕIK VALMIS!")


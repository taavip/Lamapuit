#!/usr/bin/env python3
"""Post-process translated_elements.json with glossary fixes and regenerate combined text."""
import json
import os
import re
import argparse


GLOSSARY = {
    # Finnish -> preferred English
    r"\blahopuu\b": "deadwood",
    r"\blahowood\b": "deadwood",
    r"\bdecaywood\b": "deadwood",
    r"\bdecayed wood\b": "deadwood",
    r"\bdeadwood\b": "deadwood",
    r"\blaserkeilausaineisto\b": "LiDAR data",
    r"\blaser bowling material\b": "LiDAR data",
    r"\blaser bowling\b": "LiDAR",
    r"\blaser\s+scanning data\b": "LiDAR data",
    r"\bHough conversion\b": "Hough transform",
    r"\bHough conversions\b": "Hough transform",
    r"\bHough conversion predictions\b": "Hough transform predictions",
    r"\bKnn\b": "KNN",
    r"\bkonvoluutioneuroverkko\b": "convolutional neural network",
    r"\bconvolution euro network\b": "convolutional neural network",
    r"\bconvoluted neural networks\b": "convolutional neural networks",
    r"\beuro\b": "neuro",
    r"\bpoint close data\b": "point cloud data",
    r"\bpoint close\b": "point cloud",
    r"\bpointcloud\b": "point cloud",
    r"\blaser bowling\b": "LiDAR",
    r"\bmaalaho\b": "ground decay",
    r"\bmaalahoa\b": "ground decay",
    r"\blahopuun esiintymätiheys\b": "deadwood occurrence density",
}


def apply_glossary(text: str) -> str:
    s = text
    for pat, repl in GLOSSARY.items():
        s = re.sub(pat, repl, s, flags=re.IGNORECASE)
    # tidy whitespace
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def preserve_names(original: str, translated: str) -> str:
    # if original looks like a person name (contains Finnish diacritics or two capitalized words), keep it
    if re.search(r"[A-ZÄÖÅ][a-zäöå]+\s+[A-ZÄÖÅ][a-zäöå]+", original):
        return original
    return translated


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-json", default="output/translated_pdf/translated_elements.json")
    p.add_argument("--out-json", default="output/translated_pdf/translated_elements_postprocessed.json")
    p.add_argument("--out-txt", default="output/translated_pdf/translated_combined_postprocessed.txt")
    args = p.parse_args()

    with open(args.in_json, "r", encoding="utf-8") as f:
        elements = json.load(f)

    for e in elements:
        orig = e.get("original_text", "")
        tr = e.get("translated_text", "")
        # preserve names
        tr = preserve_names(orig, tr)
        # apply glossary fixes
        tr = apply_glossary(tr)
        e["translated_text_post"] = tr

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(elements, f, ensure_ascii=False, indent=2)

    # write combined cleaned text
    with open(args.out_txt, "w", encoding="utf-8") as f:
        cur_page = None
        for e in elements:
            if e["page"] != cur_page:
                cur_page = e["page"]
                f.write(f"\n\n=== Page {cur_page} ===\n\n")
            f.write(e.get("translated_text_post", e.get("translated_text", "")).strip() + "\n\n")

    print("Wrote:", args.out_json)
    print("Wrote:", args.out_txt)


if __name__ == "__main__":
    main()

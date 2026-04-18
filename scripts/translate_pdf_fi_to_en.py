#!/usr/bin/env python3
"""Download a Finnish PDF, extract text elements, and translate each element to English.

Outputs:
- JSON: list of elements with page, bbox, original_text, translated_text
- TXT: combined translated text (page-separated)

Usage example:
  python scripts/translate_pdf_fi_to_en.py \
    --url "https://www.theseus.fi/bitstream/handle/10024/878233/Vakimies_Pirkko.pdf?sequence=2" \
    --out-dir output/translated_pdf
"""
import argparse
import json
import os
from typing import List

import requests
import fitz  # PyMuPDF
from tqdm import tqdm

from transformers import MarianMTModel, MarianTokenizer
import torch


def download_pdf(url: str, dest: str) -> str:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(1024 * 64):
            if chunk:
                f.write(chunk)
    return dest


def load_translation_model(model_name: str = "Helsinki-NLP/opus-mt-fi-en"):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def translate_texts(texts: List[str], tokenizer, model, device, batch_size: int = 16) -> List[str]:
    translations = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            translated = model.generate(**inputs, max_length=512)
        outs = tokenizer.batch_decode(translated, skip_special_tokens=True)
        translations.extend(outs)
    return translations


def extract_elements(pdf_path: str):
    doc = fitz.open(pdf_path)
    elements = []
    for pno in range(len(doc)):
        page = doc[pno]
        data = page.get_text("dict")
        for block in data.get("blocks", []):
            bbox = block.get("bbox")
            # gather all spans text in this block
            block_text_parts = []
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    txt = span.get("text", "").strip()
                    if txt:
                        block_text_parts.append(txt)
            if not block_text_parts:
                continue
            orig_text = "\n".join(block_text_parts)
            elements.append({"page": pno + 1, "bbox": bbox, "original_text": orig_text})
    return elements


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="PDF URL to download")
    p.add_argument("--pdf-path", default="/tmp/input.pdf", help="Local path to save PDF")
    p.add_argument("--out-dir", default="output/translated_pdf", help="Output directory")
    p.add_argument("--model", default="Helsinki-NLP/opus-mt-fi-en", help="HuggingFace model name")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Downloading PDF...")
    pdf_path = download_pdf(args.url, args.pdf_path)
    print("Extracting text elements from PDF...")
    elements = extract_elements(pdf_path)
    texts = [e["original_text"] for e in elements]

    if not texts:
        print("No text elements found in PDF.")
        return

    print(f"Loading translation model {args.model}...")
    tokenizer, model, device = load_translation_model(args.model)

    print(f"Translating {len(texts)} elements in batches of {args.batch_size}...")
    translated = translate_texts(texts, tokenizer, model, device, batch_size=args.batch_size)

    for e, t in zip(elements, translated):
        e["translated_text"] = t

    # Save JSON
    out_json = os.path.join(args.out_dir, "translated_elements.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(elements, f, ensure_ascii=False, indent=2)

    # Save combined translated text with page separators
    out_txt = os.path.join(args.out_dir, "translated_combined.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        cur_page = None
        for e in elements:
            if e["page"] != cur_page:
                cur_page = e["page"]
                f.write(f"\n\n=== Page {cur_page} ===\n\n")
            f.write(e["translated_text"].strip() + "\n\n")

    print("Done.")
    print("JSON:", out_json)
    print("Combined TXT:", out_txt)


if __name__ == "__main__":
    main()

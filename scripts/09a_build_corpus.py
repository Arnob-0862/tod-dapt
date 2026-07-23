"""
09a_build_corpus.py
===================
Extract raw text from all PDFs for DAPT.

Cleaning: MINIMAL (following Gururangan et al. 2020)
  1. Deduplication — same sentence once only
  2. That's it. Model handles the rest.

Output: E:\TOD_DAPT\corpus\dapt_corpus.txt
"""

import os, re
import fitz
import nltk

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import sent_tokenize

PDF_FOLDER = r"E:\TOD_DAPT\papers"
OUT_CORPUS = r"E:\TOD_DAPT\corpus\dapt_corpus.txt"
OUT_STATS  = r"E:\TOD_DAPT\corpus\corpus_stats.txt"


def extract_text(pdf_path: str) -> str:
    """Extract full raw text from PDF."""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
        # Fix hyphenated line breaks and ligatures only
        text = re.sub(r'-\n', '', text)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
    except Exception as e:
        print(f"  Error {os.path.basename(pdf_path)}: {e}")
    return text


def main():
    print("=" * 55)
    print("  TOD DAPT — CORPUS BUILDER")
    print("  Minimal cleaning: deduplication only")
    print("=" * 55)

    pdfs = sorted([f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')])
    print(f"\n  PDFs: {len(pdfs)}")

    all_sents = []
    seen = set()

    for i, fname in enumerate(pdfs, 1):
        raw_text = extract_text(os.path.join(PDF_FOLDER, fname))
        sents = sent_tokenize(raw_text)

        for s in sents:
            s = s.strip()
            if not s:
                continue
            key = s.lower()[:120]
            if key not in seen:
                seen.add(key)
                all_sents.append(s)

        if i % 30 == 0 or i == len(pdfs):
            print(f"  [{i:3d}/{len(pdfs)}]  sentences: {len(all_sents):,}")

    total_words = sum(len(s.split()) for s in all_sents)

    print(f"\n  Total sentences : {len(all_sents):,}")
    print(f"  Total words     : {total_words:,}")

    with open(OUT_CORPUS, 'w', encoding='utf-8') as f:
        for s in all_sents:
            f.write(s + '\n')

    size_mb = round(os.path.getsize(OUT_CORPUS) / 1e6, 1)
    print(f"\n✅ Corpus: {OUT_CORPUS}  ({size_mb} MB)")

    stats = '\n'.join([
        "TOD DAPT Corpus Statistics",
        "==========================",
        f"PDFs         : {len(pdfs)}",
        f"Sentences    : {len(all_sents):,}",
        f"Words        : {total_words:,}",
        f"Avg sent len : {total_words/max(len(all_sents),1):.1f} words",
        f"File size    : {size_mb} MB",
        "",
        "Cleaning: deduplication only (Gururangan et al. 2020)",
    ])
    with open(OUT_STATS, 'w', encoding='utf-8') as f:
        f.write(stats)
    print(f"✅ Stats: {OUT_STATS}")


if __name__ == "__main__":
    main()

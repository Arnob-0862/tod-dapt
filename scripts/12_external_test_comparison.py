"""
12_external_test_comparison.py
==============================
External validation: SciBERT vs TOD-SciBERT
on 50 UNSEEN test papers.

Proves model generalizes beyond training corpus.

Output: E:\TOD_DAPT\validation\
"""

import os, re, torch
import numpy as np
import pandas as pd
import fitz
import nltk

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel

# ── Paths ──────────────────────────────────────────────────
SCIBERT_PATH     = "allenai/scibert_scivocab_uncased"
TOD_SCIBERT_PATH = r"E:\TOD_DAPT\model\tod_scibert"
TEST_FOLDER      = r"E:\TOD_DAPT\Test"
OUTPUT_DIR       = r"E:\TOD_DAPT\validation"
TOP_K = 15

# ── Same aspects as 09c ────────────────────────────────────
ASPECTS = {
    "Land Use Mix":   "mixed land use transit oriented development",
    "Walkability":    "pedestrian walkability street design sidewalk",
    "Transit Access": "public transit station access bus rail frequency",
    "Density":        "residential commercial density floor area ratio",
    "Parking":        "parking reduction car dependency vehicle ownership",
    "Cycling":        "cycling bicycle infrastructure network lane",
    "Green Space":    "green space park open space environment",
    "Affordability":  "affordable housing income socioeconomic residents",
    "Station Area":   "station area development TOD zone catchment",
    "Urban Design":   "urban design placemaking street quality aesthetics",
}


def extract_sentences(pdf_path):
    try:
        doc  = fitz.open(pdf_path)
        text = "".join(page.get_text("text") + "\n" for page in doc)
        doc.close()
        text = re.sub(r'-\n', '', text)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
        return [s.strip() for s in sent_tokenize(text)
                if 30 <= len(s.strip()) <= 500]
    except:
        return []


def mean_pool(token_emb, attn_mask):
    mask = attn_mask.unsqueeze(-1).expand(token_emb.size()).float()
    return (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def embed(model, tokenizer, texts, device, batch_size=64):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc   = tokenizer(batch, padding=True, truncation=True,
                          max_length=256, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**enc)
        emb = mean_pool(out.last_hidden_state, enc['attention_mask'])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 55)
    print("  External Validation (50 Unseen Papers)")
    print("  SciBERT  vs  TOD-SciBERT")
    print("=" * 55)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")

    print("\n  Loading SciBERT...")
    sci_tok = AutoTokenizer.from_pretrained(SCIBERT_PATH)
    sci_mod = AutoModel.from_pretrained(SCIBERT_PATH).to(device).eval()

    print("  Loading TOD-SciBERT...")
    tod_tok = AutoTokenizer.from_pretrained(TOD_SCIBERT_PATH)
    tod_mod = AutoModel.from_pretrained(TOD_SCIBERT_PATH).to(device).eval()

    # Pre-embed aspect queries
    aspect_names = list(ASPECTS.keys())
    aspect_texts = list(ASPECTS.values())
    sci_q  = embed(sci_mod, sci_tok, aspect_texts, device)
    tod_q  = embed(tod_mod, tod_tok, aspect_texts, device)

    pdfs = sorted([f for f in os.listdir(TEST_FOLDER) if f.endswith('.pdf')])
    print(f"  Test papers: {len(pdfs)}")
    print(f"  Aspects: {len(ASPECTS)}\n")

    results = []

    for pi, fname in enumerate(pdfs, 1):
        print(f"  [{pi:2d}/{len(pdfs)}] {fname[:55]}")
        sents = extract_sentences(os.path.join(TEST_FOLDER, fname))
        if len(sents) < TOP_K:
            continue

        sci_s = embed(sci_mod, sci_tok, sents, device)
        tod_s = embed(tod_mod, tod_tok, sents, device)

        for ai, aspect in enumerate(aspect_names):
            sci_scores = sci_s @ sci_q[ai]
            tod_scores = tod_s @ tod_q[ai]

            sci_top = np.argsort(sci_scores)[::-1][:TOP_K]
            tod_top = np.argsort(tod_scores)[::-1][:TOP_K]

            sci_avg = float(sci_scores[sci_top].mean())
            tod_avg = float(tod_scores[tod_top].mean())
            overlap = len(set(sci_top.tolist()) & set(tod_top.tolist())) / TOP_K

            results.append({
                'paper':           fname,
                'aspect':          aspect,
                'scibert_sim':     round(sci_avg, 4),
                'tod_sim':         round(tod_avg, 4),
                'sim_improvement': round(tod_avg - sci_avg, 4),
                'overlap_at_15':   round(overlap, 4),
            })

    df = pd.DataFrame(results)

    sci_mean = df['scibert_sim'].mean()
    tod_mean = df['tod_sim'].mean()
    imp_mean = df['sim_improvement'].mean()
    imp_pct  = imp_mean / sci_mean * 100
    olp_mean = df['overlap_at_15'].mean()

    print("\n" + "=" * 55)
    print("  EXTERNAL VALIDATION RESULTS")
    print("=" * 55)
    print(f"\n  {'Model':<22} {'Avg Similarity':>15}")
    print(f"  {'-'*38}")
    print(f"  {'SciBERT':<22} {sci_mean:>15.4f}")
    print(f"  {'TOD-SciBERT':<22} {tod_mean:>15.4f}")
    print(f"  {'-'*38}")
    print(f"  Improvement: {imp_mean:+.4f}  ({imp_pct:.1f}%)")
    print(f"  Overlap@15:  {olp_mean:.1%} shared sentences")

    print(f"\n  Per-aspect (test set):")
    asp = df.groupby('aspect')['sim_improvement'].mean().sort_values(ascending=False)
    for a, v in asp.items():
        print(f"    {'▲' if v>0 else '▼'} {a:<22} {v:+.4f}")

    # Compare with training results (09c)
    train_imp = 0.0560   # from 09c results
    print(f"\n  Consistency check:")
    print(f"  Training set improvement : +{train_imp:.4f} (+{train_imp/0.7046*100:.1f}%)")
    print(f"  Test set improvement     : {imp_mean:+.4f} ({imp_pct:+.1f}%)")
    consistent = abs(imp_mean - train_imp) < 0.02
    print(f"  Consistent: {'✓ YES' if consistent else '~ CLOSE'}")

    # Save
    df.to_csv(os.path.join(OUTPUT_DIR, 'external_test_comparison.csv'), index=False)

    summary = "\n".join([
        "External Validation Results",
        "===========================",
        f"Test papers      : {df['paper'].nunique()}",
        f"Aspects          : {len(ASPECTS)}",
        f"Top-K            : {TOP_K}",
        "",
        f"SciBERT avg sim  : {sci_mean:.4f}",
        f"TOD-SciBERT sim  : {tod_mean:.4f}",
        f"Improvement      : {imp_mean:+.4f} ({imp_pct:.1f}%)",
        f"Overlap@15       : {olp_mean:.1%}",
        "",
        "Training vs Test consistency:",
        f"  Training improvement: +{train_imp:.4f}",
        f"  Test improvement    : {imp_mean:+.4f}",
        f"  Consistent          : {'Yes' if consistent else 'Approximately'}",
        "",
        "Per-aspect improvement:",
        *[f"  {a}: {v:+.4f}" for a, v in asp.items()],
    ])
    with open(os.path.join(OUTPUT_DIR, 'external_validation_summary.txt'), 'w') as f:
        f.write(summary)

    print(f"\n✅ Saved: {OUTPUT_DIR}")
    print(f"   external_test_comparison.csv")
    print(f"   external_validation_summary.txt")


if __name__ == "__main__":
    main()
